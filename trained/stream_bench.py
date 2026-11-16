import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import ROOT, FROZEN_CKPT, bench, gpu_state, idle_check, load_cache, load_model, prep, temporal_from_cfg, TemporalClassifier
from mamba_step import MambaStreamer
from selective_scan import patch_model as patch_scan

D, E, N, KC = 512, 1024, 16, 4


def head_bf16(head):
    head = head.to(torch.bfloat16)
    for l in head.layers:
        l.mixer.A_log.data = l.mixer.A_log.data.float()
        l.mixer.D.data = l.mixer.D.data.float()
    return head


def set_scan(head, mode):
    for l in head.layers:
        l.mixer.scan_impl = "ref"
        l.mixer.__dict__.pop("scan", None)
    if mode == "fused":
        patch_scan(head)


def real_feats(T, batch):
    which = "frozen"
    net, interp = load_model(which)
    u8, _ = load_cache(interp, None)
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(u8), batch * T, replace=False))
    with torch.no_grad():
        outs = [net.trunk(prep(np.stack([u8[i] for i in idx[j:j + 30]]))) for j in range(0, len(idx), 30)]
    return torch.cat(outs).view(batch, T, -1).float()


@torch.no_grad()
def verify():
    res = []
    ck = torch.load(FROZEN_CKPT, map_location="cpu")
    for dtype in (torch.float32, torch.bfloat16):
        for T in (30, 300):
            head = temporal_from_cfg(ck["config"])
            head.load_state_dict(ck["model"])
            head = head.cuda().eval()
            feats = real_feats(T, 2).cuda()
            for l in head.layers:
                l.mixer.scan_impl = "ref"
            ref = head(feats)
            if dtype == torch.bfloat16:
                head = head_bf16(head)
                par_bf = head(feats.to(dtype)).float()
            st = MambaStreamer(head, batch=2)
            for t in range(T):
                out = st.forward_step(feats[:, t].to(dtype))
            r = {"dtype": str(dtype).split(".")[1], "frames": T, "max_abs_diff_vs_parallel_fp32": (out.float() - ref).abs().max().item(),
                 "logit_abs_max": ref.abs().max().item()}
            if dtype == torch.bfloat16:
                r["max_abs_diff_vs_parallel_same_dtype"] = (out.float() - par_bf).abs().max().item()
            st2 = MambaStreamer(head, batch=2, use_triton=False)
            for t in range(T):
                out2 = st2.forward_step(feats[:, t].to(dtype))
            r["max_abs_diff_torch_step_vs_triton_step"] = (out2.float() - out.float()).abs().max().item()
            res.append(r)
            print(r, flush=True)
            del head
    return res


class MHAHead(nn.Module):
    def __init__(self, heads=8):
        super().__init__()
        self.proj = nn.Linear(1280, D)
        self.norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, heads, batch_first=True)
        self.norm_f = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Linear(D, D // 2), nn.GELU(), nn.Linear(D // 2, 2))
        self.heads = heads

    def forward(self, feats, need_weights=False):
        h = self.proj(feats)
        x = self.norm(h)
        h = h + self.attn(x, x, x, need_weights=need_weights)[0]
        return self.head(self.norm_f(h).mean(1))

    def step_cached(self, feat, kcache, vcache, n):
        h = self.proj(feat)
        x = self.norm(h)
        w, b = self.attn.in_proj_weight, self.attn.in_proj_bias
        q, k, v = F.linear(x, w, b).chunk(3, -1)
        kcache[:, n] = k
        vcache[:, n] = v
        B = x.shape[0]
        sh = lambda t: t.view(B, -1, self.heads, D // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q[:, None]), sh(kcache[:, :n + 1]), sh(vcache[:, :n + 1]))
        o = self.attn.out_proj(o.transpose(1, 2).reshape(B, D))
        return self.head(self.norm_f(h + o))


def mem_of(fn, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def flops_mamba(T, layers):
    per_tok = layers * (2 * D * 2 * E + 2 * E * KC + 2 * E * 2 * N + 2 * E * E + 6 * E * N + 2 * E * D + 4 * E) + 2 * 1280 * D
    return per_tok * T, per_tok


def flops_attn_recompute(T):
    return 2 * 1280 * D * T + T * (3 * 2 * D * D + 2 * D * D) + 4 * T * T * D


def flops_attn_cached(T):
    return 2 * 1280 * D + 3 * 2 * D * D + 2 * D * D + 4 * T * D


@torch.no_grad()
def timing(args):
    from ladder import Runner, RUNGS
    ck = torch.load(FROZEN_CKPT, map_location="cpu")
    out = {"trunk": {}, "head": {}}
    for bs in (1, 30):
        r = Runner("frozen", RUNGS[args.trunk_rung], bs, args.dw)
        x = prep(load_cache("bilinear", 64)[0][:bs])
        out["trunk"][bs] = bench(lambda: r.run(x), iters=100, warmup=10)
        del r
        torch.cuda.empty_cache()
    print("trunk ms", out["trunk"], flush=True)
    base4 = temporal_from_cfg(ck["config"])
    base4.load_state_dict(ck["model"])
    base1 = TemporalClassifier(512, 1, 16, 0.1)
    for tag, hd in (("mamba4_trained", base4), ("mamba1", base1)):
        hd = head_bf16(hd.cuda().eval())
        L = len(hd.layers)
        for T in args.windows:
            feats = torch.randn(1, T, 1280, device="cuda", dtype=torch.bfloat16)
            rec = {}
            if tag == "mamba4_trained" and T <= 300 or tag == "mamba1" and T <= 300:
                set_scan(hd, "ref")
                rec["recompute_torch_scan_loop_ms"] = bench(lambda: hd(feats), iters=10 if T <= 300 else 3, warmup=2)
            set_scan(hd, "fused")
            rec["recompute_fused_scan_ms"] = bench(lambda: hd(feats), iters=50 if T <= 300 else 20, warmup=5)
            rec["recompute_fused_scan_peak_mb"] = mem_of(lambda: hd(feats))
            st = MambaStreamer(hd, batch=1)
            f1 = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
            rec["step_eager_ms"] = bench(lambda: st.forward_step(f1), iters=100, warmup=10)
            rec["step_peak_mb"] = mem_of(lambda: st.forward_step(f1))
            rec["step_state_bytes"] = st.state_bytes()
            fin = f1.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    st.forward_step(fin)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                gout = st.forward_step(fin)
            rec["step_cuda_graph_ms"] = bench(lambda: (fin.copy_(f1), g.replay()), iters=200, warmup=10)
            rec["modeled_flops_recompute"], rec["modeled_flops_step"] = flops_mamba(T, L)
            out["head"].setdefault(tag, {})[T] = rec
            print(tag, T, rec, flush=True)
            del st, g
            torch.cuda.empty_cache()
        del hd
    att = MHAHead().cuda().eval().to(torch.bfloat16)
    for T in args.windows:
        feats = torch.randn(1, T, 1280, device="cuda", dtype=torch.bfloat16)
        rec = {"recompute_sdpa_ms": bench(lambda: att(feats, False), iters=50, warmup=5),
               "recompute_sdpa_peak_mb": mem_of(lambda: att(feats, False)),
               "recompute_naive_weights_ms": bench(lambda: att(feats, True), iters=20, warmup=3),
               "recompute_naive_weights_peak_mb": mem_of(lambda: att(feats, True))}
        kc = torch.zeros(1, T, D, device="cuda", dtype=torch.bfloat16)
        vc = torch.zeros(1, T, D, device="cuda", dtype=torch.bfloat16)
        f1 = torch.randn(1, 1280, device="cuda", dtype=torch.bfloat16)
        rec["kv_cache_step_ms"] = bench(lambda: att.step_cached(f1, kc, vc, T - 1), iters=100, warmup=10)
        rec["kv_cache_step_peak_mb"] = mem_of(lambda: att.step_cached(f1, kc, vc, T - 1))
        rec["kv_cache_bytes"] = kc.numel() * 2 * 2
        rec["modeled_flops_recompute"] = flops_attn_recompute(T)
        rec["modeled_flops_kv_cache_step"] = flops_attn_cached(T)
        out["head"].setdefault("attention1", {})[T] = rec
        print("attn", T, rec, flush=True)
        del kc, vc
        torch.cuda.empty_cache()
    return out


def write_md(res):
    V, H, tr = res["verify"], res["timing"]["head"], res["timing"]["trunk"]
    ws = sorted(H["mamba4_trained"].keys(), key=int)
    L = ["# Streaming inference: recurrent Mamba step vs window recompute vs attention", "",
         f"GPU state at start: {res['gpu_state_start']}", f"Idle check: {res['idle_check']}", "",
         "## Correctness: recurrent stepping equals the parallel scan", "",
         "Trained 4-layer head, 2 sequences of real test-image features, state carried from zero. `max abs diff` is on the 2 output logits.", "",
         "| dtype | frames | max abs diff vs parallel fp32 | max abs diff vs parallel same dtype | torch step vs Triton step | max abs logit |", "|---|---|---|---|---|---|"]
    for v in V:
        L.append(f"| {v['dtype']} | {v['frames']} | {v['max_abs_diff_vs_parallel_fp32']:.2e} | {v.get('max_abs_diff_vs_parallel_same_dtype', float('nan')):.2e} | {v['max_abs_diff_torch_step_vs_triton_step']:.2e} | {v['logit_abs_max']:.2f} |")
    L += ["", f"Image trunk (R6 stack, batch 1): {tr['1']:.2f} ms per frame; batch 30: {tr['30']:.2f} ms.", "",
          "## Per-new-frame head latency (batch 1, bf16 weights, ms; window length = frames in the sliding window)", "",
          "| Window | Mamba x4 recompute (torch scan loop) | Mamba x4 recompute (fused scan) | Mamba x4 recurrent step (eager) | Mamba x4 recurrent step (CUDA graph) | Mamba x1 recompute fused | Mamba x1 step graph | Attn x1 recompute (SDPA) | Attn x1 recompute (materialized weights) | Attn x1 with KV cache |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "-"
    for T in ws:
        a, m1, at = H["mamba4_trained"][T], H["mamba1"][T], H["attention1"][T]
        L.append(f"| {T} | {f(a.get('recompute_torch_scan_loop_ms'))} | {f(a['recompute_fused_scan_ms'])} | {f(a['step_eager_ms'])} | {f(a['step_cuda_graph_ms'])} | {f(m1['recompute_fused_scan_ms'])} | {f(m1['step_cuda_graph_ms'])} | {f(at['recompute_sdpa_ms'])} | {f(at['recompute_naive_weights_ms'])} | {f(at['kv_cache_step_ms'])} |")
    L += ["", "## Memory per new frame (MB; transient = extra peak allocation while computing; state = what must persist)", "",
          "| Window | Mamba x4 recompute transient | Mamba x4 recurrent state | Mamba x4 step transient | Attn recompute SDPA transient | Attn recompute materialized transient | Attn KV cache (persistent) |", "|---|---|---|---|---|---|---|"]
    for T in ws:
        a, at = H["mamba4_trained"][T], H["attention1"][T]
        L.append(f"| {T} | {a['recompute_fused_scan_peak_mb']:.1f} | {a['step_state_bytes'] / 2**20:.3f} | {a['step_peak_mb']:.2f} | {at['recompute_sdpa_peak_mb']:.1f} | {at['recompute_naive_weights_peak_mb']:.1f} | {at['kv_cache_bytes'] / 2**20:.2f} |")
    L += ["", "## Modeled work per new frame (FLOPs, analytic; head only, trunk excluded)", "",
          "| Window | Mamba x4 recompute | Mamba x4 step | Attn x1 recompute | Attn x1 KV-cache step |", "|---|---|---|---|---|"]
    for T in ws:
        a, at = H["mamba4_trained"][T], H["attention1"][T]
        L.append(f"| {T} | {a['modeled_flops_recompute']:.3g} | {a['modeled_flops_step']:.3g} | {at['modeled_flops_recompute']:.3g} | {at['modeled_flops_kv_cache_step']:.3g} |")
    if "window30_total" in res["timing"]:
        w = res["timing"]["window30_total"]
        L += ["", "## 30-frame sliding window, total per new frame (trunk + head, ms)", "",
              "| Strategy | ms/frame | frames/s |", "|---|---|---|"]
        for k, v in w.items():
            L.append(f"| {k} | {v:.2f} | {1000 / v:.0f} |")
    (ROOT / "results" / "streaming.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-idle-check", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=1800)
    ap.add_argument("--windows", type=int, nargs="+", default=[30, 300, 3000])
    ap.add_argument("--trunk-rung", default="R6")
    ap.add_argument("--dw", default="auto")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "streaming.json"))
    a = ap.parse_args()
    res = {"verify": None}
    res["verify"] = verify() if not a.skip_verify else json.loads(Path(a.out).read_text())["verify"]
    res["idle_check"] = None if a.no_idle_check else idle_check(a.idle_timeout)
    if res["idle_check"] is not None and not res["idle_check"]["passed"]:
        print("GPU not idle", res["idle_check"])
        return
    res["gpu_state_start"] = gpu_state()
    t = timing(a)
    h = t["head"]
    w30 = {}
    if 30 in h["mamba4_trained"] or "30" in h["mamba4_trained"]:
        k = 30
        tr1, tr30 = t["trunk"][1], t["trunk"][30]
        w30["(a) recompute everything: trunk on 30 frames + 4-layer Mamba, fused scan"] = tr30 + h["mamba4_trained"][k]["recompute_fused_scan_ms"]
        w30["(a2) cache old features: trunk on 1 frame + 4-layer Mamba over 30, fused scan"] = tr1 + h["mamba4_trained"][k]["recompute_fused_scan_ms"]
        w30["(b) recurrent step: trunk on 1 frame + eager step"] = tr1 + h["mamba4_trained"][k]["step_eager_ms"]
        w30["(b2) recurrent step: trunk on 1 frame + CUDA-graph step"] = tr1 + h["mamba4_trained"][k]["step_cuda_graph_ms"]
    t["window30_total"] = w30
    res["timing"] = t
    res["gpu_state_end"] = gpu_state()
    res["note"] = "Frame tensors are random but fixed for timing; outputs match the parallel scan, so accuracy is unchanged."
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    write_md(json.loads(json.dumps(res, default=str)))
    print((ROOT / "results" / "streaming.md").read_text())


if __name__ == "__main__":
    main()
