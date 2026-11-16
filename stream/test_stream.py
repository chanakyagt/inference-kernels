import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stream"))
from pipeline import (StreamPipeline, TestImages, load_parts, make_original, normalize, run_original, sample_images,
                      test_items, to_dtype)
from fused_mamba_step import FusedMambaStack
from fused_tail_proj import FusedTailProj, tail_reference

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
OUT = ROOT / "results" / "stream_fused_correctness.json"
from pipeline import CKPTS
NAMES = tuple(n for n in ("diloco", "finetuned") if Path(CKPTS[n]).exists())


def rel(err, ref):
    return float(err / max(ref.abs().max().item(), 1e-12))


def merge(key, val):
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1))


@torch.no_grad()
def test_kernel_a():
    res = {}
    for name in NAMES:
        features, temporal, resize = load_parts(name)
        x, _ = sample_images(resize, 64)
        trunk32 = features[:8].float()
        fmap = trunk32(x)
        tail = features[8]
        ref = tail_reference(fmap, tail[0], tail[1], temporal.proj)
        pool_ref = F.silu(tail[1](tail[0](fmap))).mean((2, 3))
        import copy
        tail_b = copy.deepcopy(tail).to(torch.bfloat16)
        proj_b = copy.deepcopy(temporal.proj).to(torch.bfloat16)
        torch_bf16 = tail_reference(fmap.to(torch.bfloat16), tail_b[0], tail_b[1], proj_b).float()
        r = {"torch_bf16_vs_fp32": {"max_abs": (torch_bf16 - ref).abs().max().item(), "rel_to_max": rel((torch_bf16 - ref).abs().max(), ref)},
             "ref_max_abs": ref.abs().max().item()}
        for dtype, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            for mode in ("two", "atomic"):
                for B in (1, 8, 64):
                    k = FusedTailProj(tail[0], tail[1], temporal.proj, dtype, mode=mode)
                    out = k(fmap[:B].to(dtype))
                    e = (out - ref[:B]).abs().max()
                    r[f"{dn}_{mode}_B{B}"] = {"max_abs": e.item(), "rel_to_max": rel(e, ref[:B])}
                    if mode == "two" and B == 8:
                        p = k.pooled(fmap[:B].to(dtype))
                        r[f"{dn}_pooled_B8"] = {"max_abs": (p - pool_ref[:B]).abs().max().item(), "rel_to_max": rel((p - pool_ref[:B]).abs().max(), pool_ref[:B])}
        cl = fmap[:8].contiguous(memory_format=torch.channels_last)
        k = FusedTailProj(tail[0], tail[1], temporal.proj, torch.float32)
        r["fp32_channels_last_B8"] = {"max_abs": (k(cl) - ref[:8]).abs().max().item()}
        res[name] = r
        print(name, "A", json.dumps({k: v for k, v in r.items() if "B8" in k or "torch" in k}), flush=True)
    merge("kernel_a", res)


def step_stack(stack, x0s, store_hidden=True):
    T = x0s.shape[1]
    hid, lg = [], []
    for t in range(T):
        stack.xbuf[0].copy_(x0s[:, t])
        stack.step_from_buf0()
        hid.append(stack.last_hidden().clone())
        lg.append(stack.logits.clone())
    return torch.stack(hid, 1), torch.stack(lg, 1)


@torch.no_grad()
def full_reference(temporal, x0s, scan):
    for L in temporal.layers:
        L.mixer.scan_impl = scan
    h = x0s
    for L in temporal.layers:
        h = L(h)
    n = temporal.norm_f(h)
    cm = n.cumsum(1) / torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)
    return h, temporal.head(cm)


@torch.no_grad()
def test_kernel_b():
    res = {}
    for name in NAMES:
        features, temporal, resize = load_parts(name)
        items = test_items()
        import numpy as np
        idx = np.random.default_rng(1).choice(len(items), 3000, replace=False)
        feats = []
        loader = torch.utils.data.DataLoader(TestImages([items[i] for i in idx], resize), batch_size=100, num_workers=8)
        for u8, _, _ in loader:
            feats.append(features(normalize(u8)).mean((2, 3)))
        feats = torch.cat(feats)
        x0all = temporal.proj(feats)
        r = {}
        for T in (30, 300, 3000):
            x0s = x0all[:T][None]
            ref_h, ref_l = full_reference(temporal, x0s, "auto")
            ref_h2, ref_l2 = full_reference(temporal, x0s, "ref")
            r[f"T{T}_triton_scan_vs_loop_scan"] = {"hidden_max_abs": (ref_h - ref_h2).abs().max().item()}
            for dtype, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
                for fo in (True, False):
                    st = FusedMambaStack(temporal, 1, dtype, fuse_out=fo)
                    hid, lg = step_stack(st, x0s)
                    eh = (hid - ref_h).abs().amax(dim=(0, 2))
                    el = (lg - ref_l).abs().amax(dim=(0, 2))
                    r[f"T{T}_{dn}_fuseout{int(fo)}"] = {
                        "hidden_max_abs": eh.max().item(), "hidden_ref_max_abs": ref_h.abs().max().item(),
                        "hidden_max_rel": float(eh.max() / ref_h.abs().max()),
                        "logit_max_abs": el.max().item(), "logit_ref_max_abs": ref_l.abs().max().item(),
                        "worst_step_hidden": int(eh.argmax()), "step_hidden_err_at_last": eh[-1].item(),
                        "prob_fake_max_abs": (lg.softmax(-1)[..., 1] - ref_l.softmax(-1)[..., 1]).abs().max().item()}
            tb = to_dtype(temporal, torch.bfloat16)
            hb, lb = full_reference(tb, x0s.to(torch.bfloat16), "ref")
            r[f"T{T}_torch_bf16_fullseq_vs_fp32"] = {
                "hidden_max_abs": (hb.float() - ref_h).abs().max().item(), "logit_max_abs": (lb.float() - ref_l).abs().max().item()}
            print(name, "B", T, json.dumps({k: v for k, v in r.items() if k.startswith(f"T{T}_")}), flush=True)
        B = 8
        xb = torch.stack([x0all[i * 30:(i + 1) * 30] for i in range(B)])
        ref_h, ref_l = full_reference(temporal, xb, "auto")
        st = FusedMambaStack(temporal, B, torch.float32)
        hid, lg = step_stack(st, xb)
        r["batch8_T30_fp32"] = {"hidden_max_abs": (hid - ref_h).abs().max().item(), "logit_max_abs": (lg - ref_l).abs().max().item()}
        res[name] = r
        print(name, "B batch8", r["batch8_T30_fp32"], flush=True)
    merge("kernel_b", res)


@torch.no_grad()
def test_e2e():
    res = {}
    for name in NAMES:
        features, temporal, resize = load_parts(name)
        x, y = sample_images(resize, 200, seed=2)
        orig = make_original(features, temporal, "fp32")
        origb = make_original(features, temporal, "bf16")
        ref = torch.cat([run_original(orig, x[i:i + 20, None]) for i in range(0, 200, 20)])
        refb = torch.cat([run_original(origb, x[i:i + 20, None], "bf16") for i in range(0, 200, 20)])
        r = {"orig_bf16_vs_orig_fp32": {"logit_max_abs": (refb - ref).abs().max().item(),
                                        "flips": int((refb.argmax(1) != ref.argmax(1)).sum())}}
        for dtype, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            for mode, fo in (("two", True), ("atomic", True), ("two", False)):
                p1 = StreamPipeline(features, temporal, 1, dtype, mode, fo)
                out = []
                for i in range(200):
                    p1.reset()
                    out.append(p1.step(x[i:i + 1]).clone())
                out = torch.cat(out)
                key = f"{dn}_tail_{mode}_fuseout{int(fo)}_B1"
                r[key] = {"logit_max_abs": (out - ref).abs().max().item(), "logit_ref_max_abs": ref.abs().max().item(),
                          "prob_fake_max_abs": (out.softmax(1)[:, 1] - ref.softmax(1)[:, 1]).abs().max().item(),
                          "flips": int((out.argmax(1) != ref.argmax(1)).sum()),
                          "vs_bf16_original_logit_max_abs": (out - refb).abs().max().item(),
                          "vs_bf16_original_flips": int((out.argmax(1) != refb.argmax(1)).sum())}
            p8 = StreamPipeline(features, temporal, 8, dtype)
            out = []
            for i in range(0, 200, 8):
                xb = x[i:i + 8]
                n = xb.shape[0]
                if n < 8:
                    xb = torch.cat([xb, xb[-1:].expand(8 - n, -1, -1, -1)])
                p8.reset()
                out.append(p8.step(xb)[:n].clone())
            out = torch.cat(out)
            r[f"{dn}_B8_batched"] = {"logit_max_abs": (out - ref).abs().max().item(), "flips": int((out.argmax(1) != ref.argmax(1)).sum())}
            g = StreamPipeline(features, temporal, 1, dtype)
            g.capture()
            out = []
            for i in range(200):
                g.reset()
                out.append(g.step_graph(x[i:i + 1]).clone())
            out = torch.cat(out)
            r[f"{dn}_cuda_graph_B1"] = {"logit_max_abs": (out - ref).abs().max().item(), "flips": int((out.argmax(1) != ref.argmax(1)).sum())}
        xs, _ = sample_images(resize, 30, seed=3)
        frames = xs[None]
        for dtype, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
            p = StreamPipeline(features, temporal, 1, dtype)
            p.reset()
            lg = torch.cat([p.step(xs[t:t + 1]).clone() for t in range(30)])
            ref30 = torch.cat([run_original(orig, frames[:, :t + 1]) for t in range(30)])
            r[f"seq30_{dn}"] = {"logit_max_abs": (lg - ref30).abs().max().item(), "logit_ref_max_abs": ref30.abs().max().item(),
                                "prob_fake_max_abs": (lg.softmax(1)[:, 1] - ref30.softmax(1)[:, 1]).abs().max().item(),
                                "flips": int((lg.argmax(1) != ref30.argmax(1)).sum()),
                                "per_step_logit_err": [round(v, 6) for v in (lg - ref30).abs().amax(1).tolist()]}
            if dn == "fp32":
                ref_full = run_original(orig, frames)
                r["seq30_fp32"]["final_vs_full_30frame_forward"] = (lg[-1:] - ref_full).abs().max().item()
        res[name] = r
        print(name, "E2E", json.dumps({k: v for k, v in r.items() if k != "seq30_fp32" and k != "seq30_bf16"}), flush=True)
        print(name, "seq30", {k: {a: b for a, b in v.items() if a != "per_step_logit_err"} for k, v in r.items() if k.startswith("seq30")}, flush=True)
    merge("e2e", res)


@torch.no_grad()
def test_full():
    res = {}
    for name in NAMES:
        features, temporal, resize = load_parts(name)
        import os
        items = test_items()
        if len(sys.argv) > 2:
            items = items[::len(items) // int(sys.argv[2])]
        ds = TestImages(items, resize)
        loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=10)
        orig = make_original(features, temporal, "fp32")
        origb = make_original(features, temporal, "bf16")
        orig64 = make_original(features, temporal, "fp64")
        pf = StreamPipeline(features, temporal, 64, torch.float32)
        pb = StreamPipeline(features, temporal, 64, torch.bfloat16)
        keys = ("orig_fp64", "orig_fp32", "orig_bf16_autocast", "orig_bf16_pure", "fused_fp32", "fused_bf16")
        logits = {k: [] for k in keys}
        labels = []
        t0 = time.time()
        for u8, y, _ in loader:
            x = normalize(u8)
            n = x.shape[0]
            labels.append(y)
            xp = x if n == 64 else torch.cat([x, x[-1:].expand(64 - n, -1, -1, -1)])
            logits["orig_fp64"].append(run_original(orig64, x[:, None]))
            logits["orig_fp32"].append(run_original(orig, x[:, None]))
            logits["orig_bf16_autocast"].append(run_original(orig, x[:, None], "bf16_autocast"))
            logits["orig_bf16_pure"].append(run_original(origb, x[:, None], "bf16"))
            pf.reset()
            logits["fused_fp32"].append(pf.step(xp)[:n].clone())
            pb.reset()
            logits["fused_bf16"].append(pb.step(xp)[:n].clone())
        labels = torch.cat(labels).cuda()
        L = {k: torch.cat(v) for k, v in logits.items()}
        r = {"n": int(labels.numel()), "seconds": time.time() - t0}
        for k, v in L.items():
            pred = v.argmax(1)
            r[k] = {"accuracy": float((pred == labels).float().mean()), "correct": int((pred == labels).sum())}
        for a, b in (("fused_fp32", "orig_fp32"), ("fused_bf16", "orig_bf16_pure"), ("fused_bf16", "orig_bf16_autocast"),
                     ("fused_bf16", "orig_fp32"), ("orig_bf16_pure", "orig_fp32"), ("orig_bf16_autocast", "orig_fp32"),
                     ("orig_fp32", "orig_fp64"), ("fused_fp32", "orig_fp64"), ("orig_bf16_pure", "orig_fp64"), ("fused_bf16", "orig_fp64")):
            d = (L[a] - L[b]).abs().amax(1)
            flips = (L[a].argmax(1) != L[b].argmax(1))
            r[f"{a}_vs_{b}"] = {"flips": int(flips.sum()), "logit_max_abs": d.max().item(), "logit_mean_abs": d.mean().item(),
                                "prob_fake_max_abs": (L[a].softmax(1)[:, 1] - L[b].softmax(1)[:, 1]).abs().max().item(),
                                "flip_margin_max": ((L[b][:, 1] - L[b][:, 0]).abs()[flips].max().item() if flips.any() else None)}
        res[name] = r
        print(name, "FULL", json.dumps(r), flush=True)
    merge("full_test", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["a", "b", "e2e", "full", "all"])
    a, _ = ap.parse_known_args()
    if a.part in ("a", "all"):
        test_kernel_a()
    if a.part in ("b", "all"):
        test_kernel_b()
    if a.part in ("e2e", "all"):
        test_e2e()
    if a.part in ("full", "all"):
        test_full()
