import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
import inventory
import qlib
import qops
from fused_dwconv_bn_silu import dwconv_bn_silu
from int8_dwconv import bf16_dwconv_nhwc, int8_dwconv_nhwc, quantize_dw_weight

BATCHES = (1, 8, 64)
BF16 = torch.bfloat16


def both(fn):
    return {"graph_ms": qops.time_graph(fn), "eager_ms": qops.time_eager(fn)}


def try_both(out, key, fn):
    try:
        out[key] = both(fn)
    except RuntimeError as e:
        torch.cuda.synchronize()
        out[key] = {"error": str(e)[:120]}


def gemm_variants(M, K, N, act):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(M, K, device=dev, generator=g).to(BF16)
    w = (torch.randn(N, K, device=dev, generator=g) * 0.1)
    bias = torch.randn(N, device=dev, generator=g)
    wb, bb = w.to(BF16).contiguous(), bias.to(BF16)
    wq, ws = qops.quant_weight_int8(w)
    wq_t = wq.t().contiguous()
    sx = 0.05
    xq = qops.quant_act(x, sx)
    out = {}
    out["bf16_cublas"] = both(lambda: (F.silu(torch.addmm(bb, x, wb.t())) if act else torch.addmm(bb, x, wb.t())))
    out["bf16_triton"] = both(lambda: qops.gemm_bf16(x, wb, bias, act))
    out["int8_triton_bf16in"] = both(lambda: qops.gemm_int8(x, wq, ws, bias, sx, act))
    out["int8_triton_int8in"] = both(lambda: qops.gemm_int8(xq, wq, ws, bias, sx, act, a_is_q=True))
    out["int8_triton_int8in_int8out"] = both(lambda: qops.gemm_int8(xq, wq, ws, bias, sx, act, a_is_q=True, out_scale=torch.full((N,), 0.05, device=dev)))
    out["quant_pass_only"] = both(lambda: qops.quant_act(x, sx))
    if K % 8 == 0 and N % 8 == 0:
        xqp = qops.pad_rows(xq)
        try_both(out, "int_mm_only", lambda: torch._int_mm(xqp, wq_t))
        try_both(out, "int_mm_pipeline", lambda: qops.int_mm_pipeline(x, wq_t, ws, bias, sx, act))
    if K % 16 == 0 and N % 16 == 0:
        w8, sw = qops.quant_weight_fp8(w)
        w8t = w8.t()
        sxt = torch.tensor(sx, device=dev)
        x8 = qops.quant_act(x, sx, fp8=True)
        pad = (-M) % 16
        x8p = F.pad(x8.view(torch.int8), (0, 0, 0, pad)).view(qops.FP8) if pad else x8
        try_both(out, "fp8_mm_only", lambda: torch._scaled_mm(x8p, w8t, scale_a=sxt, scale_b=sw.float(), bias=bb, out_dtype=BF16))
        try_both(out, "fp8_pipeline", lambda: qops.fp8_pipeline(x, w8t, sw.float(), bb, sxt, sx, act))
    return out


def dw_variants(B, C, H, W, k, s):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    p = k // 2
    x = torch.randn(B, H, W, C, device=dev, generator=g).to(BF16)
    w = torch.randn(C, 1, k, k, device=dev, generator=g) * 0.2
    bias = torch.randn(C, device=dev, generator=g)
    sx = 0.05
    xq = qops.quant_act(x, sx)
    wq, ws = quantize_dw_weight(w)
    wkc = w.reshape(C, -1).t().contiguous()
    w4, bb = w.to(BF16), bias.to(BF16)
    xn = x.permute(0, 3, 1, 2)
    xnc = xn.contiguous()
    out = {}

    def cudnn(act):
        y = F.conv2d(xn, w4, bb, s, p, groups=C)
        return F.silu(y) if act else y

    out["cudnn_bf16"] = both(lambda: cudnn(False))
    out["cudnn_bf16_then_silu"] = both(lambda: cudnn(True))
    out["triton_bf16_nhwc"] = both(lambda: bf16_dwconv_nhwc(x, wkc, bias, s, p, k, False))
    out["triton_bf16_nhwc_silu"] = both(lambda: bf16_dwconv_nhwc(x, wkc, bias, s, p, k, True))
    out["triton_bf16_nchw_existing_silu"] = both(lambda: dwconv_bn_silu(xnc, w[:, 0].contiguous(), bias, s, p, True))
    out["triton_int8_nhwc"] = both(lambda: int8_dwconv_nhwc(xq, sx, wq, ws, bias, s, p, k, False))
    out["triton_int8_nhwc_silu"] = both(lambda: int8_dwconv_nhwc(xq, sx, wq, ws, bias, s, p, k, True))
    inv = torch.full((C,), 20.0, device=dev)
    wsc = ws * sx
    out["quant_pass_plus_int8_silu"] = both(lambda: int8_dwconv_nhwc(qops.quant_act_pc(x, inv), 1.0, wq, wsc, bias, s, p, k, True))
    Ho, Wo = (H + 2 * p - k) // s + 1, (W + 2 * p - k) // s + 1
    wbytes = C * k * k
    b16 = B * C * (H * W + Ho * Wo) * 2 + wbytes * 2
    b8 = B * C * (H * W) + B * C * Ho * Wo * 2 + wbytes
    for kname, byt in (("triton_bf16_nhwc_silu", b16), ("triton_int8_nhwc_silu", b8), ("cudnn_bf16", b16)):
        out[kname]["bytes"] = byt
        out[kname]["GBps_graph"] = byt / (out[kname]["graph_ms"] * 1e-3) / 1e9
    return out


def best(d, names):
    c = [(d[n]["graph_ms"], n) for n in names if n in d and "graph_ms" in d[n]]
    return min(c) if c else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="diloco")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--dw-only", action="store_true")
    a = ap.parse_args()
    qlib.OUT.mkdir(parents=True, exist_ok=True)
    idle = None if a.no_wait else qops.wait_idle()
    before = qops.gpu_state()
    model = qlib.build_model(a.ckpt)
    inv = {r["name"]: r for r in inventory.inventory(model, qlib.default_size(a.ckpt))}
    mods = dict(model.named_modules())
    prev = qlib.OUT / f"layer_speed_{a.tag}.json"
    res = {"gemm": {}, "dw": {}}
    if a.dw_only:
        res["gemm"] = json.loads(prev.read_text())["gemm"]
    cache = {}
    for name, r in inv.items():
        kind = r["kind"]
        if kind in ("stem3x3", "mamba_conv1d") or (a.dw_only and not kind.startswith("dw")):
            continue
        act = False
        if kind == "pw1x1":
            act = isinstance(mods[name.rsplit(".", 1)[0]][-1], torch.nn.SiLU)
        if kind == "se_fc":
            act = name.endswith("fc1")
        for B in BATCHES:
            if kind.startswith("dw"):
                key = ("dw", B, r["cout"], r["h"], r["w"], r["k"], r["stride"])
                if key not in cache:
                    cache[key] = dw_variants(B, r["cout"], r["h"], r["w"], r["k"], r["stride"])
                res["dw"].setdefault(name, {})[str(B)] = {"shape": [B, r["h"], r["w"], r["cout"], r["k"], r["stride"]], "variants": cache[key]}
            else:
                M = B * r["ho"] * r["wo"]
                key = ("g", M, r["cin"], r["cout"], act)
                if key not in cache:
                    cache[key] = gemm_variants(M, r["cin"], r["cout"], act)
                res["gemm"].setdefault(name, {})[str(B)] = {"M": M, "K": r["cin"], "N": r["cout"], "act": act, "kind": kind, "variants": cache[key]}
        print(name, flush=True)
    res["provenance"] = {"idle_samples": idle, "gpu_before": before, "gpu_after": qops.gpu_state(), "torch": torch.__version__,
                         "timing": "median of 15 CUDA-graph replays of 20 back-to-back calls (graph_ms) and median of 60 eager calls with CUDA events (eager_ms)"}
    (qlib.OUT / f"layer_speed_{a.tag}.json").write_text(json.dumps(res, indent=1))
    write_md(res, inv, a.tag)


BF16_NAMES = ("bf16_cublas", "bf16_triton")
Q_NAMES = ("int8_triton_bf16in", "int_mm_pipeline", "fp8_pipeline")


def write_md(res, inv, tag):
    L = [f"# Per-layer real speed ({tag})", "",
         "All times are GPU time per call in milliseconds, measured by replaying a CUDA graph (no launch overhead). 'bf16 best' is the faster of cuBLAS (+ separate SiLU) and a Triton bf16 GEMM with the same fused epilogue as the int8 kernel, so the speedup is not inflated by fusion alone.",
         "'Quant best' is the fastest complete quantized path that takes bf16 in and gives bf16 out, quantization of the input and dequantization included: Triton int8 GEMM with fused quantize/dequantize, torch._int_mm + quantize + dequantize kernels, or torch._scaled_mm FP8 + quantize kernel.", "",
         f"Provenance: {json.dumps(res['provenance'])}", ""]
    for B in BATCHES:
        L += [f"## Batch {B}: GEMM-type layers (1x1 convolutions, SE, Linear)", "",
              "| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in res["gemm"].items():
            e = d[str(B)]
            v = e["variants"]
            tb, nb = best(v, BF16_NAMES)
            tq, nq = best(v, Q_NAMES)
            def g(k):
                return f"{v[k]['graph_ms']:.4f}" if k in v and "graph_ms" in v[k] else "n/a"
            L.append(f"| {name} | {e['kind']} | {e['M']}x{e['K']}x{e['N']} | {tb:.4f} | {nb} | {tq:.4f} | {nq} | {tb / tq:.2f}x | "
                     f"{g('int8_triton_int8in_int8out')} | {g('int_mm_only')} | {g('fp8_mm_only')} |")
        L.append("")
        L += [f"## Batch {B}: depthwise convolutions (bias + SiLU fused where marked)", "",
              "| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, d in res["dw"].items():
            e = d[str(B)]
            v = e["variants"]
            bb = min(v["cudnn_bf16_then_silu"]["graph_ms"], v["triton_bf16_nhwc_silu"]["graph_ms"], v["triton_bf16_nchw_existing_silu"]["graph_ms"])
            i8 = v["triton_int8_nhwc_silu"]["graph_ms"]
            iq = v["quant_pass_plus_int8_silu"]["graph_ms"]
            L.append(f"| {name} | {'x'.join(map(str, e['shape'][:4]))}, {e['shape'][4]}, s{e['shape'][5]} | {v['cudnn_bf16_then_silu']['graph_ms']:.4f} | {v['triton_bf16_nhwc_silu']['graph_ms']:.4f} | "
                     f"{v['triton_bf16_nchw_existing_silu']['graph_ms']:.4f} | {i8:.4f} | {iq:.4f} | {v['triton_bf16_nhwc_silu']['GBps_graph']:.0f} | {v['triton_int8_nhwc_silu']['GBps_graph']:.0f} | {bb / i8:.2f}x | {bb / iq:.2f}x |")
        L.append("")
    (qlib.OUT / f"layer_speed_{tag}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
