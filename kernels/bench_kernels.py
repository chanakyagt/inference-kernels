import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
from common import DETECTOR_REPO, SPEC_BW, bench, load_detector

sys.path.insert(0, str(DETECTOR_REPO))
from fused_dwconv_bn_silu import FusedDWConvBNSiLU, is_dw_block, patch_model as patch_dw
from selective_scan import fused_block_scan, patch_model as patch_scan
from dfmamba.ssm import SelectiveSSM

OUT = ROOT / "kernels" / "results"


def dw_shapes(frames):
    m = load_detector(torch.bfloat16)
    found = []
    hooks = []
    for mod in m.modules():
        if is_dw_block(mod):
            hooks.append(mod.register_forward_hook(lambda mo, i, o: found.append((mo, i[0].detach()))))
    with torch.no_grad():
        m.encoder(torch.randn(1, frames, 3, 224, 224, device="cuda", dtype=torch.bfloat16))
    for h in hooks:
        h.remove()
    return found


def bench_dw(frames):
    rows = []
    for blk, x in dw_shapes(frames):
        conv, bn = blk[0], blk[1]
        fused = FusedDWConvBNSiLU(conv, bn)
        eager = lambda: F.silu(bn(conv(x)))
        torch._dynamo.reset()
        comp = torch.compile(lambda t: F.silu(bn(conv(t))), dynamic=False)
        with torch.no_grad():
            y = eager()
            t_eager = bench(eager, iters=100)
            t_fused = bench(lambda: fused(x), iters=100)
            try:
                t_comp = bench(lambda: comp(x), iters=100)
            except Exception:
                t_comp = None
        useful = (x.numel() + y.numel()) * x.element_size() + conv.weight.numel() * 4
        rows.append({"in": list(x.shape), "k": conv.kernel_size[0], "s": conv.stride[0],
                     "eager_ms": t_eager, "compiled_ms": t_comp, "fused_ms": t_fused,
                     "useful_MB": useful / 1e6,
                     "eager_GBps": useful / t_eager / 1e6, "fused_GBps": useful / t_fused / 1e6,
                     "fused_pct_bw": 100 * useful / t_fused / 1e6 * 1e9 / SPEC_BW})
    return rows


def bench_scan():
    rows = []
    torch.manual_seed(0)
    blk = SelectiveSSM(512, d_state=16).cuda().eval().to(torch.bfloat16)
    blk.scan_impl = "ref"
    blk.A_log.data = blk.A_log.data.float(); blk.D.data = blk.D.data.float()
    for (B, L) in ((1, 30), (4, 30), (1, 300), (1, 3000)):
        x = torch.randn(B, L, 1024, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            it = 20 if L <= 300 else 3
            t_loop = bench(lambda: blk.scan(x), iters=it, warmup=2)
            t_fused = bench(lambda: fused_block_scan(blk, x), iters=50)
        E, N = 1024, 16
        loop_bytes = 4 * B * L * E * N * 2 * 2
        fused_bytes = 4 * B * L * (3 * E + 2 * N)
        rows.append({"batch": B, "frames": L, "loop_ms": t_loop, "fused_ms": t_fused, "speedup": t_loop / t_fused,
                     "loop_min_MB": loop_bytes / 1e6, "fused_MB": fused_bytes / 1e6})
    return rows


def bench_end_to_end():
    rows = []
    for clips in (1, 4):
        x = torch.randn(clips, 30, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
        for name, dw, sc in (("baseline", False, False), ("fused dwconv", True, False),
                             ("fused scan", False, True), ("both", True, True)):
            m = load_detector(torch.bfloat16)
            if dw:
                patch_dw(m)
            if sc:
                patch_scan(m)
            with torch.no_grad():
                t = bench(lambda: m(x), iters=30)
            rows.append({"clips": clips, "frames": 30, "variant": name, "ms": t, "clips_per_s": 1000 * clips / t})
            if name in ("baseline", "both"):
                with torch.no_grad():
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(3):
                            m(x)
                    torch.cuda.current_stream().wait_stream(s)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        m(x)
                    t = bench(g.replay, iters=30)
                rows.append({"clips": clips, "frames": 30, "variant": name + " + CUDA graph", "ms": t,
                             "clips_per_s": 1000 * clips / t})
                del g
    return rows


def main():
    torch.backends.cudnn.benchmark = True
    OUT.mkdir(exist_ok=True)
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "dwconv": bench_dw(30), "scan": bench_scan(), "end_to_end": bench_end_to_end()}
    (OUT / "bench.json").write_text(json.dumps(res, indent=1))
    L = [f"# Kernel benchmarks ({res['gpu']}, torch {res['torch']}, bfloat16, median of repeated runs)", "",
         "## Depthwise conv + BatchNorm + SiLU, the 16 EfficientNet-B0 depthwise blocks at 30 frames", "",
         "Useful bytes = input + output + weights, read or written once. Achieved bandwidth = useful bytes / time.", "",
         "| Input (N,C,H,W) | k | s | PyTorch eager ms | torch.compile ms | Fused Triton ms | Speedup vs eager | Eager GB/s | Fused GB/s | Fused % of 288 GB/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["dwconv"]:
        c = f"{r['compiled_ms']:.3f}" if r["compiled_ms"] else "n/a"
        L.append(f"| {'x'.join(map(str, r['in']))} | {r['k']} | {r['s']} | {r['eager_ms']:.3f} | {c} | {r['fused_ms']:.3f} | "
                 f"{r['eager_ms']/r['fused_ms']:.2f}x | {r['eager_GBps']:.0f} | {r['fused_GBps']:.0f} | {r['fused_pct_bw']:.0f}% |")
    te = sum(r["eager_ms"] for r in res["dwconv"]); tf = sum(r["fused_ms"] for r in res["dwconv"])
    tc = sum(r["compiled_ms"] for r in res["dwconv"] if r["compiled_ms"])
    L += ["", f"Total over 16 blocks: eager {te:.2f} ms, torch.compile {tc:.2f} ms, fused {tf:.2f} ms ({te/tf:.2f}x vs eager).", "",
          "## Selective scan, one Mamba layer (d_inner 1024, d_state 16)", "",
          "| Batch | Frames | PyTorch loop ms | Fused Triton ms | Speedup | Loop: (B,L,E,N) tensors MB | Fused: MB read+written |",
          "|---|---|---|---|---|---|---|"]
    for r in res["scan"]:
        L.append(f"| {r['batch']} | {r['frames']} | {r['loop_ms']:.3f} | {r['fused_ms']:.3f} | {r['speedup']:.1f}x | "
                 f"{r['loop_min_MB']:.1f} | {r['fused_MB']:.2f} |")
    L += ["", "## Whole detector forward (EfficientNet-B0 + 4 Mamba layers + head), 30 frames per clip", "",
          "| Clips | Variant | ms | Clips/s |", "|---|---|---|---|"]
    for r in res["end_to_end"]:
        L.append(f"| {r['clips']} | {r['variant']} | {r['ms']:.2f} | {r['clips_per_s']:.1f} |")
    (OUT / "bench.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
