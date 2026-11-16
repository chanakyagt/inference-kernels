import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DETECTOR_REPO, bench, load_detector, peak

sys.path.insert(0, str(DETECTOR_REPO))
from dfmamba.ssm import SelectiveSSM


def numel(t):
    return t.numel() if torch.is_tensor(t) else 0


def flops_of(mod, inp, out):
    x = inp[0]
    if isinstance(mod, nn.Conv2d):
        kh, kw = mod.kernel_size
        return 2 * out.numel() * (mod.in_channels // mod.groups) * kh * kw
    if isinstance(mod, nn.Conv1d):
        return 2 * out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0]
    if isinstance(mod, nn.Linear):
        return 2 * (x.numel() // x.shape[-1]) * mod.in_features * mod.out_features
    if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
        return 4 * out.numel()
    if isinstance(mod, (nn.SiLU, nn.Sigmoid, nn.ReLU)):
        return 4 * out.numel()
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return x.numel()
    return out.numel()


def kind_of(name, mod):
    if isinstance(mod, nn.Conv2d):
        if mod.groups == mod.in_channels and mod.groups > 1:
            return f"conv dw {mod.kernel_size[0]}x{mod.kernel_size[0]}"
        if "fc" in name or ".1.fc" in name:
            return "conv 1x1 (SE)"
        return "conv 1x1" if mod.kernel_size == (1, 1) else f"conv {mod.kernel_size[0]}x{mod.kernel_size[0]}"
    return type(mod).__name__


def section(name):
    if name.startswith("encoder"):
        return "EfficientNet-B0"
    if name.startswith("temporal.layers"):
        return "Mamba"
    if name.startswith("temporal.proj"):
        return "Projection"
    return "Head"


def capture(model, x):
    rec = {}
    hooks = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and not isinstance(mod, (nn.Dropout, nn.Identity)) \
                and type(mod).__name__ != "StochasticDepth":
            def hook(m, i, o, name=name):
                rec.setdefault(name, (m, tuple(t.detach() for t in i), o.detach()))
            hooks.append(mod.register_forward_hook(hook))
    scans = {}
    originals = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SelectiveSSM):
            originals[name] = mod.scan
            def wrapped(x, name=name, mod=mod):
                y = originals[name](x)
                scans.setdefault(name + ".scan", (mod, x.detach(), y.detach()))
                return y
            mod.scan = wrapped
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    for name, mod in model.named_modules():
        if isinstance(mod, SelectiveSSM):
            mod.scan = originals[name]
    return rec, scans


def scan_cost(mod, x):
    B, L, E = x.shape
    N = mod.d_state
    flops = B * L * E * N * 7 + 2 * B * L * E * mod.dt_proj.in_features + 2 * B * L * E * 2 * N
    elt = x.element_size()
    bytes_ = elt * B * L * E * 2 + 4 * (B * L * E * N * 2) * 2 + 4 * E * N
    return flops, bytes_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--clips", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    torch.backends.cudnn.benchmark = True
    model = load_detector(dt)
    x = torch.randn(a.clips, a.frames, 3, 224, 224, device="cuda", dtype=dt)
    total_ms = bench(lambda: model(x), iters=30)
    feat_ms = bench(lambda: model.encoder(x), iters=30)
    rec, scans = capture(model, x)
    pk = peak()
    rows = []
    for name, (mod, inp, out) in rec.items():
        ms = bench(lambda: mod(*inp), iters=50)
        f = flops_of(mod, inp, out)
        wbytes = sum(p.numel() * p.element_size() for p in mod.parameters(recurse=False)) + \
            sum(b.numel() * b.element_size() for b in mod.buffers(recurse=False))
        by = sum(numel(t) * t.element_size() for t in inp) + out.numel() * out.element_size() + wbytes
        rows.append({"name": name, "section": section(name), "kind": kind_of(name, mod),
                     "out_shape": list(out.shape), "ms": ms, "flops": f, "bytes": by})
    for name, (mod, xin, y) in scans.items():
        ms = bench(lambda: mod.scan(xin), iters=20)
        f, by = scan_cost(mod, xin)
        rows.append({"name": name, "section": "Mamba", "kind": "selective scan (python loop)",
                     "out_shape": list(y.shape), "ms": ms, "flops": f, "bytes": by})
    for r in rows:
        r["intensity"] = r["flops"] / r["bytes"]
        r["gbps"] = r["bytes"] / r["ms"] / 1e6
        r["tflops"] = r["flops"] / r["ms"] / 1e9
        r["pct_bw"] = 100 * r["gbps"] * 1e9 / pk["bw"]
        r["bound"] = "memory" if r["intensity"] < pk["ridge"] else "compute"

    by_kind = defaultdict(lambda: {"ms": 0.0, "flops": 0, "bytes": 0, "count": 0})
    for r in rows:
        k = by_kind[(r["section"], r["kind"])]
        k["ms"] += r["ms"]; k["flops"] += r["flops"]; k["bytes"] += r["bytes"]; k["count"] += 1
    leaf_ms = sum(r["ms"] for r in rows if "selective scan" not in r["kind"])
    summary = {"dtype": a.dtype, "clips": a.clips, "frames": a.frames, "end_to_end_ms": total_ms,
               "backbone_ms": feat_ms, "sum_of_isolated_ops_ms": sum(r["ms"] for r in rows),
               "sum_leaf_ms_excl_scan": leaf_ms, "peak": pk, "gpu": torch.cuda.get_device_name(),
               "torch": torch.__version__}
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dtype}_c{a.clips}_t{a.frames}"
    Path(a.out, f"per_layer_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    lines = [f"# Per-layer profile: {a.clips} clip(s) x {a.frames} frames, {a.dtype}, {summary['gpu']}", "",
             f"- End-to-end forward: **{total_ms:.2f} ms**; EfficientNet-B0 part alone: {feat_ms:.2f} ms",
             f"- Sum of every op timed in isolation: {summary['sum_of_isolated_ops_ms']:.2f} ms "
             "(differs from end-to-end because isolated timing has no launch overlap and skips "
             "un-hooked ops such as residual adds and the squeeze-excitation multiply)",
             f"- Ceilings used: {pk['bw']/1e9:.0f} GB/s, {pk['flops']/1e12:.1f} TFLOP/s "
             f"({pk['dtype_note']}), ridge {pk['ridge']:.0f} FLOP/byte", "",
             "## Grouped by kind", "",
             "| Section | Kind | Ops | Time (ms) | Share | GFLOP | MB moved | FLOP/byte | Achieved GB/s | Bound |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    tot = summary["sum_of_isolated_ops_ms"]
    for (sec, kind), k in sorted(by_kind.items(), key=lambda kv: -kv[1]["ms"]):
        ai = k["flops"] / k["bytes"]
        lines.append(f"| {sec} | {kind} | {k['count']} | {k['ms']:.3f} | {100*k['ms']/tot:.1f}% | "
                     f"{k['flops']/1e9:.2f} | {k['bytes']/1e6:.1f} | {ai:.1f} | {k['bytes']/k['ms']/1e6:.0f} | "
                     f"{'memory' if ai < pk['ridge'] else 'compute'} |")
    lines += ["", "## Every op", "",
              "| Op | Kind | Output shape | Time (ms) | GFLOP | MB moved | FLOP/byte | GB/s | % of peak BW |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['kind']} | {'x'.join(map(str, r['out_shape']))} | {r['ms']:.4f} | "
                     f"{r['flops']/1e9:.3f} | {r['bytes']/1e6:.2f} | {r['intensity']:.1f} | {r['gbps']:.0f} | {r['pct_bw']:.0f}% |")
    Path(a.out, f"per_layer_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12 + len(by_kind) + 2]))


if __name__ == "__main__":
    main()
