import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SPEC_BW = 128 / 8 * 18e9


def main():
    prof = json.loads((ROOT / "profiling" / "results" / "per_layer_bfloat16_c4_t30.json").read_text())
    bench = json.loads((ROOT / "kernels" / "results" / "bench.json").read_text())
    ceil = json.loads((ROOT / "env" / "ceilings.json").read_text())
    peak = ceil["bf16_dense_TFLOPs"] * 1e12
    groups = defaultdict(lambda: [0.0, 0, 0])
    for r in prof["rows"]:
        g = groups[(r["section"], r["kind"])]
        g[0] += r["ms"]; g[1] += r["flops"]; g[2] += r["bytes"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=150)
    xs = [10 ** (i / 20) for i in range(-30, 81)]
    ax.plot(xs, [min(peak, x * SPEC_BW) / 1e12 for x in xs], color="black", lw=1.6,
            label=f"Roofline: 288 GB/s, {peak/1e12:.0f} TFLOP/s BF16 (measured)")
    ax.plot(xs, [min(ceil['fp8_dense_TFLOPs'], x * SPEC_BW / 1e12) for x in xs], color="gray", lw=0.8, ls=":",
            label=f"FP8 tensor-core ceiling ({ceil['fp8_dense_TFLOPs']:.0f} TFLOP/s measured)")
    colors = {"EfficientNet-B0": "#1f6feb", "Mamba": "#b91c1c", "Head": "#6b7280", "Projection": "#6b7280"}
    for (sec, kind), (ms, f, b) in groups.items():
        if ms < 0.05 or "selective scan" in kind:
            continue
        x, y = f / b, f / ms / 1e9
        ax.scatter(x, y, s=20 + 160 * ms / 10, color=colors[sec], alpha=0.75, zorder=3)
        ax.annotate(kind.replace("selective scan (python loop)", "scan (loop)"), (x, y), fontsize=6,
                    xytext=(4, 3), textcoords="offset points", color=colors[sec])
    dw = bench["dwconv"]
    f = sum(2 * r["in"][0] * r["in"][1] * (r["in"][2] // r["s"]) * (r["in"][3] // r["s"]) * r["k"] ** 2 for r in dw)
    b = sum(r["useful_MB"] for r in dw) * 1e6
    for label, ms, mk in (("dwconv+BN+SiLU eager", sum(r["eager_ms"] for r in dw), "o"),
                          ("dwconv+BN+SiLU fused", sum(r["fused_ms"] for r in dw), "*")):
        ax.scatter(f / b, f / ms / 1e9, marker=mk, s=110, color="#16a34a", zorder=4,
                   edgecolor="black" if mk == "*" else None, label=label)
    sc = next(r for r in bench["scan"] if r["batch"] == 4 and r["frames"] == 30)
    E, N, B, L = 1024, 16, 4, 30
    sf = B * L * E * N * 7 + 2 * B * L * E * E + 2 * B * L * E * 2 * N
    wb = (E * E + E + E * 2 * N) * 2 + E * N * 4
    for label, ms, by, mk in (("Mamba scan, PyTorch loop (1 layer)", sc["loop_ms"], sc["loop_min_MB"] * 1e6 + wb, "o"),
                              ("Mamba scan, fused Triton (1 layer)", sc["fused_ms"], sc["fused_MB"] * 1e6 + wb, "*")):
        ax.scatter(sf / by, sf / ms / 1e9, marker=mk, s=110, color="#f59e0b", zorder=4,
                   edgecolor="black" if mk == "*" else None, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4); ax.set_ylim(1e-4, 200)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title("Deepfake detector on RTX 4060 Ti, BF16, 4 clips x 30 frames\n"
                 "dot size = time share; every op sits far left of the ridge", fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "roofline" / "roofline_4060ti.png")


if __name__ == "__main__":
    main()
