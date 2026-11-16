import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"

NAMES = {
    "fused": "Fused path (kernels A + B)",
    "torch_stream": "Baseline 1: unfused PyTorch recurrent step",
    "window_fused_scan": "Baseline 2a: window head recomputed with fused scan, trunk once per frame",
    "window_full_recompute": "Baseline 2b: trunk recomputed on all 30 frames, then fused-scan head",
    "window_attention": "Baseline 3: 1-layer attention over the 30-frame window, trunk once per frame",
}


def mib(x):
    return f"{x / 2**20:.2f}"


def sci(x):
    return f"{x:.2e}"


def render():
    d = json.loads((RES / "stream_fused.json").read_text())
    cj, tj = RES / "stream_fused_correctness.json", RES / "stream_tuning.json"
    if cj.exists():
        d["correctness"] = json.loads(cj.read_text())
    if tj.exists():
        d["tuning"] = json.loads(tj.read_text())
    (RES / "stream_fused.json").write_text(json.dumps(d, indent=1))
    L = []
    e0, e1 = d["env_start"], d["env_end"]
    L.append("# Fused CNN-to-SSM streaming path: measurements")
    L.append("")
    L.append(f"GPU {e0['gpu']}, torch {d['torch']}. Idle check before timing (utilisation percent, 3 samples 5 s apart): {d['idle_samples_pct']}. "
             f"Start: SM clock {e0['sm_clock_MHz']} MHz, memory clock {e0['mem_clock_MHz']} MHz, {e0['temp_C']} C, {e0['power_W']} W. "
             f"End: SM clock {e1['sm_clock_MHz']} MHz, {e1['temp_C']} C, {e1['power_W']} W. "
             "Latency is wall clock per new frame including a device sync; the input image is already preprocessed and on the GPU.")
    L.append("")
    for key, r in d["runs"].items():
        L.append(f"## {key}")
        L.append("")
        L.append("| Method | Eager ms (p10-p90) | CUDA graph ms (p10-p90) | GPU kernels per frame | CPU kernel launches (eager) | Peak MB eager | Peak MB graph | State bytes per stream |")
        L.append("|---|---|---|---|---|---|---|---|")
        for m, v in r["methods"].items():
            L.append(f"| {NAMES[m]} | {v['eager']['median_ms']:.3f} ({v['eager']['p10_ms']:.3f}-{v['eager']['p90_ms']:.3f}) | "
                     f"{v['graph']['median_ms']:.3f} ({v['graph']['p10_ms']:.3f}-{v['graph']['p90_ms']:.3f}) | {v['launches_eager']['gpu_kernels']} | "
                     f"{v['launches_eager']['cpu_kernel_launch_calls']} | {v['memory_eager']['peak_allocated_MB']:.0f} | "
                     f"{v['memory_graph']['peak_allocated_MB']:.0f} | {v['state_bytes_per_stream']:.0f} |")
        st = r["stages_graph_ms"]
        L.append("")
        L.append("Stage times inside a CUDA graph (ms): " + ", ".join(f"{k} {v:.4f}" for k, v in st.items() if k.endswith("_ms")) + ".")
        L.append("")
        la = st["launches_after_trunk"]
        L.append("GPU kernels after the trunk: " + ", ".join(f"{k} {v['gpu_kernels']}" for k, v in la.items()) + ".")
        L.append("")
        b = r["bytes"]
        ka, kb = b["fused_kernel_A"], b["fused_kernel_B"]
        b["must_move_after_trunk_weights_plus_state"] = ka["weights"] + ka["state_rw"] + kb["weights"] + kb["state_rw"]
        b["fused_after_trunk_activation_traffic"] = ka["activations"] + kb["activations"]
        b["unfused_after_trunk_activation_traffic"] = b["unfused_after_trunk"] - b["must_move_after_trunk_weights_plus_state"]
        L.append(f"Analytic DRAM MiB per frame (element size {b['elem_bytes']} B). Trunk: unfused {mib(b['unfused_trunk'])}, with fused depthwise kernel {mib(b['fused_trunk'])}. "
                 f"After the trunk: unfused {mib(b['unfused_after_trunk'])}, fused {mib(b['fused_after_trunk'])} "
                 f"(weights plus state that must move anyway: {mib(b['must_move_after_trunk_weights_plus_state'])}; activation traffic: unfused {mib(b['unfused_after_trunk_activation_traffic'])}, fused {mib(b['fused_after_trunk_activation_traffic'])}). "
                 f"Whole frame: unfused {mib(b['unfused_total'])}, fused {mib(b['fused_total'])}; window baselines 2a {mib(b['window_fused_scan_total'])}, 2b {mib(b['window_full_recompute_total'])}, 3 {mib(b['window_attention_total'])}.")
        L.append("")
    c = d.get("correctness")
    if c:
        L.append("## Correctness")
        L.append("")
        if "kernel_a" in c:
            L.append("### Kernel A (tail + pool + proj) against PyTorch fp32, real trunk maps, max abs error of the d_model output")
            L.append("")
            L.append("| Checkpoint | fp32 two-kernel B=1 / 8 / 64 | bf16 two-kernel B=1 / 8 / 64 | PyTorch bf16 tail | output max abs |")
            L.append("|---|---|---|---|---|")
            for n, r in c["kernel_a"].items():
                f = " / ".join(sci(r[f"fp32_two_B{B}"]["max_abs"]) for B in (1, 8, 64))
                g = " / ".join(sci(r[f"bf16_two_B{B}"]["max_abs"]) for B in (1, 8, 64))
                L.append(f"| {n} | {f} | {g} | {sci(r['torch_bf16_vs_fp32']['max_abs'])} | {r['ref_max_abs']:.2f} |")
            L.append("")
        if "kernel_b" in c:
            L.append("### Kernel B stepped over T frames against the full-sequence parallel scan, worst error over all steps")
            L.append("")
            L.append("| Checkpoint | T | variant | hidden max abs | hidden max abs / max hidden | logit max abs |")
            L.append("|---|---|---|---|---|---|")
            for n, r in c["kernel_b"].items():
                for T in (30, 300, 3000):
                    for v in ("fp32_fuseout1", "fp32_fuseout0", "bf16_fuseout1"):
                        x = r[f"T{T}_{v}"]
                        L.append(f"| {n} | {T} | {v} | {sci(x['hidden_max_abs'])} | {sci(x['hidden_max_rel'])} | {sci(x['logit_max_abs'])} |")
                    x = r[f"T{T}_torch_bf16_fullseq_vs_fp32"]
                    L.append(f"| {n} | {T} | PyTorch bf16 full sequence (for scale) | {sci(x['hidden_max_abs'])} | | {sci(x['logit_max_abs'])} |")
            L.append("")
        if "e2e" in c:
            L.append("### End to end, 200 real images as 1-frame sequences, and a 30-image sequence")
            L.append("")
            L.append("| Checkpoint | Variant | logit max abs vs fp32 original | flips | note |")
            L.append("|---|---|---|---|---|")
            for n, r in c["e2e"].items():
                for k, v in r.items():
                    if isinstance(v, dict) and "flips" in v and "logit_max_abs" in v:
                        L.append(f"| {n} | {k} | {sci(v['logit_max_abs'])} | {v['flips']} | |")
                L.append(f"| {n} | bf16 original vs fp32 original | {sci(r['orig_bf16_vs_orig_fp32']['logit_max_abs'])} | {r['orig_bf16_vs_orig_fp32']['flips']} | the original's own bf16 drift |")
            L.append("")
        if "full_test" in c:
            L.append("### Full test set (10905 images, each a 1-frame sequence)")
            L.append("")
            L.append("| Checkpoint | Pipeline | accuracy |")
            L.append("|---|---|---|")
            for n, r in c["full_test"].items():
                for k in ("orig_fp64", "orig_fp32", "orig_bf16_autocast", "orig_bf16_pure", "fused_fp32", "fused_bf16"):
                    L.append(f"| {n} | {k} | {r[k]['accuracy']:.5f} ({r[k]['correct']}/{r['n']}) |")
            L.append("")
            L.append("| Checkpoint | Comparison | prediction flips | logit max abs | logit mean abs |")
            L.append("|---|---|---|---|---|")
            for n, r in c["full_test"].items():
                for k, v in r.items():
                    if "_vs_" in k:
                        L.append(f"| {n} | {k} | {v['flips']} | {v['logit_max_abs']:.4f} | {v['logit_mean_abs']:.5f} |")
            L.append("")
    (RES / "stream_fused.md").write_text("\n".join(L))


if __name__ == "__main__":
    render()
