import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT / "cuda"))
from common import DETECTOR_REPO

sys.path.insert(0, str(DETECTOR_REPO))
from build import load, scan_cuda
from dfmamba.ssm import SelectiveSSM, scan_reference
from selective_scan import selective_scan_fused

OUT = ROOT / "results" / "ncu"
E, N = 1024, 16
SHAPES = [(1, 30), (4, 30), (1, 300), (4, 300), (1, 3000), (4, 3000)]
CPBS = (16, 32, 64, 128, 256, 512)
SS = (1, 2, 4, 8, 16)


def smi():
    q = "utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,clocks.max.sm,power.draw"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(["util_pct", "temp_c", "sm_mhz", "mem_mhz", "sm_max_mhz", "power_w"], [float(v) for v in out]))


def wait_idle(max_wait_s=900):
    t0 = time.time()
    ok = 0
    while time.time() - t0 < max_wait_s:
        ok = ok + 1 if smi()["util_pct"] <= 5 else 0
        if ok >= 3:
            return True
        time.sleep(5)
    return False


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def make_inputs(B, L):
    torch.manual_seed(0)
    blk = SelectiveSSM(E // 2, d_state=N).cuda().eval()
    x = torch.randn(B, L, E, device="cuda")
    with torch.no_grad():
        return blk.ssm_inputs(x)


def occupancy_table(ext):
    rows = []
    for cpb in CPBS:
        for s in SS:
            if not 32 <= cpb * s <= 1024:
                continue
            regs, smem, local, blocks, thr, maxthr, sms = ext.kernel_info(False, cpb, s)
            rows.append({"cpb": cpb, "s": s, "threads": cpb * s, "regs": regs, "static_smem_B": smem,
                         "local_B": local, "blocks_per_sm": blocks, "threads_per_sm": thr,
                         "occupancy_pct": 100 * thr / maxthr})
    return rows


def main():
    ext = load()
    gate = wait_idle()
    start = smi()
    occ = occupancy_table(ext)
    sweep, bench = [], []
    for (B, L) in SHAPES:
        u, d, A, Bm, Cm, D = make_inputs(B, L)
        ub, db, Bb, Cb = (t.to(torch.bfloat16) for t in (u, d, Bm, Cm))
        cfgs = []
        with torch.no_grad():
            for o in occ:
                t = timeit(lambda: scan_cuda(u, d, A, Bm, Cm, D, o["cpb"], o["s"]), 30)
                cfgs.append((t, o))
                sweep.append({"batch": B, "frames": L, "cpb": o["cpb"], "s": o["s"], "threads": o["threads"],
                              "grid_blocks": ((E + o["cpb"] - 1) // o["cpb"]) * B, "ms": t, "regs": o["regs"],
                              "occupancy_pct": o["occupancy_pct"]})
            cfgs.sort(key=lambda p: p[0])
            best = cfgs[0][1]
            cpb, s = best["cpb"], best["s"]
            t_cuda = timeit(lambda: scan_cuda(u, d, A, Bm, Cm, D, cpb, s), 200)
            t_cuda_bf = timeit(lambda: scan_cuda(ub, db, A, Bb, Cb, D, cpb, s), 200)
            t_default = timeit(lambda: scan_cuda(u, d, A, Bm, Cm, D, 32, 2), 200)
            t_tri = timeit(lambda: selective_scan_fused(u, d, A, Bm, Cm, D), 200)
            it = 200 if L <= 30 else 50 if L <= 300 else 10
            t_loop = timeit(lambda: scan_reference(u, d, A, Bm, Cm, D), it, warmup=3)
        fb = 4 * B * L * (3 * E + 2 * N)
        bench.append({"batch": B, "frames": L, "best_cpb": cpb, "best_s": s, "loop_iters": it,
                      "pytorch_loop_ms": t_loop, "triton_fp32_ms": t_tri, "cuda_fp32_ms": t_cuda,
                      "cuda_bf16_ms": t_cuda_bf, "cuda_fp32_default_cfg_ms": t_default,
                      "fp32_bytes_MB": fb / 1e6, "bf16_bytes_MB": fb / 2e6,
                      "cuda_fp32_GBps": fb / t_cuda / 1e6, "triton_fp32_GBps": fb / t_tri / 1e6,
                      "cuda_bf16_GBps": fb / 2 / t_cuda_bf / 1e6,
                      "cuda_vs_triton": t_tri / t_cuda, "cuda_vs_loop": t_loop / t_cuda,
                      "top3_configs": [{"cpb": o["cpb"], "s": o["s"], "ms": t} for t, o in cfgs[:3]],
                      "worst_config": {"cpb": cfgs[-1][1]["cpb"], "s": cfgs[-1][1]["s"], "ms": cfgs[-1][0]}})
        print(bench[-1], flush=True)
    end = smi()
    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "idle_gate_met": gate,
           "smi_start": start, "smi_end": end, "timing": "CUDA events, median; 200 iters except PyTorch loop (see loop_iters)",
           "bench": bench, "sweep": sweep, "occupancy": occ}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cuda_scan_bench.json").write_text(json.dumps(res, indent=1))
    L = ["| Batch | Frames | PyTorch loop ms | Triton fp32 ms | CUDA fp32 ms | CUDA bf16 ms | best (cpb, S) | CUDA vs Triton | CUDA vs loop | CUDA fp32 GB/s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in bench:
        L.append(f"| {r['batch']} | {r['frames']} | {r['pytorch_loop_ms']:.3f} | {r['triton_fp32_ms']:.4f} | {r['cuda_fp32_ms']:.4f} | "
                 f"{r['cuda_bf16_ms']:.4f} | ({r['best_cpb']}, {r['best_s']}) | {r['cuda_vs_triton']:.2f}x | {r['cuda_vs_loop']:.0f}x | {r['cuda_fp32_GBps']:.1f} |")
    print("\n".join(L))
    print("idle gate met:", gate, "start", start, "end", end)


if __name__ == "__main__":
    main()
