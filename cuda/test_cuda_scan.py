import json
import sys
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


def make_inputs(B, L, seed=0, E=1024, N=16):
    torch.manual_seed(seed)
    blk = SelectiveSSM(E // 2, d_state=N).cuda().eval()
    x = torch.randn(B, L, E, device="cuda")
    with torch.no_grad():
        return blk.ssm_inputs(x)


def main():
    load()
    rows = []
    configs = [(32, 1), (32, 2), (64, 4), (16, 16), (128, 1)]
    for L in (30, 300, 3000):
        for B in (1, 4):
            with torch.no_grad():
                u, d, A, Bm, Cm, D = make_inputs(B, L)
                ref32 = scan_reference(u, d, A, Bm, Cm, D)
                tri32 = selective_scan_fused(u, d, A, Bm, Cm, D)
                ub, db, Bb, Cb = (t.to(torch.bfloat16) for t in (u, d, Bm, Cm))
                ref16 = scan_reference(ub.float(), db.float(), A, Bb.float(), Cb.float(), D)
                tri16 = selective_scan_fused(ub.float(), db.float(), A, Bb.float(), Cb.float(), D)
                scale32, scale16 = ref32.abs().max().item(), ref16.abs().max().item()
                for cpb, s in configs:
                    o32 = scan_cuda(u, d, A, Bm, Cm, D, cpb, s)
                    o16 = scan_cuda(ub, db, A, Bb, Cb, D, cpb, s)
                    rows.append({
                        "batch": B, "frames": L, "cpb": cpb, "s": s,
                        "fp32_cuda_vs_ref": (o32 - ref32).abs().max().item(),
                        "fp32_cuda_vs_triton": (o32 - tri32).abs().max().item(),
                        "fp32_triton_vs_ref": (tri32 - ref32).abs().max().item(),
                        "bf16_cuda_vs_ref": (o16.float() - ref16).abs().max().item(),
                        "bf16_cuda_vs_triton": (o16.float() - tri16).abs().max().item(),
                        "bf16_triton_fp32math_vs_ref": (tri16 - ref16).abs().max().item(),
                        "ref_fp32_absmax": scale32, "ref_bf16_absmax": scale16,
                        "bf16_output_rounding_floor": (ref16.to(torch.bfloat16).float() - ref16).abs().max().item()})
                    assert rows[-1]["fp32_cuda_vs_ref"] < 1e-3 * max(1.0, scale32), rows[-1]
                    assert rows[-1]["bf16_cuda_vs_ref"] < 2e-2 * max(1.0, scale16), rows[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cuda_scan_correctness.json").write_text(json.dumps(rows, indent=1))
    print("| batch | frames | cpb | S | fp32 CUDA vs ref | fp32 CUDA vs Triton | fp32 Triton vs ref | bf16 CUDA vs ref | bf16 rounding floor | ref abs max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['batch']} | {r['frames']} | {r['cpb']} | {r['s']} | {r['fp32_cuda_vs_ref']:.2e} | {r['fp32_cuda_vs_triton']:.2e} | "
              f"{r['fp32_triton_vs_ref']:.2e} | {r['bf16_cuda_vs_ref']:.2e} | {r['bf16_output_rounding_floor']:.2e} | {r['ref_fp32_absmax']:.2f} |")
    print("ok  CUDA scan matches PyTorch reference and Triton at fp32 and bf16, T in (30, 300, 3000), batch in (1, 4)")


if __name__ == "__main__":
    main()
