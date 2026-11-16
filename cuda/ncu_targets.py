import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT / "cuda"))
from common import DETECTOR_REPO

sys.path.insert(0, str(DETECTOR_REPO))
from build import scan_cuda
from dfmamba.ssm import SelectiveSSM
from fused_dwconv_bn_silu import FusedDWConvBNSiLU
from selective_scan import selective_scan_fused

NVTX = torch.cuda.nvtx


def main(target):
    torch.manual_seed(0)
    B, L, E = 4, 30, 1024
    u, d, A, Bm, Cm, D = SelectiveSSM(E // 2, d_state=16).cuda().eval().ssm_inputs(torch.randn(B, L, E, device="cuda"))
    conv = nn.Conv2d(240, 240, 5, 1, 2, groups=240, bias=False).cuda().to(torch.bfloat16)
    bn = nn.BatchNorm2d(240).cuda().eval().to(torch.bfloat16)
    fused = FusedDWConvBNSiLU(conv, bn)
    x = torch.randn(30, 240, 28, 28, device="cuda", dtype=torch.bfloat16)
    runs = {
        "triton_scan": lambda: selective_scan_fused(u, d, A, Bm, Cm, D),
        "cuda_scan": lambda: scan_cuda(u, d, A, Bm, Cm, D, 32, 2),
        "fused_dw": lambda: fused(x),
        "unfused_dw": lambda: F.silu(bn(conv(x))),
    }
    fn = runs[target]
    with torch.no_grad():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        NVTX.range_push(target)
        fn()
        torch.cuda.synchronize()
        NVTX.range_pop()


if __name__ == "__main__":
    main(sys.argv[1])
