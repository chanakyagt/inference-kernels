import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
from common import DETECTOR_REPO, load_detector

sys.path.insert(0, str(DETECTOR_REPO))
from fused_dwconv_bn_silu import FusedDWConvBNSiLU, patch_model as patch_dw
from selective_scan import fused_block_scan, patch_model as patch_scan
from int8_dwconv import int8_dwconv_nhwc, quantize_dw_weight
from dfmamba.ssm import SelectiveSSM


def ref_dw(x, conv, bn):
    return F.silu(bn(conv(x)))


def test_dwconv():
    torch.manual_seed(0)
    shapes = [(30, 96, 112, 112, 3, 2), (30, 144, 56, 56, 3, 1), (30, 144, 56, 56, 5, 2), (30, 240, 28, 28, 5, 1),
              (30, 480, 14, 14, 3, 1), (30, 672, 14, 14, 5, 2), (30, 1152, 7, 7, 5, 1), (3, 32, 33, 17, 3, 2),
              (2, 7, 9, 9, 5, 1)]
    for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 3e-2), (torch.float16, 5e-3)):
        for (N, C, H, W, K, S) in shapes:
            conv = nn.Conv2d(C, C, K, S, K // 2, groups=C, bias=False).cuda()
            bn = nn.BatchNorm2d(C).cuda().eval()
            bn.running_mean.uniform_(-0.5, 0.5); bn.running_var.uniform_(0.5, 2.0)
            bn.weight.data.uniform_(0.5, 1.5); bn.bias.data.uniform_(-0.2, 0.2)
            x = torch.randn(N, C, H, W, device="cuda")
            with torch.no_grad():
                ref = ref_dw(x, conv, bn)
                out = FusedDWConvBNSiLU(conv, bn)(x.to(dtype)).float()
            err = ((out - ref).abs() / (ref.abs() + 1)).max().item()
            assert out.shape == ref.shape and err < tol, (dtype, (N, C, H, W, K, S), err)
    print(f"ok  fused dwconv+BN+SiLU matches PyTorch: {len(shapes)} shapes x fp32/bf16/fp16")


def test_scan():
    torch.manual_seed(0)
    for (B, L, D, N) in ((1, 30, 512, 16), (2, 30, 512, 16), (3, 7, 40, 16), (1, 300, 512, 16), (4, 1, 64, 8)):
        blk = SelectiveSSM(D, d_state=N).cuda().eval()
        blk.scan_impl = "ref"
        for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 2e-2)):
            b = blk.to(dtype)
            b.A_log.data = b.A_log.data.float(); b.D.data = b.D.data.float()
            x = torch.randn(B, L, 2 * D, device="cuda", dtype=dtype)
            with torch.no_grad():
                ref = b.scan(x).float()
                out = fused_block_scan(b, x).float()
            err = ((out - ref).abs() / (ref.abs() + 1)).max().item()
            assert err < tol, (dtype, (B, L, D, N), err)
    print("ok  fused selective scan matches the PyTorch loop: 5 shapes x fp32/bf16")


def test_int8_dwconv():
    torch.manual_seed(0)
    shapes = [(1, 96, 112, 112, 3, 2), (8, 144, 56, 56, 5, 2), (2, 240, 28, 28, 5, 1), (64, 480, 14, 14, 3, 1),
              (4, 672, 14, 14, 5, 2), (64, 1152, 7, 7, 5, 1), (3, 33, 17, 9, 3, 2), (2, 7, 9, 9, 5, 1)]
    for (N, C, H, W, K, S) in shapes:
        conv = nn.Conv2d(C, C, K, S, K // 2, groups=C, bias=True).cuda()
        x = torch.randn(N, C, H, W, device="cuda") * 2
        sx = x.abs().max().item() / 127
        xq = torch.round(x / sx).clamp(-127, 127).to(torch.int8)
        wq, ws = quantize_dw_weight(conv.weight)
        xq_nhwc = xq.permute(0, 2, 3, 1).contiguous()
        wdq = (wq.t().float() * ws[:, None]).reshape(C, 1, K, K)
        with torch.no_grad():
            ref = F.silu(F.conv2d(xq.float() * sx, wdq, conv.bias, S, K // 2, groups=C))
            out32 = int8_dwconv_nhwc(xq_nhwc, sx, wq, ws, conv.bias.detach().float().contiguous(), S, K // 2, K, True, torch.float32)
            out16 = int8_dwconv_nhwc(xq_nhwc, sx, wq, ws, conv.bias.detach().float().contiguous(), S, K // 2, K, True, torch.bfloat16)
        e32 = ((out32.permute(0, 3, 1, 2) - ref).abs() / (ref.abs() + 1)).max().item()
        e16 = ((out16.permute(0, 3, 1, 2).float() - ref).abs() / (ref.abs() + 1)).max().item()
        assert e32 < 1e-4 and e16 < 1e-2, ((N, C, H, W, K, S), e32, e16)
    print(f"ok  int8-storage depthwise conv matches PyTorch on dequantized values: {len(shapes)} shapes, fp32 out < 1e-4, bf16 out < 1e-2")


def test_whole_model():
    torch.manual_seed(0)
    x = torch.randn(2, 30, 3, 224, 224, device="cuda", dtype=torch.float32)
    m = load_detector(torch.float32)
    with torch.no_grad():
        ref = m(x)
        a = patch_dw(m)
        s = patch_scan(m)
        out = m(x)
    err = (out - ref).abs().max().item()
    assert a == 16 and s == 4 and err < 1e-3, (a, s, err)
    print(f"ok  whole detector with both kernels: {a} dw blocks + {s} scans patched, max logit diff {err:.2e}")


if __name__ == "__main__":
    test_dwconv()
    test_scan()
    test_int8_dwconv()
    test_whole_model()
