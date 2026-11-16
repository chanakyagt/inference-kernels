import sys
from pathlib import Path

import torch
import torch.nn as nn
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fused_dwconv_bn_silu import fold_bn, is_dw_block


@triton.jit
def _dw_nhwc_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, H, W, Ho, Wo,
                    K: tl.constexpr, S: tl.constexpr, P: tl.constexpr, ACT: tl.constexpr,
                    BP: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    offs = tl.program_id(1) * BP + tl.arange(0, BP)
    cs = tl.program_id(2) * BC + tl.arange(0, BC)
    m = offs < Ho * Wo
    mc = cs < C
    oh = offs // Wo
    ow = offs % Wo
    acc = tl.zeros((BP, BC), dtype=tl.float32) + tl.load(b_ptr + cs, mask=mc, other=0.0)[None, :]
    for kh in tl.static_range(K):
        ih = oh * S - P + kh
        mh = m & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = ow * S - P + kw
            mm = mh & (iw >= 0) & (iw < W)
            v = tl.load(x_ptr + ((n * H + ih) * W + iw)[:, None] * C + cs[None, :],
                        mask=mm[:, None] & mc[None, :], other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + (kh * K + kw) * C + cs, mask=mc, other=0.0)
            acc += v * wv[None, :]
    if ACT:
        acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + (n * Ho * Wo + offs)[:, None] * C + cs[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=m[:, None] & mc[None, :])


def dwconv_bn_silu_nhwc(x, w_khwc, b, stride, padding, act=True):
    N, C, H, W = x.shape
    K = w_khwc.shape[0]
    Ho = (H + 2 * padding - K) // stride + 1
    Wo = (W + 2 * padding - K) // stride + 1
    xl = x.permute(0, 2, 3, 1).contiguous()
    y = torch.empty((N, Ho, Wo, C), device=x.device, dtype=x.dtype)
    BC = min(64, max(16, triton.next_power_of_2(C)))
    BP = 32 if Ho * Wo >= 32 else 16
    grid = (N, triton.cdiv(Ho * Wo, BP), triton.cdiv(C, BC))
    _dw_nhwc_kernel[grid](xl, w_khwc, b, y, C, H, W, Ho, Wo, K=K, S=stride, P=padding, ACT=act,
                          BP=BP, BC=BC, num_warps=4)
    return y.permute(0, 3, 1, 2)


class FusedDWConvBNSiLUNHWC(nn.Module):
    def __init__(self, conv, bn, act=True):
        super().__init__()
        w, b = fold_bn(conv, bn)
        self.register_buffer("w", w.permute(1, 2, 0).contiguous())
        self.register_buffer("b", b)
        self.stride = conv.stride[0]
        self.padding = conv.padding[0]
        self.act = act

    def forward(self, x):
        return dwconv_bn_silu_nhwc(x, self.w, self.b, self.stride, self.padding, self.act)


def patch_model(model):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if is_dw_block(child):
                setattr(mod, cname, FusedDWConvBNSiLUNHWC(child[0], child[1]))
                n += 1
    return n
