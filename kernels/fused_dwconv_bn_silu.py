import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _dw_kernel(x_ptr, w_ptr, b_ptr, y_ptr,
               C, H, W, Ho, Wo,
               K: tl.constexpr, S: tl.constexpr, P: tl.constexpr,
               ACT: tl.constexpr, BLOCK: tl.constexpr):
    nc = tl.program_id(0)
    pid = tl.program_id(1)
    c = nc % C
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < Ho * Wo
    oh = offs // Wo
    ow = offs % Wo
    acc = tl.zeros((BLOCK,), dtype=tl.float32) + tl.load(b_ptr + c)
    xp = x_ptr + nc * H * W
    for kh in tl.static_range(K):
        ih = oh * S - P + kh
        mh = m & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = ow * S - P + kw
            mm = mh & (iw >= 0) & (iw < W)
            v = tl.load(xp + ih * W + iw, mask=mm, other=0.0).to(tl.float32)
            acc += v * tl.load(w_ptr + c * K * K + kh * K + kw)
    if ACT:
        acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + nc * Ho * Wo + offs, acc.to(y_ptr.dtype.element_ty), mask=m)


def fold_bn(conv, bn):
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    w = conv.weight.float()[:, 0] * scale[:, None, None]
    b = bn.bias.float() - bn.running_mean.float() * scale
    if conv.bias is not None:
        b = b + conv.bias.float() * scale
    return w.contiguous(), b.contiguous()


def dwconv_bn_silu(x, w, b, stride, padding, act=True, block=None):
    N, C, H, W = x.shape
    K = w.shape[-1]
    Ho = (H + 2 * padding - K) // stride + 1
    Wo = (W + 2 * padding - K) // stride + 1
    x = x.contiguous()
    y = torch.empty((N, C, Ho, Wo), device=x.device, dtype=x.dtype)
    if block is None:
        block = max(32, min(512, triton.next_power_of_2(Ho * Wo)))
    warps = max(1, min(4, block // 64))
    grid = (N * C, triton.cdiv(Ho * Wo, block))
    _dw_kernel[grid](x, w, b, y, C, H, W, Ho, Wo, K=K, S=stride, P=padding, ACT=act, BLOCK=block, num_warps=warps)
    return y


class FusedDWConvBNSiLU(nn.Module):
    def __init__(self, conv, bn, act=True):
        super().__init__()
        w, b = fold_bn(conv, bn)
        self.register_buffer("w", w)
        self.register_buffer("b", b)
        self.stride = conv.stride[0]
        self.padding = conv.padding[0]
        self.act = act

    def forward(self, x):
        return dwconv_bn_silu(x, self.w, self.b, self.stride, self.padding, self.act)


def is_dw_block(mod):
    return isinstance(mod, nn.Sequential) and len(mod) == 3 and isinstance(mod[0], nn.Conv2d) \
        and mod[0].groups == mod[0].in_channels > 1 and isinstance(mod[1], nn.BatchNorm2d) \
        and isinstance(mod[2], nn.SiLU)


def patch_model(model):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if is_dw_block(child):
                setattr(mod, cname, FusedDWConvBNSiLU(child[0], child[1]))
                n += 1
    return n
