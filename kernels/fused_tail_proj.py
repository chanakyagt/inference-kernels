import torch
import triton
import triton.language as tl


@triton.jit
def _tail_pool_kernel(x_ptr, w_ptr, b_ptr, pool_ptr, wp_ptr, out_ptr,
                      CIN: tl.constexpr, COUT: tl.constexpr, DM: tl.constexpr, sxb, sxc, sxp, inv_p,
                      P: tl.constexpr, PB: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr,
                      BD: tl.constexpr, FUSE_PROJ: tl.constexpr, IPREC: tl.constexpr):
    b = tl.program_id(0)
    pn = tl.program_id(1)
    offs_p = tl.arange(0, PB)
    mp = offs_p < P
    offs_n = pn * BN + tl.arange(0, BN)
    mn = offs_n < COUT
    acc = tl.zeros((PB, BN), dtype=tl.float32)
    for k0 in tl.static_range(0, CIN, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < CIN
        xt = tl.load(x_ptr + b * sxb + offs_p[:, None] * sxp + offs_k[None, :] * sxc,
                     mask=mp[:, None] & mk[None, :], other=0.0)
        wt = tl.load(w_ptr + offs_n[None, :] * CIN + offs_k[:, None],
                     mask=mn[None, :] & mk[:, None], other=0.0)
        acc = tl.dot(xt, wt, acc, input_precision=IPREC)
    bias = tl.load(b_ptr + offs_n, mask=mn, other=0.0)
    acc = acc + bias[None, :]
    acc = acc * tl.sigmoid(acc)
    acc = tl.where(mp[:, None], acc, 0.0)
    pooled = tl.sum(acc, axis=0) * inv_p
    if FUSE_PROJ:
        for d0 in tl.static_range(0, DM, BD):
            offs_d = d0 + tl.arange(0, BD)
            md = offs_d < DM
            wp = tl.load(wp_ptr + offs_d[:, None] * COUT + offs_n[None, :],
                         mask=md[:, None] & mn[None, :], other=0.0).to(tl.float32)
            part = tl.sum(wp * pooled[None, :], axis=1)
            tl.atomic_add(out_ptr + b * DM + offs_d, part, mask=md)
    else:
        tl.store(pool_ptr + b * COUT + offs_n, pooled, mask=mn)


@triton.jit
def _proj_kernel(pool_ptr, wp_ptr, bp_ptr, out_ptr, COUT: tl.constexpr, DM: tl.constexpr,
                 BD: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    offs_d = tl.program_id(1) * BD + tl.arange(0, BD)
    md = offs_d < DM
    acc = tl.zeros((BD,), dtype=tl.float32)
    for k0 in tl.static_range(0, COUT, BK):
        offs_k = k0 + tl.arange(0, BK)
        mk = offs_k < COUT
        pk = tl.load(pool_ptr + b * COUT + offs_k, mask=mk, other=0.0)
        wt = tl.load(wp_ptr + offs_d[:, None] * COUT + offs_k[None, :],
                     mask=md[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wt * pk[None, :], axis=1)
    acc += tl.load(bp_ptr + offs_d, mask=md, other=0.0)
    tl.store(out_ptr + b * DM + offs_d, acc, mask=md)


def fold_bn_1x1(conv, bn):
    scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
    w = conv.weight.float()[:, :, 0, 0] * scale[:, None]
    b = bn.bias.float() - bn.running_mean.float() * scale
    if conv.bias is not None:
        b = b + conv.bias.float() * scale
    return w.contiguous(), b.contiguous()


class FusedTailProj:
    def __init__(self, conv, bn, proj, dtype=torch.float32, mode="two", bn_tile=32, proj_bd=32, proj_bk=128, fuse_bd=64, bk=64, nw=4):
        w, b = fold_bn_1x1(conv, bn)
        self.dtype = dtype
        self.w = w.to(dtype).contiguous()
        self.b = b.contiguous()
        self.wp = proj.weight.detach().to(dtype).contiguous()
        self.bp = proj.bias.detach().float().contiguous()
        self.cout, self.cin = self.w.shape
        self.dm = self.wp.shape[0]
        self.mode = mode
        self.bn_tile = bn_tile
        self.proj_bd = proj_bd
        self.proj_bk = proj_bk
        self.fuse_bd = fuse_bd
        self.iprec = "ieee" if dtype == torch.float32 else "tf32"
        self.bk, self.nw = bk, nw
        self._pool = {}

    def _buf(self, B, dev):
        if B not in self._pool:
            self._pool[B] = torch.empty(B, self.cout, device=dev, dtype=torch.float32)
        return self._pool[B]

    def pooled(self, x):
        B, C, H, W = x.shape
        x = x.to(self.dtype)
        out = self._buf(B, x.device)
        P = H * W
        sxp = x.stride(3)
        _tail_pool_kernel[(B, triton.cdiv(self.cout, self.bn_tile))](
            x, self.w, self.b, out, self.wp, out, self.cin, self.cout, self.dm,
            x.stride(0), x.stride(1), sxp, 1.0 / P,
            P=P, PB=triton.next_power_of_2(P), BK=self.bk, BN=self.bn_tile, BD=self.fuse_bd,
            FUSE_PROJ=False, IPREC=self.iprec, num_warps=self.nw)
        return out

    def __call__(self, x, out=None):
        B, C, H, W = x.shape
        x = x.to(self.dtype)
        if out is None:
            out = torch.empty(B, self.dm, device=x.device, dtype=torch.float32)
        P = H * W
        grid = (B, triton.cdiv(self.cout, self.bn_tile))
        common = dict(P=P, PB=triton.next_power_of_2(P), BK=self.bk, BN=self.bn_tile, BD=self.fuse_bd, IPREC=self.iprec, num_warps=self.nw)
        if self.mode == "atomic":
            out.copy_(self.bp.expand(B, -1))
            _tail_pool_kernel[grid](x, self.w, self.b, out, self.wp, out, self.cin, self.cout, self.dm,
                                    x.stride(0), x.stride(1), x.stride(3), 1.0 / P, FUSE_PROJ=True, **common)
        else:
            pool = self._buf(B, x.device)
            _tail_pool_kernel[grid](x, self.w, self.b, pool, self.wp, out, self.cin, self.cout, self.dm,
                                    x.stride(0), x.stride(1), x.stride(3), 1.0 / P, FUSE_PROJ=False, **common)
            _proj_kernel[(B, triton.cdiv(self.dm, self.proj_bd))](
                pool, self.wp, self.bp, out, self.cout, self.dm, BD=self.proj_bd, BK=self.proj_bk, num_warps=4)
        return out


def tail_reference(x, conv, bn, proj):
    import torch.nn.functional as F
    y = F.silu(bn(conv(x)))
    return proj(y.mean((2, 3)))
