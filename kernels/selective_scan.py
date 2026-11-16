import torch
import triton
import triton.language as tl


@triton.jit
def _scan_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, y_ptr,
                 L, E,
                 sx_b, sx_l, sb_b, sb_l,
                 N: tl.constexpr, BLOCK_E: tl.constexpr):
    b = tl.program_id(0)
    pe = tl.program_id(1)
    e = pe * BLOCK_E + tl.arange(0, BLOCK_E)
    n = tl.arange(0, N)
    me = e < E
    A = tl.load(A_ptr + e[:, None] * N + n[None, :], mask=me[:, None], other=0.0)
    Dv = tl.load(D_ptr + e, mask=me, other=0.0)
    h = tl.zeros((BLOCK_E, N), dtype=tl.float32)
    xb = x_ptr + b * sx_b
    db = dt_ptr + b * sx_b
    yb = y_ptr + b * sx_b
    Bb = B_ptr + b * sb_b
    Cb = C_ptr + b * sb_b
    for t in range(L):
        xv = tl.load(xb + t * sx_l + e, mask=me, other=0.0).to(tl.float32)
        dv = tl.load(db + t * sx_l + e, mask=me, other=0.0).to(tl.float32)
        Bv = tl.load(Bb + t * sb_l + n).to(tl.float32)
        Cv = tl.load(Cb + t * sb_l + n).to(tl.float32)
        h = h * tl.exp(dv[:, None] * A) + (dv * xv)[:, None] * Bv[None, :]
        yv = tl.sum(h * Cv[None, :], axis=1) + Dv * xv
        tl.store(yb + t * sx_l + e, yv.to(y_ptr.dtype.element_ty), mask=me)


def selective_scan_fused(x, delta, A, Bm, Cm, D, block_e=64):
    Bsz, L, E = x.shape
    N = A.shape[1]
    x = x.contiguous()
    delta = delta.contiguous()
    Bm = Bm.contiguous()
    Cm = Cm.contiguous()
    A = A.float().contiguous()
    D = D.float().contiguous()
    y = torch.empty_like(x)
    grid = (Bsz, triton.cdiv(E, block_e))
    _scan_kernel[grid](x, delta, A, Bm, Cm, D, y, L, E,
                       x.stride(0), x.stride(1), Bm.stride(0), Bm.stride(1),
                       N=N, BLOCK_E=block_e, num_warps=4)
    return y


def fused_block_scan(block, x):
    u, delta, A, Bm, Cm, D = block.ssm_inputs(x)
    return selective_scan_fused(u, delta, A, Bm, Cm, D).to(x.dtype)


def patch_model(model):
    from dfmamba.ssm import SelectiveSSM
    n = 0
    for mod in model.modules():
        if isinstance(mod, SelectiveSSM):
            mod.scan = (lambda x, mod=mod: fused_block_scan(mod, x))
            n += 1
    return n
