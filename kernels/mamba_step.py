import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _step_kernel(x_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, D_ptr, h_ptr, y_ptr, E,
                 N: tl.constexpr, BLOCK_E: tl.constexpr):
    b = tl.program_id(0)
    e = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    n = tl.arange(0, N)
    me = e < E
    A = tl.load(A_ptr + e[:, None] * N + n[None, :], mask=me[:, None], other=0.0)
    Dv = tl.load(D_ptr + e, mask=me, other=0.0)
    hp = h_ptr + b * E * N + e[:, None] * N + n[None, :]
    h = tl.load(hp, mask=me[:, None], other=0.0)
    xv = tl.load(x_ptr + b * E + e, mask=me, other=0.0).to(tl.float32)
    dv = tl.load(dt_ptr + b * E + e, mask=me, other=0.0).to(tl.float32)
    Bv = tl.load(B_ptr + b * N + n).to(tl.float32)
    Cv = tl.load(C_ptr + b * N + n).to(tl.float32)
    h = h * tl.exp(dv[:, None] * A) + (dv * xv)[:, None] * Bv[None, :]
    tl.store(hp, h, mask=me[:, None])
    yv = tl.sum(h * Cv[None, :], axis=1) + Dv * xv
    tl.store(y_ptr + b * E + e, yv.to(y_ptr.dtype.element_ty), mask=me)


def ssm_step_fused(x, delta, A, Bm, Cm, D, h, block_e=64):
    Bsz, E = x.shape
    N = A.shape[1]
    y = torch.empty_like(x)
    _step_kernel[(Bsz, triton.cdiv(E, block_e))](x, delta, A, Bm, Cm, D, h, y, E, N=N, BLOCK_E=block_e, num_warps=4)
    return y


class MambaStreamer:
    def __init__(self, temporal, batch=1, use_triton=True):
        self.t = temporal
        p = next(temporal.parameters())
        self.dev, self.dtype = p.device, p.dtype
        self.use_triton = use_triton
        self.B = batch
        mix = temporal.layers[0].mixer
        self.E, self.N, self.K = mix.d_inner, mix.d_state, mix.conv1d.kernel_size[0]
        self.A = [(-torch.exp(l.mixer.A_log.float())).contiguous() for l in temporal.layers]
        self.D = [l.mixer.D.float().contiguous() for l in temporal.layers]
        self.cw = [l.mixer.conv1d.weight[:, 0, :].contiguous() for l in temporal.layers]
        self.cb = [l.mixer.conv1d.bias for l in temporal.layers]
        self.conv = [torch.zeros(batch, self.E, self.K - 1, device=self.dev, dtype=self.dtype) for _ in temporal.layers]
        self.h = [torch.zeros(batch, self.E, self.N, device=self.dev, dtype=torch.float32) for _ in temporal.layers]
        self.ysum = torch.zeros(batch, temporal.norm_f.weight.shape[0], device=self.dev, dtype=torch.float32)
        self.cnt = torch.zeros((), device=self.dev, dtype=torch.float32)

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.conv + self.h) + self.ysum.numel() * 4

    def reset(self):
        for t in self.conv + self.h:
            t.zero_()
        self.ysum.zero_()
        self.cnt.zero_()

    def layer(self, i, x):
        L = self.t.layers[i]
        m = L.mixer
        u, z = m.in_proj(L.norm(x)).chunk(2, -1)
        win = torch.cat([self.conv[i], u.unsqueeze(-1)], -1)
        self.conv[i].copy_(win[:, :, 1:])
        c = (win.float() * self.cw[i].float()[None]).sum(-1) + self.cb[i].float()
        u = F.silu(c).to(x.dtype)
        delta = F.softplus(m.dt_proj(u).float())
        Bm, Cm = m.x_proj(u).float().chunk(2, -1)
        if self.use_triton:
            y = ssm_step_fused(u.float().contiguous(), delta.contiguous(), self.A[i], Bm.contiguous(),
                               Cm.contiguous(), self.D[i], self.h[i]).to(x.dtype)
        else:
            uf = u.float()
            h = torch.exp(delta[..., None] * self.A[i]) * self.h[i] + delta[..., None] * Bm[:, None, :] * uf[..., None]
            self.h[i].copy_(h)
            y = ((h * Cm[:, None, :]).sum(-1) + uf * self.D[i]).to(x.dtype)
        return x + m.out_proj(y * F.silu(z))

    def step(self, feat):
        x = self.t.proj(feat)
        for i in range(len(self.t.layers)):
            x = self.layer(i, x)
        self.ysum.add_(self.t.norm_f(x).float())
        self.cnt.add_(1)

    def logits(self):
        return self.t.head((self.ysum / self.cnt).to(self.dtype))

    def forward_step(self, feat):
        self.step(feat)
        return self.logits()
