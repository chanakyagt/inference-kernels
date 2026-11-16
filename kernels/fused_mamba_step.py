import torch
import triton
import triton.language as tl


@triton.jit
def _norm_inproj_conv_kernel(x_ptr, g_ptr, be_ptr, win_ptr, cw_ptr, cb_ptr, cs_ptr,
                             u_ptr, zs_ptr, xcopy_ptr,
                             D: tl.constexpr, E: tl.constexpr, eps,
                             DP: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    pe = tl.program_id(1)
    e = pe * BE + tl.arange(0, BE)
    me = e < E
    offs = tl.arange(0, DP)
    mo = offs < D
    row = tl.load(x_ptr + b * D + offs, mask=mo, other=0.0).to(tl.float32)
    mean = tl.sum(row, axis=0) / D
    xc = tl.where(mo, row - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / D + eps)
    if pe == 0:
        tl.store(xcopy_ptr + b * D + offs, row, mask=mo)
    acc_u = tl.zeros((BE,), dtype=tl.float32)
    acc_z = tl.zeros((BE,), dtype=tl.float32)
    for k0 in tl.static_range(0, D, BK):
        ko = k0 + tl.arange(0, BK)
        mk = ko < D
        xk = tl.load(x_ptr + b * D + ko, mask=mk, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + ko, mask=mk, other=0.0)
        bb = tl.load(be_ptr + ko, mask=mk, other=0.0)
        xn = tl.where(mk, (xk - mean) * rstd * g + bb, 0.0)
        wu = tl.load(win_ptr + e[:, None] * D + ko[None, :], mask=me[:, None] & mk[None, :], other=0.0).to(tl.float32)
        wz = tl.load(win_ptr + (E + e)[:, None] * D + ko[None, :], mask=me[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc_u += tl.sum(wu * xn[None, :], axis=1)
        acc_z += tl.sum(wz * xn[None, :], axis=1)
    cs = cs_ptr + b * 3 * E
    s0 = tl.load(cs + e, mask=me, other=0.0)
    s1 = tl.load(cs + E + e, mask=me, other=0.0)
    s2 = tl.load(cs + 2 * E + e, mask=me, other=0.0)
    w0 = tl.load(cw_ptr + e * 4, mask=me, other=0.0)
    w1 = tl.load(cw_ptr + e * 4 + 1, mask=me, other=0.0)
    w2 = tl.load(cw_ptr + e * 4 + 2, mask=me, other=0.0)
    w3 = tl.load(cw_ptr + e * 4 + 3, mask=me, other=0.0)
    cb = tl.load(cb_ptr + e, mask=me, other=0.0)
    c = cb + w0 * s0 + w1 * s1 + w2 * s2 + w3 * acc_u
    tl.debug_barrier()
    tl.store(cs + e, s1, mask=me)
    tl.store(cs + E + e, s2, mask=me)
    tl.store(cs + 2 * E + e, acc_u, mask=me)
    tl.store(u_ptr + b * E + e, c * tl.sigmoid(c), mask=me)
    tl.store(zs_ptr + b * E + e, acc_z * tl.sigmoid(acc_z), mask=me)


@triton.jit
def _xproj_dt_scan_kernel(u_ptr, zs_ptr, wx_ptr, wdt_ptr, bdt_ptr, A_ptr, Dp_ptr, h_ptr,
                          wout_ptr, xnext_ptr, yg_ptr,
                          D: tl.constexpr, E: tl.constexpr,
                          N: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr, BD: tl.constexpr,
                          FUSE_OUT: tl.constexpr):
    b = tl.program_id(0)
    pe = tl.program_id(1)
    e = pe * BE + tl.arange(0, BE)
    me = e < E
    n = tl.arange(0, N)
    accB = tl.zeros((N,), dtype=tl.float32)
    accC = tl.zeros((N,), dtype=tl.float32)
    accd = tl.zeros((BE,), dtype=tl.float32)
    for k0 in tl.static_range(0, E, BK):
        ko = k0 + tl.arange(0, BK)
        mk = ko < E
        uk = tl.load(u_ptr + b * E + ko, mask=mk, other=0.0)
        wb = tl.load(wx_ptr + n[:, None] * E + ko[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        wc = tl.load(wx_ptr + (N + n)[:, None] * E + ko[None, :], mask=mk[None, :], other=0.0).to(tl.float32)
        accB += tl.sum(wb * uk[None, :], axis=1)
        accC += tl.sum(wc * uk[None, :], axis=1)
        wd = tl.load(wdt_ptr + e[:, None] * E + ko[None, :], mask=me[:, None] & mk[None, :], other=0.0).to(tl.float32)
        accd += tl.sum(wd * uk[None, :], axis=1)
    pre = accd + tl.load(bdt_ptr + e, mask=me, other=0.0)
    s = 1.0 + tl.exp(tl.minimum(pre, 20.0))
    sp = tl.where(s == 1.0, tl.exp(pre), tl.log(s) * tl.exp(tl.minimum(pre, 20.0)) / (s - 1.0))
    dt = tl.where(pre > 20.0, pre, sp)
    ue = tl.load(u_ptr + b * E + e, mask=me, other=0.0)
    ze = tl.load(zs_ptr + b * E + e, mask=me, other=0.0)
    A = tl.load(A_ptr + e[:, None] * N + n[None, :], mask=me[:, None], other=0.0)
    hp = h_ptr + (b * E + e)[:, None] * N + n[None, :]
    h = tl.load(hp, mask=me[:, None], other=0.0)
    h = h * tl.exp(dt[:, None] * A) + (dt * ue)[:, None] * accB[None, :]
    tl.store(hp, h, mask=me[:, None])
    y = tl.sum(h * accC[None, :], axis=1) + tl.load(Dp_ptr + e, mask=me, other=0.0) * ue
    yg = tl.where(me, y * ze, 0.0)
    if FUSE_OUT:
        for d0 in tl.static_range(0, D, BD):
            d = d0 + tl.arange(0, BD)
            md = d < D
            wo = tl.load(wout_ptr + d[:, None] * E + e[None, :], mask=md[:, None] & me[None, :], other=0.0).to(tl.float32)
            tl.atomic_add(xnext_ptr + b * D + d, tl.sum(wo * yg[None, :], axis=1), mask=md)
    else:
        tl.store(yg_ptr + b * E + e, yg, mask=me)


@triton.jit
def _outproj_residual_kernel(yg_ptr, wout_ptr, xnext_ptr, D: tl.constexpr, E: tl.constexpr,
                             BD: tl.constexpr, BK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    md = d < D
    acc = tl.load(xnext_ptr + b * D + d, mask=md, other=0.0)
    for k0 in tl.static_range(0, E, BK):
        ko = k0 + tl.arange(0, BK)
        mk = ko < E
        yk = tl.load(yg_ptr + b * E + ko, mask=mk, other=0.0)
        wo = tl.load(wout_ptr + d[:, None] * E + ko[None, :], mask=md[:, None] & mk[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(wo * yk[None, :], axis=1)
    tl.store(xnext_ptr + b * D + d, acc, mask=md)


@triton.jit
def _head_kernel(x_ptr, g_ptr, be_ptr, ysum_ptr, cnt_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr,
                 D: tl.constexpr, H: tl.constexpr, eps,
                 DP: tl.constexpr, BH: tl.constexpr):
    b = tl.program_id(0)
    offs = tl.arange(0, DP)
    mo = offs < D
    row = tl.load(x_ptr + b * D + offs, mask=mo, other=0.0).to(tl.float32)
    mean = tl.sum(row, axis=0) / D
    xc = tl.where(mo, row - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(xc * xc, axis=0) / D + eps)
    g = tl.load(g_ptr + offs, mask=mo, other=0.0)
    bb = tl.load(be_ptr + offs, mask=mo, other=0.0)
    nrm = xc * rstd * g + bb
    ys = tl.load(ysum_ptr + b * D + offs, mask=mo, other=0.0) + nrm
    tl.store(ysum_ptr + b * D + offs, ys, mask=mo)
    cnt = tl.load(cnt_ptr + b) + 1.0
    tl.debug_barrier()
    tl.store(cnt_ptr + b, cnt)
    m = tl.where(mo, ys / cnt, 0.0)
    o0 = tl.zeros((BH,), dtype=tl.float32)
    o1 = tl.zeros((BH,), dtype=tl.float32)
    for h0 in tl.static_range(0, H, BH):
        hh = h0 + tl.arange(0, BH)
        mh = hh < H
        w1 = tl.load(w1_ptr + hh[:, None] * D + offs[None, :], mask=mh[:, None] & mo[None, :], other=0.0).to(tl.float32)
        hid = tl.sum(w1 * m[None, :], axis=1) + tl.load(b1_ptr + hh, mask=mh, other=0.0)
        hid = 0.5 * hid * (1.0 + tl.erf(hid * 0.7071067811865476))
        hid = tl.where(mh, hid, 0.0)
        o0 += hid * tl.load(w2_ptr + hh, mask=mh, other=0.0).to(tl.float32)
        o1 += hid * tl.load(w2_ptr + H + hh, mask=mh, other=0.0).to(tl.float32)
    tl.store(out_ptr + b * 2, tl.sum(o0, axis=0) + tl.load(b2_ptr))
    tl.store(out_ptr + b * 2 + 1, tl.sum(o1, axis=0) + tl.load(b2_ptr + 1))


class FusedMambaStack:
    def __init__(self, temporal, batch, dtype=torch.float32, fuse_out=True, be1=None, be2=None, bk=None, bd=64, nw1=None, nw2=None, hbh=32, hnw=8):
        dev = next(temporal.parameters()).device
        self.dev, self.B, self.dtype = dev, batch, dtype
        self.fuse_out = fuse_out
        mix0 = temporal.layers[0].mixer
        self.D, self.E, self.N = mix0.d_model, mix0.d_inner, mix0.d_state
        assert mix0.conv1d.kernel_size[0] == 4 and (self.N & (self.N - 1)) == 0
        small = batch <= 2
        be1 = be1 or (16 if small else 32)
        be2 = be2 or (16 if small else 32)
        bk = bk or (128 if small else 64)
        nw1 = nw1 or 4
        nw2 = nw2 or (8 if small else 4)
        self.be1, self.be2, self.bk, self.bd = be1, be2, bk, bd
        self.nw1, self.nw2 = nw1, nw2
        self.hbh, self.hnw = hbh, hnw
        self.DP = triton.next_power_of_2(self.D)
        wd = lambda t: t.detach().to(dtype).contiguous()
        f32 = lambda t: t.detach().float().contiguous()
        self.layers = []
        for L in temporal.layers:
            m = L.mixer
            self.layers.append(dict(
                g=f32(L.norm.weight), be=f32(L.norm.bias), eps=L.norm.eps,
                win=wd(m.in_proj.weight), cw=f32(m.conv1d.weight[:, 0, :]), cb=f32(m.conv1d.bias),
                wx=wd(m.x_proj.weight), wdt=wd(m.dt_proj.weight), bdt=f32(m.dt_proj.bias),
                A=(-torch.exp(m.A_log.detach().float())).contiguous(), Dp=f32(m.D), wout=wd(m.out_proj.weight)))
        self.nf_g, self.nf_b, self.nf_eps = f32(temporal.norm_f.weight), f32(temporal.norm_f.bias), temporal.norm_f.eps
        self.w1, self.b1 = wd(temporal.head[0].weight), f32(temporal.head[0].bias)
        self.w2, self.b2 = wd(temporal.head[3].weight), f32(temporal.head[3].bias)
        self.H = self.w1.shape[0]
        self.alloc(batch)

    def alloc(self, B):
        self.B = B
        d, E, N = self.dev, self.E, self.N
        nl = len(self.layers)
        self.h = [torch.zeros(B, E, N, device=d) for _ in range(nl)]
        self.cs = [torch.zeros(B, 3, E, device=d) for _ in range(nl)]
        self.ysum = torch.zeros(B, self.D, device=d)
        self.cnt = torch.zeros(B, device=d)
        self.xbuf = [torch.empty(B, self.D, device=d) for _ in range(nl + 1)]
        self.ubuf = torch.empty(B, E, device=d)
        self.zbuf = torch.empty(B, E, device=d)
        self.ygbuf = torch.empty(B, E, device=d)
        self.logits = torch.empty(B, 2, device=d)

    def reset(self):
        for t in self.h + self.cs + [self.ysum, self.cnt]:
            t.zero_()

    def state_bytes(self):
        return sum(t.numel() * 4 for t in self.h + self.cs + [self.ysum, self.cnt])

    def layer(self, i, xin, xout):
        p = self.layers[i]
        B, D, E, N = self.B, self.D, self.E, self.N
        _norm_inproj_conv_kernel[(B, triton.cdiv(E, self.be1))](
            xin, p["g"], p["be"], p["win"], p["cw"], p["cb"], self.cs[i], self.ubuf, self.zbuf, xout,
            D, E, p["eps"], DP=self.DP, BE=self.be1, BK=self.bk, num_warps=self.nw1)
        _xproj_dt_scan_kernel[(B, triton.cdiv(E, self.be2))](
            self.ubuf, self.zbuf, p["wx"], p["wdt"], p["bdt"], p["A"], p["Dp"], self.h[i],
            p["wout"], xout, self.ygbuf, D, E,
            N=N, BE=self.be2, BK=self.bk, BD=self.bd, FUSE_OUT=self.fuse_out, num_warps=self.nw2)
        if not self.fuse_out:
            _outproj_residual_kernel[(B, triton.cdiv(D, self.bd))](
                self.ygbuf, p["wout"], xout, D, E, BD=self.bd, BK=self.bk, num_warps=4)

    def step(self, x0):
        self.xbuf[0].copy_(x0)
        return self.step_from_buf0()

    def step_from_buf0(self):
        for i in range(len(self.layers)):
            self.layer(i, self.xbuf[i], self.xbuf[i + 1])
        self.finish(self.xbuf[-1])
        return self.logits

    def finish(self, x):
        _head_kernel[(self.B,)](x, self.nf_g, self.nf_b, self.ysum, self.cnt, self.w1, self.b1, self.w2, self.b2,
                                self.logits, self.D, self.H, self.nf_eps, DP=self.DP, BH=self.hbh, num_warps=self.hnw)

    def last_hidden(self):
        return self.xbuf[-1]


def reference_step_stack(temporal, states, x0):
    import torch.nn.functional as F
    x = x0
    for i, L in enumerate(temporal.layers):
        m = L.mixer
        h, cs = states["h"][i], states["cs"][i]
        u, z = m.in_proj(L.norm(x)).chunk(2, -1)
        win = torch.cat([cs, u.unsqueeze(1)], 1)
        states["cs"][i] = win[:, 1:]
        c = (win * m.conv1d.weight[:, 0, :].t()[None]).sum(1) + m.conv1d.bias
        u = F.silu(c)
        dt = F.softplus(m.dt_proj(u))
        Bm, Cm = m.x_proj(u).chunk(2, -1)
        A = -torch.exp(m.A_log.float())
        h = torch.exp(dt[..., None] * A) * h + dt[..., None] * Bm[:, None, :] * u[..., None]
        states["h"][i] = h
        y = (h * Cm[:, None, :]).sum(-1) + u * m.D
        x = x + m.out_proj(y * F.silu(z))
    return x
