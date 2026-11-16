"""Fused model v2: batch-1 launch-count reduction on top of fmodel.py (v1 stays untouched).

New vs v1 (each piece is timed against the v1/PyTorch path on its real shape at build time and falls back if slower):
  dw+pool : depthwise kernel writes per-tile partial channel sums (SE average-pool) in its epilogue
  se_tail : ONE kernel per image: pool finalise -> fc1 -> SiLU -> fc2 -> sigmoid  (replaces mean, 2 gemv, silu, sigmoid + casts)
  stem    : ONE kernel: fp32 NCHW input -> fp16 cast, 3x3 s2 conv, bias, SiLU -> NHWC fp16 (replaces cast, cuDNN conv, silu, permute)
  headtail: ONE kernel: 7x7 mean + fc (replaces mean, gemv, float cast)
"""
import os, sys
import torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl
from fmodel import PW, dw_nhwc, build, size_of, dev, dt, layers as pw_layers

FORCE2 = os.environ.get("FM2_FORCE")  # "new" / "old" for ablations


def bench_graph(fn):
    try:
        return triton.testing.do_bench_cudagraph(fn, rep=40)
    except Exception:
        return triton.testing.do_bench(fn, warmup=10, rep=40)


# ---------------- depthwise with SE-pool partial sums ----------------
@triton.jit
def _dw_pool_kernel(x_ptr, w_ptr, b_ptr, y_ptr, part_ptr, C, H, W, Ho, Wo, PT,
                    K: tl.constexpr, S: tl.constexpr, P: tl.constexpr, BP: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0); pt = tl.program_id(1)
    offs = pt * BP + tl.arange(0, BP)
    cs = tl.program_id(2) * BC + tl.arange(0, BC)
    m = offs < Ho * Wo
    mc = cs < C
    oh = offs // Wo; ow = offs % Wo
    acc = tl.zeros((BP, BC), dtype=tl.float32) + tl.load(b_ptr + cs, mask=mc, other=0.0)[None, :]
    for kh in tl.static_range(K):
        ih = oh * S - P + kh
        mh = m & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = ow * S - P + kw
            mm = mh & (iw >= 0) & (iw < W)
            v = tl.load(x_ptr + ((n * H + ih) * W + iw)[:, None] * C + cs[None, :], mask=mm[:, None] & mc[None, :], other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + (kh * K + kw) * C + cs, mask=mc, other=0.0)
            acc += v * wv[None, :]
    acc = acc * tl.sigmoid(acc)
    o16 = acc.to(tl.float16)
    msk = m[:, None] & mc[None, :]
    tl.store(y_ptr + (n * Ho * Wo + offs)[:, None] * C + cs[None, :], o16, mask=msk)
    ps = tl.sum(tl.where(msk, o16.to(tl.float32), 0.0), axis=0)
    tl.store(part_ptr + (n * PT + pt) * C + cs, ps, mask=mc)


def dw_pool(x, w, b, stride, pad):
    N, H, W, C = x.shape
    K = w.shape[0]
    Ho = (H + 2 * pad - K) // stride + 1; Wo = (W + 2 * pad - K) // stride + 1
    y = torch.empty((N, Ho, Wo, C), device=x.device, dtype=x.dtype)
    BC = min(64, max(16, triton.next_power_of_2(C)))
    BP = 32 if Ho * Wo >= 32 else 16
    PT = triton.cdiv(Ho * Wo, BP)
    part = torch.empty((N, PT, C), device=x.device, dtype=torch.float32)
    _dw_pool_kernel[(N, PT, triton.cdiv(C, BC))](x, w, b, y, part, C, H, W, Ho, Wo, PT, K=K, S=stride, P=pad, BP=BP, BC=BC, num_warps=4)
    return y, part, Ho * Wo


# ---------------- SE tail: pool finalise + fc1 + SiLU + fc2 + sigmoid ----------------
SE_CFGS = [triton.Config({"BC": bc, "BPT": bp}, num_warps=w) for bc in (32, 64, 128, 256) for bp in (8, 32, 128) for w in (2, 4, 8)]


@triton.autotune(SE_CFGS, key=["C", "PT"])
@triton.jit
def se_tail_kernel(part_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, g_ptr, C, CS, PT, inv_hw,
                   BJ: tl.constexpr, BC: tl.constexpr, BPT: tl.constexpr):
    n = tl.program_id(0)
    j = tl.arange(0, BJ); mj = j < CS
    acc1 = tl.zeros((BJ,), dtype=tl.float32)
    for c0 in range(0, C, BC):
        c = c0 + tl.arange(0, BC); mc = c < C
        s = tl.zeros((BC,), dtype=tl.float32)
        for p0 in range(0, PT, BPT):
            p = p0 + tl.arange(0, BPT)
            v = tl.load(part_ptr + (n * PT + p)[:, None] * C + c[None, :], mask=(p < PT)[:, None] & mc[None, :], other=0.0)
            s += tl.sum(v, axis=0)
        s = (s * inv_hw).to(tl.float16).to(tl.float32)
        w1 = tl.load(w1_ptr + j[:, None] * C + c[None, :], mask=mj[:, None] & mc[None, :], other=0.0).to(tl.float32)
        acc1 += tl.sum(w1 * s[None, :], axis=1)
    h = acc1 + tl.load(b1_ptr + j, mask=mj, other=0.0).to(tl.float32)
    h = h.to(tl.float16).to(tl.float32)
    h = h * tl.sigmoid(h)
    h = h.to(tl.float16).to(tl.float32)
    for c0 in range(0, C, BC):
        c = c0 + tl.arange(0, BC); mc = c < C
        w2 = tl.load(w2_ptr + c[:, None] * CS + j[None, :], mask=mc[:, None] & mj[None, :], other=0.0).to(tl.float32)
        g = tl.sum(w2 * h[None, :], axis=1) + tl.load(b2_ptr + c, mask=mc, other=0.0).to(tl.float32)
        g = tl.sigmoid(g.to(tl.float16).to(tl.float32))
        tl.store(g_ptr + n * C + c, g.to(tl.float16), mask=mc)


def se_tail(part, hw, w1, b1, w2, b2):
    N, PT, C = part.shape
    CS = w1.shape[0]
    g = torch.empty((N, C), device=part.device, dtype=dt)
    se_tail_kernel[(N,)](part, w1, b1, w2, b2, g, C, CS, PT, 1.0 / hw, BJ=max(16, triton.next_power_of_2(CS)))
    return g


# ---------------- stem ----------------
@triton.jit
def _stem_kernel(x_ptr, w_ptr, b_ptr, y_ptr, H, W, Ho, Wo, COUT: tl.constexpr, BP: tl.constexpr):
    n = tl.program_id(0)
    offs = tl.program_id(1) * BP + tl.arange(0, BP)
    co = tl.arange(0, COUT)
    m = offs < Ho * Wo
    oh = offs // Wo; ow = offs % Wo
    acc = tl.zeros((BP, COUT), dtype=tl.float32) + tl.load(b_ptr + co).to(tl.float32)[None, :]
    for kh in tl.static_range(3):
        ih = oh * 2 - 1 + kh
        for kw in tl.static_range(3):
            iw = ow * 2 - 1 + kw
            mm = m & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            for ci in tl.static_range(3):
                v = tl.load(x_ptr + ((n * 3 + ci) * H + ih) * W + iw, mask=mm, other=0.0).to(tl.float16).to(tl.float32)
                wv = tl.load(w_ptr + ((kh * 3 + kw) * 3 + ci) * COUT + co).to(tl.float32)
                acc += v[:, None] * wv[None, :]
    acc = acc.to(tl.float16).to(tl.float32)
    acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + (n * Ho * Wo + offs)[:, None] * COUT + co[None, :], acc.to(tl.float16), mask=m[:, None])


# ---------------- head tail: mean + fc ----------------
@triton.jit
def _head_kernel(x_ptr, w_ptr, b_ptr, o_ptr, HW, C, inv_hw, BHW: tl.constexpr, BC: tl.constexpr):
    n = tl.program_id(0)
    p = tl.arange(0, BHW); mp = p < HW
    a0 = tl.zeros((BC,), dtype=tl.float32); a1 = tl.zeros((BC,), dtype=tl.float32)
    for c0 in range(0, C, BC):
        c = c0 + tl.arange(0, BC); mc = c < C
        v = tl.load(x_ptr + (n * HW + p)[:, None] * C + c[None, :], mask=mp[:, None] & mc[None, :], other=0.0).to(tl.float32)
        mean = (tl.sum(v, axis=0) * inv_hw).to(tl.float16).to(tl.float32)
        a0 += mean * tl.load(w_ptr + c, mask=mc, other=0.0).to(tl.float32)
        a1 += mean * tl.load(w_ptr + C + c, mask=mc, other=0.0).to(tl.float32)
    o0 = (tl.sum(a0, axis=0) + tl.load(b_ptr).to(tl.float32)).to(tl.float16).to(tl.float32)
    o1 = (tl.sum(a1, axis=0) + tl.load(b_ptr + 1).to(tl.float32)).to(tl.float16).to(tl.float32)
    tl.store(o_ptr + n * 2, o0); tl.store(o_ptr + n * 2 + 1, o1)


class Switch:
    """Holds a new and an old implementation; times both once (eager, before graph capture) and keeps the faster."""
    def __init__(self, name):
        self.name, self.impl, self.times = name, None, None

    def decide(self, new, old):
        if FORCE2 in ("new", "old"):
            self.impl = FORCE2
        else:
            new(); old()
            tn, to = bench_graph(new), bench_graph(old)
            self.times = (tn, to); self.impl = "new" if tn <= to else "old"


class MBConv2(nn.Module):
    def __init__(self, mb, name):
        super().__init__()
        blk = list(mb.block)
        self.expand = PW(blk[0][0], True, name + ".expand") if len(blk) == 4 else None
        dw, se, prj = blk[-3:]
        self.dw_w, self.dw_b, self.stride, self.pad = dw.w, dw.b, dw.stride, dw.padding
        self.w1 = se.fc1.weight.detach()[:, :, 0, 0].contiguous(); self.b1 = se.fc1.bias.detach().contiguous()
        self.w2 = se.fc2.weight.detach()[:, :, 0, 0].contiguous(); self.b2 = se.fc2.bias.detach().contiguous()
        self.project = PW(prj[0], False, name + ".project")
        self.res = mb.use_res_connect
        self.sw = Switch(name)
        self.name = name

    def se_old(self, h):
        s = h.mean((1, 2))
        s = F.silu(F.linear(s, self.w1, self.b1))
        return torch.sigmoid(F.linear(s, self.w2, self.b2))

    def dw_se_new(self, h):
        y, part, hw = dw_pool(h, self.dw_w, self.dw_b, self.stride, self.pad)
        return y, se_tail(part, hw, self.w1, self.b1, self.w2, self.b2)

    def dw_se_old(self, h):
        y = dw_nhwc(h, self.dw_w, self.dw_b, self.stride, self.pad)
        return y, self.se_old(y)

    def forward(self, x):
        h = self.expand(x) if self.expand is not None else x
        if self.sw.impl is None:
            self.sw.decide(lambda: self.dw_se_new(h), lambda: self.dw_se_old(h))
        y, s = (self.dw_se_new if self.sw.impl == "new" else self.dw_se_old)(h)
        return self.project(y, scale=s, res=x if self.res else None)


class FusedNet2(nn.Module):
    def __init__(self, net):
        super().__init__()
        f = net.trunk[0]
        self.stem_conv = f[0]
        c = f[0][0]
        self.stem_w = c.weight.detach().permute(2, 3, 1, 0).reshape(27, -1).contiguous()  # [(kh,kw,ci), cout]
        self.stem_b = c.bias.detach().contiguous()
        self.blocks = nn.ModuleList([MBConv2(mb, f"s{si}b{bi}") for si in range(1, 8) for bi, mb in enumerate(f[si])])
        self.headconv = PW(f[8][0], True, "headconv")
        self.fc = net.head[1]
        self.sw_stem, self.sw_head = Switch("stem"), Switch("headtail")

    def stem_new(self, x):
        N, _, H, W = x.shape
        Ho, Wo = H // 2, W // 2
        y = torch.empty((N, Ho, Wo, self.stem_w.shape[1]), device=x.device, dtype=dt)
        BP = 64
        _stem_kernel[(N, triton.cdiv(Ho * Wo, BP))](x, self.stem_w, self.stem_b, y, H, W, Ho, Wo, COUT=self.stem_w.shape[1], BP=BP, num_warps=4)
        return y

    def stem_old(self, x):
        x = x.contiguous(memory_format=torch.channels_last).to(dt)
        return self.stem_conv(x).permute(0, 2, 3, 1).contiguous()

    def head_new(self, x):
        N, H, W, C = x.shape
        o = torch.empty((N, 2), device=x.device, dtype=torch.float32)
        _head_kernel[(N,)](x, self.fc.weight, self.fc.bias, o, H * W, C, 1.0 / (H * W), BHW=triton.next_power_of_2(H * W), BC=128, num_warps=4)
        return o

    def head_old(self, x):
        return self.fc(x.mean((1, 2))).float()

    def forward(self, x):  # x: fp32 NCHW
        if self.sw_stem.impl is None:
            self.sw_stem.decide(lambda: self.stem_new(x), lambda: self.stem_old(x))
        h = (self.stem_new if self.sw_stem.impl == "new" else self.stem_old)(x)
        for b in self.blocks:
            h = b(h)
        h = self.headconv(h)
        if self.sw_head.impl is None:
            self.sw_head.decide(lambda: self.head_new(h), lambda: self.head_old(h))
        return (self.head_new if self.sw_head.impl == "new" else self.head_old)(h)


class FusedRunner2:
    def __init__(self, which, bs):
        torch.backends.cudnn.benchmark = True
        self.bs, self.size = bs, size_of(which)
        self.ref, _ = build(which)
        self.fn = FusedNet2(self.ref)
        self.static_in = torch.randn(bs, 3, self.size, self.size, device="cuda")
        with torch.no_grad():
            self.fn(self.static_in)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.fn(self.static_in)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.fn(self.static_in)
        self.decisions = {m.name: (m.impl, m.times) for m in pw_layers(self.fn)}
        self.decisions2 = {b.name: (b.sw.impl, b.sw.times) for b in self.fn.blocks}
        self.decisions2["stem"] = (self.fn.sw_stem.impl, self.fn.sw_stem.times)
        self.decisions2["headtail"] = (self.fn.sw_head.impl, self.fn.sw_head.times)

    @torch.no_grad()
    def run(self, x):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out
