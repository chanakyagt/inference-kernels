"""Fused fp16 EfficientNet-B0 forward (own Triton GEMM epilogues/prologues + own Triton NHWC depthwise).

Everything runs on NHWC [N,H,W,C] contiguous fp16 tensors.
  expand 1x1   : Triton GEMM, bias + SiLU in the epilogue
  depthwise    : existing Triton NHWC kernel (kernels/fused_dwconv_nhwc.py)
  SE           : PyTorch (mean + 2 tiny linears)
  project 1x1  : Triton GEMM, SE gate applied while loading A (prologue), bias + residual in the epilogue
Each 1x1 layer is timed against the unfused cuDNN path (conv+bias, then SiLU / scale / add) on its real shape
at build time and falls back to cuDNN if Triton is not faster (recorded in .decisions).
"""
import os, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trained")); sys.path.insert(0, str(ROOT / "kernels"))
from fused_dwconv_nhwc import _dw_nhwc_kernel, patch_model as patch_dw_nhwc
from tcommon import fold_conv_bn, load_model, size_of

dev, dt = "cuda", torch.float16


def _prune(configs, named_args, **kw):
    M = named_args["M"]
    cap = max(32, triton.next_power_of_2(M))
    out = [c for c in configs if c.kwargs["BM"] <= cap and not (c.kwargs["BM"] * c.kwargs["BN"] > 256 * 64 and c.num_warps == 4)]
    return out or configs[:1]


CFGS = [triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bm in (32, 64, 128, 256) for bn in (32, 64, 128) for bk in (32, 64) for w in (4, 8) for st in (3, 4)]


@triton.autotune(CFGS, key=["M", "N", "K", "SCALE", "RES", "ACT"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def pw_kernel(a_ptr, w_ptr, b_ptr, s_ptr, r_ptr, y_ptr, M, N, K, HW,
              ACT: tl.constexpr, SCALE: tl.constexpr, RES: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mm = rm < M; mn = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    img = rm // HW
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < K
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if SCALE:
            s = tl.load(s_ptr + img[:, None] * K + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
            a = (a.to(tl.float32) * s.to(tl.float32)).to(tl.float16)
        w = tl.load(w_ptr + rk[:, None] * N + rn[None, :], mask=mk[:, None] & mn[None, :], other=0.0)
        acc = tl.dot(a, w, acc)
    acc += tl.load(b_ptr + rn, mask=mn, other=0.0).to(tl.float32)[None, :]
    if ACT:
        acc = acc * tl.sigmoid(acc)
    msk = mm[:, None] & mn[None, :]
    if RES:
        acc += tl.load(r_ptr + rm[:, None] * N + rn[None, :], mask=msk, other=0.0).to(tl.float32)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.float16), mask=msk)


def pw_triton(a2d, wkn, b, act, scale=None, res=None, hw=1):
    M, K = a2d.shape; N = wkn.shape[1]
    y = torch.empty((M, N), device=dev, dtype=dt)
    grid = lambda m: (triton.cdiv(M, m["BM"]), triton.cdiv(N, m["BN"]))
    pw_kernel[grid](a2d, wkn, b, scale if scale is not None else b, res if res is not None else y, y,
                    M, N, K, hw, ACT=act, SCALE=scale is not None, RES=res is not None)
    return y


def bench_ms(fn):
    return triton.testing.do_bench(fn, warmup=10, rep=40)


class PW(nn.Module):
    """1x1 conv (BN already folded) with optional SiLU / SE-scale prologue / residual epilogue."""

    def __init__(self, conv, act, name):
        super().__init__()
        self.name, self.act = name, act
        w = conv.weight.detach()[:, :, 0, 0]               # [Cout, Cin]
        self.register_buffer("wkn", w.t().contiguous())     # [Cin, Cout]
        self.register_buffer("w4", conv.weight.detach().contiguous(memory_format=torch.channels_last))
        self.register_buffer("bias", conv.bias.detach().contiguous())
        self.impl, self.times = None, None
        self.force = os.environ.get("FM_FORCE_IMPL")  # "triton" / "cudnn" for ablations

    def f_triton(self, x, scale, res):
        N, H, W, C = x.shape
        y = pw_triton(x.view(-1, C), self.wkn, self.bias, self.act, scale, None if res is None else res.view(-1, res.shape[-1]), H * W)
        return y.view(N, H, W, -1)

    def f_cudnn(self, x, scale, res):
        if scale is not None:
            x = x * scale[:, None, None, :]
        y = F.conv2d(x.permute(0, 3, 1, 2), self.w4, self.bias)
        y = y.permute(0, 2, 3, 1)
        if self.act:
            y = F.silu(y)
        if res is not None:
            y = y + res
        return y.contiguous()

    def forward(self, x, scale=None, res=None):
        if self.impl is None:
            if self.force:
                self.impl = self.force
            else:
                self.f_triton(x, scale, res)  # autotune outside timing
                tt = bench_ms(lambda: self.f_triton(x, scale, res))
                tc = bench_ms(lambda: self.f_cudnn(x, scale, res))
                self.times = (tt, tc)
                self.impl = "triton" if tt <= tc else "cudnn"
        return (self.f_triton if self.impl == "triton" else self.f_cudnn)(x, scale, res)


def dw_nhwc(x, w, b, stride, pad):
    N, H, W, C = x.shape
    K = w.shape[0]
    Ho = (H + 2 * pad - K) // stride + 1; Wo = (W + 2 * pad - K) // stride + 1
    y = torch.empty((N, Ho, Wo, C), device=x.device, dtype=x.dtype)
    BC = min(64, max(16, triton.next_power_of_2(C)))
    BP = 32 if Ho * Wo >= 32 else 16
    grid = (N, triton.cdiv(Ho * Wo, BP), triton.cdiv(C, BC))
    _dw_nhwc_kernel[grid](x, w, b, y, C, H, W, Ho, Wo, K=K, S=stride, P=pad, ACT=True, BP=BP, BC=BC, num_warps=4)
    return y


class FusedMBConv(nn.Module):
    def __init__(self, mb, name):
        super().__init__()
        blk = list(mb.block)
        self.expand = None
        if len(blk) == 4:
            self.expand = PW(blk[0][0], True, name + ".expand")
            blk = blk[1:]
        dw, se, prj = blk
        self.dw_w, self.dw_b, self.stride, self.pad = dw.w, dw.b, dw.stride, dw.padding
        self.se1, self.se2 = se.fc1, se.fc2
        self.project = PW(prj[0], False, name + ".project")
        self.res = mb.use_res_connect

    def se_gate(self, h):
        s = h.mean((1, 2))
        s = F.silu(F.linear(s, self.se1.weight[:, :, 0, 0], self.se1.bias))
        return torch.sigmoid(F.linear(s, self.se2.weight[:, :, 0, 0], self.se2.bias))

    def forward(self, x):
        h = self.expand(x) if self.expand is not None else x
        h = dw_nhwc(h, self.dw_w, self.dw_b, self.stride, self.pad)
        s = self.se_gate(h)
        return self.project(h, scale=s, res=x if self.res else None)


class FusedNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        f = net.trunk[0]
        self.stem = f[0]
        self.blocks = nn.ModuleList([FusedMBConv(mb, f"s{si}b{bi}") for si in range(1, 8) for bi, mb in enumerate(f[si])])
        self.headconv = PW(f[8][0], True, "headconv")
        self.fc = net.head[1]

    def forward(self, x):
        x = self.stem(x).permute(0, 2, 3, 1).contiguous()
        for b in self.blocks:
            x = b(x)
        x = self.headconv(x)
        return self.fc(x.mean((1, 2)))


def build(which):
    net, _ = load_model(which)
    patch_dw_nhwc(net.trunk)
    fold_conv_bn(net.trunk)
    net = net.to(memory_format=torch.channels_last).to(dt)
    return net, FusedNet(net)   # net = unfused R6f modules (same weights)


def layers(fn):
    return [m for m in fn.modules() if isinstance(m, PW)]


class FusedRunner:
    """Same interface as ladder.Runner (run(), graph) so trt_bench.perf()/predict() work."""

    def __init__(self, which, bs):
        torch.backends.cudnn.benchmark = True
        self.bs, self.size = bs, size_of(which)
        self.ref, self.fn = build(which)
        self.static_in = torch.randn(bs, 3, self.size, self.size, device="cuda")
        with torch.no_grad():
            self.fwd(self.static_in)          # tunes + decides impl per 1x1 layer (eager)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.fwd(self.static_in)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.fwd(self.static_in)
        self.decisions = {m.name: (m.impl, m.times) for m in layers(self.fn)}

    def fwd(self, x):
        x = x.contiguous(memory_format=torch.channels_last).to(dt)
        return self.fn(x).float()

    @torch.no_grad()
    def run(self, x):
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out
