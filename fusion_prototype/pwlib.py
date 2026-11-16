"""1x1 conv micro-benchmarks, EfficientNet-B0 @256 px, batch 64, fp16, channels_last.
Compares: cuDNN conv2d (+bias) + separate SiLU, torch.matmul/addmm on NHWC view + SiLU,
Triton GEMM with fused bias+SiLU (expand) / fused SE-scale prologue + residual epilogue (project)."""
import json, sys
import torch, torch.nn.functional as F
import triton, triton.language as tl

torch.backends.cudnn.benchmark = True
dev, dt = "cuda", torch.float16
B = 64

# (stage, cin, cout, H) for the 15 expand convs; project convs (cin, cout, H, residual)
EXP = [("s2b0", 16, 96, 128), ("s2b1", 24, 144, 64), ("s3b0", 24, 144, 64), ("s3b1", 40, 240, 32),
       ("s4b0", 40, 240, 32), ("s4b1", 80, 480, 16), ("s4b2", 80, 480, 16), ("s5b0", 80, 480, 16),
       ("s5b1", 112, 672, 16), ("s5b2", 112, 672, 16), ("s6b0", 112, 672, 16), ("s6b1", 192, 1152, 8),
       ("s6b2", 192, 1152, 8), ("s6b3", 192, 1152, 8), ("s7b0", 192, 1152, 8)]
# project: input is dw output (spatial after stride)
PRJ = [("s1b0", 32, 16, 128, False), ("s2b0", 96, 24, 64, False), ("s2b1", 144, 24, 64, True),
       ("s3b0", 144, 40, 32, False), ("s3b1", 240, 40, 32, True), ("s4b0", 240, 80, 16, False),
       ("s4b1", 480, 80, 16, True), ("s4b2", 480, 80, 16, True), ("s5b0", 480, 112, 16, False),
       ("s5b1", 672, 112, 16, True), ("s5b2", 672, 112, 16, True), ("s6b0", 672, 192, 8, False),
       ("s6b1", 1152, 192, 8, True), ("s6b2", 1152, 192, 8, True), ("s6b3", 1152, 192, 8, True),
       ("s7b0", 1152, 320, 8, False)]


def bench(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(3):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / iters)
    return best


CFGS = [triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=st)
        for bm in (64, 128, 256) for bn in (32, 64, 128) for w in (4, 8) for st in (3,)]


@triton.autotune(CFGS, key=["M", "N", "K"])
@triton.jit
def pw_kernel(a_ptr, w_ptr, b_ptr, s_ptr, r_ptr, y_ptr, M, N, K, HW,
              BK: tl.constexpr, ACT: tl.constexpr, SCALE: tl.constexpr, RES: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
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
        if SCALE:  # SE gate per (image, channel), applied in the prologue
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
    BK = max(16, min(64, triton.next_power_of_2(K)))
    grid = lambda m: (triton.cdiv(M, m["BM"]), triton.cdiv(N, m["BN"]))
    pw_kernel[grid](a2d, wkn, b, scale if scale is not None else b, res if res is not None else y, y,
                    M, N, K, hw, BK=BK, ACT=act, SCALE=scale is not None, RES=res is not None)
    return y


def gbps(nbytes, ms):
    return nbytes / ms / 1e6


