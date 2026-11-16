import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def _gemm_configs():
    out = []
    for bm in (16, 32, 64, 128):
        for bn in (32, 64, 128):
            for bk in (32, 64, 128):
                if bm * bn >= 8192 and bk == 128:
                    continue
                out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=4 if bm * bn >= 2048 else 2, num_stages=3))
    return out


def _prune_gemm(configs, named_args, **kwargs):
    cap = max(16, triton.next_power_of_2(named_args["M"]))
    keep = [c for c in configs if c.kwargs["BM"] <= cap]
    return keep or configs[:1]


@triton.autotune(configs=_gemm_configs(), key=["M", "N", "K", "MODE", "A_Q", "OUT_Q"], warmup=5, rep=25,
                 prune_configs_by={"early_config_prune": _prune_gemm})
@triton.jit
def _gemm(a_ptr, w_ptr, sw_ptr, b_ptr, y_ptr, iso_ptr, M, N, K, sx, inv_sx,
          MODE: tl.constexpr, A_Q: tl.constexpr, ACT: tl.constexpr, OUT_Q: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0)
    pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    nm = rn < N
    if MODE == 1:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        km = rk < K
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=mm[:, None] & km[None, :], other=0)
        w = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=nm[None, :] & km[:, None], other=0)
        if MODE == 1:
            if not A_Q:
                a = libdevice.rint(a.to(tl.float32) * inv_sx)
                a = tl.minimum(tl.maximum(a, -127.0), 127.0).to(tl.int8)
            acc = tl.dot(a, w, acc)
        else:
            acc = tl.dot(a, w, acc)
    f = acc.to(tl.float32)
    if MODE == 1:
        sw = tl.load(sw_ptr + rn, mask=nm, other=0.0)
        f = f * (sw * sx)[None, :]
    b = tl.load(b_ptr + rn, mask=nm, other=0.0)
    f = f + b[None, :]
    if ACT:
        f = f * tl.sigmoid(f)
    if OUT_Q:
        iso = tl.load(iso_ptr + rn, mask=nm, other=1.0)
        q = libdevice.rint(f * iso[None, :])
        q = tl.minimum(tl.maximum(q, -127.0), 127.0).to(tl.int8)
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], q, mask=mm[:, None] & nm[None, :])
    else:
        tl.store(y_ptr + rm[:, None] * N + rn[None, :], f.to(y_ptr.dtype.element_ty), mask=mm[:, None] & nm[None, :])


def gemm_int8(a, w_q, w_scale, bias, sx, act=False, a_is_q=False, out_scale=None):
    M, K = a.shape
    N = w_q.shape[0]
    y = torch.empty((M, N), device=a.device, dtype=torch.int8 if out_scale is not None else torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    iso = bias if out_scale is None else (1.0 / out_scale).float().contiguous()
    _gemm[grid](a, w_q, w_scale, bias, y, iso, M, N, K, float(sx), 1.0 / float(sx),
                MODE=1, A_Q=a_is_q, ACT=act, OUT_Q=out_scale is not None)
    return y


def gemm_bf16(a, w, bias, act=False):
    M, K = a.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]))
    _gemm[grid](a, w, bias, bias, y, bias, M, N, K, 1.0, 1.0, MODE=0, A_Q=False, ACT=act, OUT_Q=False)
    return y


@triton.jit
def _quant(x_ptr, y_ptr, n, inv, FP8: tl.constexpr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    v = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.float32) * inv
    if FP8:
        tl.store(y_ptr + o, tl.minimum(tl.maximum(v, -448.0), 448.0).to(tl.float8e4nv), mask=m)
    else:
        tl.store(y_ptr + o, tl.minimum(tl.maximum(libdevice.rint(v), -127.0), 127.0).to(tl.int8), mask=m)


def quant_act(x, scale, fp8=False):
    y = torch.empty(x.shape, device=x.device, dtype=FP8 if fp8 else torch.int8)
    n = x.numel()
    _quant[(triton.cdiv(n, 4096),)](x, y, n, 1.0 / float(scale), FP8=fp8, BLOCK=4096)
    return y


@triton.jit
def _quant_pc(x_ptr, y_ptr, n, C, inv_ptr, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    inv = tl.load(inv_ptr + o % C, mask=m, other=1.0)
    v = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.float32) * inv
    tl.store(y_ptr + o, tl.minimum(tl.maximum(libdevice.rint(v), -127.0), 127.0).to(tl.int8), mask=m)


def quant_act_pc(x, inv_scale):
    y = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    n = x.numel()
    _quant_pc[(triton.cdiv(n, 4096),)](x, y, n, x.shape[-1], inv_scale, BLOCK=4096)
    return y


@triton.jit
def _dequant(c_ptr, sw_ptr, b_ptr, y_ptr, M, N, sx, ACT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    c = tl.load(c_ptr + rm[:, None] * N + rn[None, :], mask=m, other=0).to(tl.float32)
    sw = tl.load(sw_ptr + rn, mask=rn < N, other=0.0)
    b = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    f = c * (sw * sx)[None, :] + b[None, :]
    if ACT:
        f = f * tl.sigmoid(f)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], f.to(y_ptr.dtype.element_ty), mask=m)


def pad_rows(a, mult=32, min_rows=32):
    M = a.shape[0]
    target = max(min_rows, (M + mult - 1) // mult * mult)
    return F.pad(a, (0, 0, 0, target - M)) if target != M else a


def int_mm_pipeline(a, w_q_t, w_scale, bias, sx, act=False, a_is_q=False):
    M, K = a.shape
    N = w_q_t.shape[1]
    aq = a if a_is_q else quant_act(a, sx)
    aq = pad_rows(aq)
    c = torch._int_mm(aq, w_q_t)[:M]
    y = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    _dequant[(triton.cdiv(M, 32), triton.cdiv(N, 64))](c, w_scale, bias, y, M, N, float(sx), ACT=act, BM=32, BN=64)
    return y


def fp8_pipeline(a, w8_t, sw_t, bias_bf16, sx_t, sx, act=False):
    M, K = a.shape
    a8 = quant_act(a, sx, fp8=True)
    pad = (-M) % 16
    if pad:
        a8 = F.pad(a8.view(torch.int8), (0, 0, 0, pad)).view(FP8)
    y = torch._scaled_mm(a8, w8_t, scale_a=sx_t, scale_b=sw_t, bias=bias_bf16, out_dtype=torch.bfloat16)[:M]
    return F.silu(y) if act else y


def quant_weight_int8(w):
    w = w.float()
    s = w.abs().amax(1).clamp(min=1e-12) / 127
    return torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8).contiguous(), s.contiguous()


def quant_weight_fp8(w):
    w = w.float()
    s = (w.abs().amax().clamp(min=1e-12) / FP8_MAX).reshape(())
    return (w / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).contiguous(), s


def time_graph(fn, reps=20, replays=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    ts.sort()
    return ts[len(ts) // 2]


def time_eager(fn, iters=60, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def wait_idle(thresh=5, samples=3, gap=5.0, max_wait=3600):
    import subprocess
    t0 = time.time()
    hist = []
    while time.time() - t0 < max_wait:
        u = int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.strip().split()[0])
        hist.append(u)
        if len(hist) >= samples and all(x <= thresh for x in hist[-samples:]):
            return hist[-samples:]
        time.sleep(gap)
    raise RuntimeError("GPU never idle")


def gpu_state():
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return q
