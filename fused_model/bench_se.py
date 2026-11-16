"""Micro-benchmark of the SE tail kernel configs per block shape at a given batch (graph-timed)."""
import sys, itertools
import torch, triton
import fmodel2
from fmodel2 import *

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
shapes = [(32, 8, 392), (96, 4, 98), (144, 6, 98), (144, 6, 25), (240, 10, 25), (240, 10, 7), (480, 20, 7), (480, 20, 2), (672, 28, 7), (672, 28, 2), (1152, 48, 2), (1152, 48, 7)]
for C, CS, PT in shapes:
    part = torch.randn(bs, PT, C, device="cuda", dtype=torch.float32)
    w1 = torch.randn(CS, C, device="cuda", dtype=dt) * 0.05; b1 = torch.randn(CS, device="cuda", dtype=dt)
    w2 = torch.randn(C, CS, device="cuda", dtype=dt) * 0.05; b2 = torch.randn(C, device="cuda", dtype=dt)
    g = torch.empty(bs, C, device="cuda", dtype=dt)
    BJ = max(16, triton.next_power_of_2(CS))
    best = []
    for bc, bp, w in itertools.product((32, 64, 128, 256, 512), (2, 8, 32, 128), (1, 2, 4, 8)):
        if bp > max(2, triton.next_power_of_2(PT)) and bp != 2:
            continue
        try:
            k = lambda: fmodel2.se_tail_kernel.fn[(bs,)](part, w1, b1, w2, b2, g, C, CS, PT, 1.0, BJ=BJ, BC=bc, BPT=bp, num_warps=w)
            k()
            best.append((triton.testing.do_bench_cudagraph(k, rep=20) * 1000, bc, bp, w))
        except Exception as e:
            pass
    best.sort()
    old = lambda: torch.sigmoid(F.linear(F.silu(F.linear(part.sum(1).to(dt), w1, b1)), w2, b2))
    print(C, CS, PT, "best", [(round(t, 2), bc, bp, w) for t, bc, bp, w in best[:3]], "old-chain us", round(triton.testing.do_bench_cudagraph(old, rep=20) * 1000, 2), flush=True)
