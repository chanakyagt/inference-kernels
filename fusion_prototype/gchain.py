import sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.argv = sys.argv[:1]
from pwlib import EXP, PRJ, pw_triton, dev, dt
import torchvision
torch.backends.cudnn.benchmark = True

def graph_time(fn, iters=200):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    for _ in range(10): g.replay()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(3):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
        for _ in range(iters): g.replay()
        b.record(); torch.cuda.synchronize(); best = min(best, a.elapsed_time(b) / iters)
    return best

