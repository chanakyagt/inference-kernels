import torch, torch.nn.functional as F, json
from pwlib import EXP, PRJ, dev, dt
from gchain import graph_time
torch.backends.cudnn.benchmark = True
out = {}
for B in (1, 64):
    L = []
    for name, ci, co, H in EXP:
        x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        w = torch.randn(co, ci, 1, 1, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        L.append((x, w))
    Pp = []
    for name, ci, co, H, r in PRJ:
        x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        w = torch.randn(co, ci, 1, 1, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        Pp.append((x, w))
    def f():
        for x, w in L: F.conv2d(x, w)
    def g():
        for x, w in Pp: F.conv2d(x, w)
    out[B] = dict(expand_gemm_only=graph_time(f), project_gemm_only=graph_time(g))
    print(B, out[B], flush=True)
json.dump(out, open("nobias.json", "w"))
