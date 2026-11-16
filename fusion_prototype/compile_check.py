import sys, json, time
sys.path.insert(0, "/tmp/gpu_b")
sys.argv = sys.argv[:1]
import torch, torch.nn.functional as F
from pwlib import EXP, pw_triton, dev, dt
from gchain import graph_time
torch.backends.cudnn.benchmark = True
mode = sys.argv[1] if len(sys.argv) > 1 else "default"
B = 64
E = []
for name, ci, co, H in EXP:
    x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
    w = (torch.randn(co, ci, 1, 1, device=dev, dtype=dt) * ci ** -0.5).contiguous(memory_format=torch.channels_last)
    b = torch.randn(co, device=dev, dtype=dt) * .1
    E.append((x, w, b, x.permute(0, 2, 3, 1).reshape(-1, ci), w.view(co, ci).t().contiguous()))
def eager():
    return [F.silu(F.conv2d(x, w, b), inplace=True) for x, w, b, a, wk in E]
def trit():
    return [pw_triton(a, wk, b, True) for x, w, b, a, wk in E]
res = {}
# correctness of triton vs eager across whole chain
with torch.no_grad():
    ye, yt = eager(), trit()
    res["triton_maxerr"] = max((t.float() - e.permute(0,2,3,1).reshape(-1, e.shape[1]).float()).abs().max().item() for t, e in zip(yt, ye))
    res["eager_ms"] = graph_time(eager); res["triton_ms"] = graph_time(trit)
    for md in ["default", "max-autotune-no-cudagraphs"]:
        t0 = time.time()
        f = torch.compile(eager, mode=md)
        f(); f(); torch.cuda.synchronize()
        res["compile_" + md + "_ms"] = graph_time(f)
        res["compile_" + md + "_s"] = round(time.time() - t0, 1)
        print(md, res, flush=True)
print(json.dumps(res, indent=1))
