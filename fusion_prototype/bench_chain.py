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

res = {}
for B in (1, 64):
    E = []
    for name, ci, co, H in EXP:
        x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        w = (torch.randn(co, ci, 1, 1, device=dev, dtype=dt) * ci ** -0.5).contiguous(memory_format=torch.channels_last)
        b = torch.randn(co, device=dev, dtype=dt) * .1
        E.append((x, w, b, x.permute(0, 2, 3, 1).reshape(-1, ci), w.view(co, ci).t().contiguous()))
    P = []
    for name, ci, co, H, r in PRJ:
        x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        s = torch.rand(B, ci, 1, 1, device=dev, dtype=dt)
        w = (torch.randn(co, ci, 1, 1, device=dev, dtype=dt) * ci ** -0.5).contiguous(memory_format=torch.channels_last)
        b = torch.randn(co, device=dev, dtype=dt) * .1
        rt = torch.randn(B, co, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last) if r else None
        P.append((x, s, w, b, rt, x.permute(0, 2, 3, 1).reshape(-1, ci), s.view(B, ci).contiguous(), w.view(co, ci).t().contiguous(),
                  rt.permute(0, 2, 3, 1).reshape(-1, co) if r else None, H * H))
    def e_torch():
        for x, w, b, a, wk in E: F.silu(F.conv2d(x, w, b), inplace=True)
    def e_tr():
        for x, w, b, a, wk in E: pw_triton(a, wk, b, True)
    def p_torch():
        for x, s, w, b, rt, *_ in P:
            y = F.conv2d(x * s, w, b)
            if rt is not None: y = y + rt
    def p_tr():
        for x, s, w, b, rt, a, s2, wk, r2, hw in P: pw_triton(a, wk, b, False, scale=s2, res=r2, hw=hw)
    e_tr(); p_tr(); torch.cuda.synchronize()  # autotune outside capture
    res[B] = dict(expand_torch=graph_time(e_torch), expand_triton=graph_time(e_tr),
                  project_torch=graph_time(p_torch), project_triton=graph_time(p_tr))
    print(B, {k: round(v, 4) for k, v in res[B].items()}, flush=True)

# kernel count of a stock (BN-folded, cuDNN depthwise) EfficientNet-B0 trunk at b1, graph-timed
m = torchvision.models.efficientnet_b0(weights=None).features.eval()
for mod in m.modules():
    for n, c in list(mod.named_children()):
        if isinstance(c, torchvision.ops.misc.Conv2dNormActivation) and isinstance(c[1], nn.BatchNorm2d):
            conv, bn = c[0], c[1]
            sc = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            conv.weight.data *= sc[:, None, None, None]
            conv.bias = nn.Parameter(bn.bias - bn.running_mean * sc)
            c[1] = nn.Identity()
m = m.cuda().half().to(memory_format=torch.channels_last)
xin = torch.randn(1, 3, 256, 256, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
with torch.no_grad():
    res["stock_trunk_b1_graph_ms"] = graph_time(lambda: m(xin))
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        m(xin); torch.cuda.synchronize()
    names = [e.name for e in pr.events() if e.device_type.name == "CUDA"]
    res["stock_trunk_b1_kernels"] = len(names)
print("stock trunk b1 graph ms", res["stock_trunk_b1_graph_ms"], "kernels", res["stock_trunk_b1_kernels"])
json.dump(res, open("chain.json", "w"), indent=1)
