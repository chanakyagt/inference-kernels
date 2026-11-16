"""Profile fused model at b1: kernel list in the eager forward and inside the CUDA graph."""
import json, os, sys, torch
from collections import Counter
from torch.profiler import profile, ProfilerActivity
import sys
from fmodel import FusedRunner
from fmodel2 import FusedRunner2
which = os.environ["DFM_CKPT_ROOT"] + "/bb_runs/ft_effnet_b0_v2"
r = (FusedRunner2 if len(sys.argv) > 1 else FusedRunner)(which, 1)
print({k: v for k, v in getattr(r, "decisions2", {}).items() if v[0] != "new"})
x = r.static_in
def kern(prof):
    return [(e.name, e.device_time_total if hasattr(e,'device_time_total') else e.cuda_time_total, e.count) for e in prof.key_averages() if e.device_type.name=="CUDA"]
with torch.no_grad():
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
        for _ in range(5): r.run(x)
        torch.cuda.synchronize()
    ks=[e for e in p.events() if e.device_type.name=="CUDA"]
    print("graph kernels per run:", len(ks)/5)
    tot=0; rows=Counter(); cnt=Counter()
    for e in ks:
        n=e.name[:70]; rows[n]+=e.device_time/5; cnt[n]+=1/5; tot+=e.device_time/5
    print("sum kernel us per run", tot)
    for n,t in rows.most_common(25): print("%8.1f us  x%-4g %s"%(t,cnt[n],n))
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p2:
        r.fwd(x); torch.cuda.synchronize()
    ks2=[e for e in p2.events() if e.device_type.name=="CUDA"]
    print("eager kernels:", len(ks2))
    # gaps: span vs sum
    ks.sort(key=lambda e:e.time_range.start)
    one = ks[:len(ks)//5]
    span = one[-1].time_range.end - one[0].time_range.start
    print("one graph run span us", span, "sum", sum(e.device_time for e in one))
