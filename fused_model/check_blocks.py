"""Per-block max abs diff: fused block vs unfused R6f block, on real Test activations."""
import sys, os, torch
import fmodel
from fmodel import *
from tcommon import interp_of, load_cache, prep

which = os.environ['DFM_CKPT_ROOT'] + '/bb_runs/ft_effnet_b0_v2'
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
u8, y = load_cache(interp_of(which), None, size_of(which))
ref, fn = build(which)
x = prep(u8[:bs]).contiguous(memory_format=torch.channels_last).to(dt)
mbs = [mb for si in range(1, 8) for mb in ref.trunk[0][si]]
rows = []
with torch.no_grad():
    h = ref.trunk[0][0](x)
    hn = h.permute(0, 2, 3, 1).contiguous()
    for mb, fb in zip(mbs, fn.blocks):
        o_ref = mb(h)
        o_f = fb(hn)
        d = (o_ref.permute(0, 2, 3, 1).float() - o_f.float()).abs()
        rows.append((fb.project.name, float(d.max()), float(o_ref.float().abs().max()), fb.expand.impl if fb.expand else None, fb.project.impl))
        h = o_ref; hn = o_ref.permute(0, 2, 3, 1).contiguous()
for r in rows:
    print("%-12s maxdiff %.4g  (ref max |act| %.3g)  expand=%s project=%s" % r)
