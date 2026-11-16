"""v2 pieces vs unfused reference on real Test activations: stem, each block (output + SE gate), head logits."""
import sys, os
os.environ["FM2_FORCE"] = "new"
import torch
from fmodel2 import *
from tcommon import interp_of, load_cache, prep

which = os.environ['DFM_CKPT_ROOT'] + '/bb_runs/ft_effnet_b0_v2'
bs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
u8, y = load_cache(interp_of(which), None, size_of(which))
ref, _ = build(which)
fn = FusedNet2(ref)
x = prep(u8[:bs])
mbs = [mb for si in range(1, 8) for mb in ref.trunk[0][si]]
with torch.no_grad():
    xr = x.contiguous(memory_format=torch.channels_last).to(dt)
    h = ref.trunk[0][0](xr); hn = fn.stem_new(x)
    print("stem maxdiff %.4g (ref max %.3g)" % ((h.permute(0, 2, 3, 1).float() - hn.float()).abs().max(), h.abs().max()))
    hn = h.permute(0, 2, 3, 1).contiguous()
    for mb, fb in zip(mbs, fn.blocks):
        o_ref = mb(h); o_f = fb(hn)
        hh = fb.expand(hn) if fb.expand is not None else hn
        y2, part, hw = dw_pool(hh, fb.dw_w, fb.dw_b, fb.stride, fb.pad)
        g_new = se_tail(part, hw, fb.w1, fb.b1, fb.w2, fb.b2).float()
        g_old = fb.se_old(dw_nhwc(hh, fb.dw_w, fb.dw_b, fb.stride, fb.pad)).float()
        d = (o_ref.permute(0, 2, 3, 1).float() - o_f.float()).abs()
        print("%-6s block maxdiff %.4g (ref max %.3g)  gate maxdiff %.3g" % (fb.name, d.max(), o_ref.float().abs().max(), (g_new - g_old).abs().max()))
        h = o_ref; hn = o_ref.permute(0, 2, 3, 1).contiguous()
    z = fn.headconv(hn)
    print("head logits maxdiff", (fn.head_new(z) - fn.head_old(z)).abs().max().item())
    full = fn(x); refo = ref(xr).float()
    print("whole-model logits maxdiff", (full - refo).abs().max().item(), "ref max", refo.abs().max().item())
