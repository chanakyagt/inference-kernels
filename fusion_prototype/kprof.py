import sys, torch, torch.nn.functional as F
from pwlib import EXP, dev, dt
from torch.profiler import profile, ProfilerActivity
torch.backends.cudnn.benchmark = True
for B in (1, 64):
    for name, ci, co, H in EXP[:2] + EXP[5:6] + EXP[11:12]:
        x = torch.randn(B, ci, H, H, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
        w = (torch.randn(co, ci, 1, 1, device=dev, dtype=dt) * ci ** -0.5).contiguous(memory_format=torch.channels_last)
        b = torch.randn(co, device=dev, dtype=dt) * .1
        for _ in range(5): F.silu(F.conv2d(x, w, b), inplace=True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(10): y = F.silu(F.conv2d(x, w, b), inplace=True)
            torch.cuda.synchronize()
        ks = {}
        for e in pr.events():
            if e.device_type.name == "CUDA":
                k = ks.setdefault(e.name[:110], [0, 0.0]); k[0] += 1; k[1] += e.device_time
        print(f"B={B} {name} {ci}->{co}@{H} out_cl={y.is_contiguous(memory_format=torch.channels_last)}")
        for k, (n, t) in ks.items(): print(f"   x{n//10} {t/n:8.1f} us  {k}")
