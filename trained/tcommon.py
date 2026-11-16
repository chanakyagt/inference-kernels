import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT / "quantization"))
from common import DETECTOR_REPO, bench

sys.path.insert(0, str(DETECTOR_REPO))
from dfmamba.metrics import auc as auc_fn
from dfmamba.model import FrameEncoder, temporal_from_cfg, TemporalClassifier

TEST_DIR = Path(os.environ.get("TEST_DIR", Path.home() / "data" / "MegaSub" / "Test"))
CACHE = Path(os.environ.get("IMG_CACHE", Path.home() / "img_cache"))
CKPT_ROOT = Path(os.environ.get("DFM_CKPT_ROOT", "/tmp"))
SCRATCH = Path(os.environ.get("DFM_SCRATCH", "/tmp"))
FROZEN_CKPT = CKPT_ROOT / "img_runs" / "diloco_H100" / "model.pt"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PARTS = ROOT / "results" / "trained_parts"


def list_test():
    items = []
    for label, cls in ((0, "Real"), (1, "Fake")):
        d = TEST_DIR / cls
        items += [(p, label) for p in sorted(d.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return items


def size_of(which):
    if which == "frozen":
        return 224
    return int(json.loads((Path(which) / "log.json").read_text())[0]["size"])


def interp_of(which):
    return "bilinear" if which == "frozen" else "bicubic"


def _cache_file(interp, size):
    return CACHE / (f"test_{interp}_u8.npy" if size == 224 else f"test_{interp}{size}_u8.npy")


def _load_one(args):
    from PIL import Image
    p, interp, size = args
    resample = Image.BILINEAR if interp == "bilinear" else Image.BICUBIC
    return np.asarray(Image.open(p).convert("RGB").resize((size, size), resample), dtype=np.uint8)


def build_cache(interp, size=224):
    from multiprocessing import Pool
    CACHE.mkdir(parents=True, exist_ok=True)
    f = _cache_file(interp, size)
    if f.exists():
        return
    items = list_test()
    t0 = time.time()
    with Pool(4) as pool:
        arrs = pool.map(_load_one, [(str(p), interp, size) for p, _ in items], chunksize=64)
    np.save(f, np.stack(arrs))
    np.save(CACHE / "labels.npy", np.array([l for _, l in items], dtype=np.int64))
    (CACHE / "names.json").write_text(json.dumps([str(p) for p, _ in items]))
    print(f"cache {interp}: {len(items)} images in {time.time() - t0:.0f}s", flush=True)


def load_cache(interp, limit=None, size=224):
    build_cache(interp, size)
    x = np.load(_cache_file(interp, size), mmap_mode="r")
    y = np.load(CACHE / "labels.npy")
    if limit and limit < len(y):
        idx = np.sort(np.random.default_rng(0).choice(len(y), limit, replace=False))
        return np.ascontiguousarray(x[idx]), y[idx]
    return x, y


def prep(u8):
    t = torch.from_numpy(np.ascontiguousarray(u8)).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    return ((t - MEAN.cuda()) / STD.cuda()).contiguous()


class Net(nn.Module):
    def __init__(self, trunk, head, kind):
        super().__init__()
        self.trunk, self.head, self.kind = trunk, head, kind

    def forward(self, x):
        f = self.trunk(x)
        if self.kind == "mamba":
            f = f.unsqueeze(1)
        return self.head(f)


def make_trunk():
    enc = FrameEncoder(pretrained=True)
    return nn.Sequential(enc.features, enc.pool, nn.Flatten(1))


def find_finetuned():
    base = CKPT_ROOT / "bb_runs"
    return sorted(p for p in base.glob("ft_*effnet_b0*") if (p / "model.pt").exists()) if base.exists() else []


def load_model(which):
    if which == "frozen":
        ck = torch.load(FROZEN_CKPT, map_location="cpu")
        head = temporal_from_cfg(ck["config"])
        head.load_state_dict(ck["model"])
        net = Net(make_trunk(), head, "mamba")
        interp = "bilinear"
    else:
        ck = torch.load(Path(which) / "model.pt", map_location="cpu")
        sd = ck["model"]
        trunk = make_trunk()
        trunk.load_state_dict({k[len("trunk.body."):]: v for k, v in sd.items() if k.startswith("trunk.body.")})
        if "head.proj.weight" in sd:
            d = sd["head.proj.weight"].shape[0]
            n_layers = len({k.split(".")[2] for k in sd if k.startswith("head.layers.")})
            head = TemporalClassifier(d_model=d, n_layers=n_layers, d_state=16, feat_dim=sd["head.proj.weight"].shape[1])
            kind = "mamba"
        else:
            head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
            kind = "linear"
        head.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
        net = Net(trunk, head, kind)
        interp = "bicubic"
    net = net.eval().cuda()
    for p in net.parameters():
        p.requires_grad_(False)
    if net.kind == "mamba":
        for layer in net.head.layers:
            layer.mixer.scan_impl = "ref"
    return net, interp


def fold_conv_bn(model):
    n = 0
    for mod in list(model.modules()):
        if isinstance(mod, nn.Sequential) and len(mod) >= 2 and isinstance(mod[0], nn.Conv2d) and isinstance(mod[1], nn.BatchNorm2d):
            conv, bn = mod[0], mod[1]
            scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
            w = conv.weight.float() * scale[:, None, None, None]
            b = bn.bias.float() - bn.running_mean.float() * scale
            if conv.bias is not None:
                b = b + conv.bias.float() * scale
            new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                            conv.dilation, conv.groups, bias=True).to(conv.weight.device)
            new.weight.data.copy_(w)
            new.bias.data.copy_(b)
            new.requires_grad_(False)
            mod[0] = new
            mod[1] = nn.Identity()
            n += 1
    return n


def smi(fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return out


def gpu_state():
    v = smi("utilization.gpu,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,memory.used")
    k = ["util_pct", "sm_clock_mhz", "mem_clock_mhz", "max_sm_clock_mhz", "temp_c", "power_w", "mem_used_mib"]
    out = {}
    for a, b in zip(k, v):
        try:
            out[a] = float(b)
        except ValueError:
            out[a] = b
    return out


def idle_check(timeout_s=0, gap=5, n=3, thr=5):
    t0 = time.time()
    while True:
        reads = []
        for i in range(n):
            reads.append(float(smi("utilization.gpu")[0]))
            if i < n - 1:
                time.sleep(gap)
        ok = all(r <= thr for r in reads)
        if ok or time.time() - t0 >= timeout_s:
            return {"samples_pct": reads, "passed": ok, "waited_s": round(time.time() - t0, 1)}
        time.sleep(gap)


class PowerSampler:
    def __init__(self, path):
        self.path = path

    def start(self):
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
                                  stdout=self.f)
        time.sleep(0.5)

    def stop(self):
        self.p.terminate()
        self.p.wait()
        self.f.close()
        out = []
        for line in open(self.path):
            try:
                ts, w = line.strip().split(", ")
                out.append((datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(w)))
            except ValueError:
                pass
        return out


def mean_between(samples, t0, t1):
    v = [w for t, w in samples if t0 <= t <= t1]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)


def idle_power(seconds=5, attempts=6, mean_max=20.0, peak_max=30.0):
    best = None
    for k in range(attempts):
        torch.cuda.synchronize()
        time.sleep(2)
        s = PowerSampler(str(SCRATCH / "idle_power.csv"))
        s.start()
        t0 = time.time()
        time.sleep(seconds)
        t1 = time.time()
        samp = s.stop()
        mean, n = mean_between(samp, t0 + 1, t1)
        peak = max([w for t, w in samp if t0 + 1 <= t <= t1] or [float("nan")])
        rec = {"mean_w": mean, "peak_w": peak, "samples": n, "attempt": k + 1, "clean": mean < mean_max and peak < peak_max}
        if best is None or rec["mean_w"] < best["mean_w"]:
            best = rec
        if rec["clean"]:
            return rec
        time.sleep(10)
    return best


def window_stats(marks, bs, start_idx):
    wins = []
    i0 = start_idx
    for i in range(start_idx + 1, len(marks)):
        if marks[i][0] - marks[i0][0] >= 1.0:
            wins.append((marks[i][1] - marks[i0][1]) * bs / (marks[i][0] - marks[i0][0]))
            i0 = i
    if len(wins) < 3:
        return {"windows": wins, "stable": False, "cv": float("nan"), "min_over_median": float("nan")}
    med = sorted(wins)[len(wins) // 2]
    mean = sum(wins) / len(wins)
    sd = (sum((w - mean) ** 2 for w in wins) / len(wins)) ** 0.5
    return {"windows": wins, "stable": min(wins) >= 0.85 * med and sd / mean <= 0.08, "cv": sd / mean,
            "min_over_median": min(wins) / med}


def sustained(run, x, seconds, tag="x"):
    bs = x.shape[0]
    idle = idle_power()
    run(x)
    torch.cuda.synchronize()
    s = PowerSampler(str(SCRATCH / f"power_{tag}.csv"))
    s.start()
    t0 = time.time()
    n = 0
    marks = []
    mark_idx = None
    while True:
        for _ in range(16):
            run(x)
            n += 1
        torch.cuda.synchronize()
        now = time.time()
        marks.append((now, n))
        if mark_idx is None and now - t0 >= 3:
            mark_idx = len(marks) - 1
        if now - t0 >= seconds:
            break
    t1 = time.time()
    samp = s.stop()
    m0 = marks[mark_idx]
    ips = (n - m0[1]) * bs / (t1 - m0[0])
    p, pn = mean_between(samp, m0[0], t1)
    st = window_stats(marks, bs, mark_idx)
    return {"batch": bs, "img_per_s": ips, "mean_power_w": p, "power_samples": pn, "idle_power_w": idle["mean_w"],
            "idle_peak_w": idle["peak_w"], "idle_clean": idle["clean"], "idle_attempts": idle["attempt"],
            "idle_samples": idle["samples"], "J_per_img_raw": p / ips, "J_per_img_idle_subtracted": (p - idle["mean_w"]) / ips,
            "window_s": t1 - m0[0], "stable": st["stable"], "throughput_cv": st["cv"], "min_over_median": st["min_over_median"],
            "gpu_state_end": gpu_state()}


def metrics(y, p, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(p) > thr).astype(int)
    return {"accuracy": float((pred == y).mean()), "auc": auc_fn(y, p),
            "tp": int(((pred == 1) & (y == 1)).sum()), "tn": int(((pred == 0) & (y == 0)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()), "fn": int(((pred == 0) & (y == 1)).sum()), "n": int(len(y))}
