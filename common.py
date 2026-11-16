import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DETECTOR_REPO = Path(os.environ.get("DETECTOR_REPO", ROOT.parent / "deepfake-detector-mamba")).resolve()
SPEC_BW = 128 / 8 * 18e9


def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def peak():
    f = ROOT / "env" / "ceilings.json"
    if f.exists():
        c = json.loads(f.read_text())
        flops = c["bf16_dense_TFLOPs"] * 1e12
        return {"bw": SPEC_BW, "bw_measured": c["read_GBps"] * 1e9, "flops": flops,
                "ridge": flops / SPEC_BW, "dtype_note": "measured BF16 tensor-core matmul"}
    return {"bw": SPEC_BW, "bw_measured": None, "flops": 45e12, "ridge": 45e12 / SPEC_BW,
            "dtype_note": "assumed; run env/measure.py"}


def load_detector(dtype=torch.bfloat16, ckpt=None):
    sys.path.insert(0, str(DETECTOR_REPO))
    from dfmamba.model import Detector
    torch.manual_seed(0)
    m = Detector(pretrained=False)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = m.eval().cuda()
    for layer in m.temporal.layers:
        layer.mixer.scan_impl = "ref"
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    if dtype != torch.float32:
        m = m.to(dtype)
        for layer in m.temporal.layers:
            layer.mixer.A_log.data = layer.mixer.A_log.data.float()
            layer.mixer.D.data = layer.mixer.D.data.float()
    return m
