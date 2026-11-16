"""Is TensorRT's fp16 gain the number format or TensorRT's kernels?

Runs the own PyTorch path in bf16 and fp16 (with and without the Triton depthwise kernel) in one
session, with the same harness as trt_bench.py: accuracy + flips on all Test images, latency
b1/b8/b64, sustained img/s and J/img at b1 and b64.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import CACHE, ROOT, idle_check, interp_of, load_cache, metrics, size_of
from trt_bench import perf, predict
from ladder import RUNGS, Runner

OUT = ROOT / "results" / "trt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rungs", nargs="+", default=["R6b", "R6f", "R6d", "R6g"])
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    which = args.model
    u8, y = load_cache(interp_of(which), None, size_of(which))
    pr = np.load(CACHE / f"probs_{Path(which).name}_R0s_{len(y)}.npy")
    summary = {"model": which, "idle_check": idle_check(600), "results": {}}
    for key in args.rungs:
        cfg = RUNGS[key]
        rec = {"name": cfg["name"]}
        r = Runner(which, cfg, 64)
        t0 = time.time()
        p = predict(r, u8, 64)
        rec["accuracy"] = {**metrics(y, p), "eval_s": time.time() - t0,
                           "pred_flips_vs_R0s": int(((pr > 0.5) != (p > 0.5)).sum()),
                           "max_abs_dprob_vs_R0s": float(np.abs(pr - p).max())}
        del r
        torch.cuda.empty_cache()
        rec["perf"] = perf(lambda bs: Runner(which, cfg, bs), u8, args.seconds, f"fp16chk_{key}")
        summary["results"][key] = rec
        (OUT / "fp16_check.json").write_text(json.dumps(summary, indent=1, default=str))
        print(key, rec["accuracy"]["accuracy"], rec["perf"]["latency_ms"], flush=True)


if __name__ == "__main__":
    main()
