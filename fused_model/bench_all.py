"""Same-session comparison: fused (own) vs R6f vs TensorRT fp16 (builder_optimization_level=5).
usage: bench_all.py [--seconds 20] [--skip-acc] [--only fused r6f trt]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
import tensorrt as trt
import trt_bench
from trt_bench import perf, predict, TRTRunner, ENG, LOG
from tcommon import CACHE, ROOT, idle_check, interp_of, load_cache, metrics, size_of
from ladder import RUNGS, Runner
from fmodel import FusedRunner
from fmodel2 import FusedRunner2

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--skip-acc", action="store_true")
ap.add_argument("--only", nargs="+", default=["fused", "r6f", "trt"])
ap.add_argument("--out", default="bench_all.json")
a = ap.parse_args()
which = os.environ["DFM_CKPT_ROOT"] + "/bb_runs/ft_effnet_b0_v2"
size = size_of(which)
u8, y = load_cache(interp_of(which), None, size)
pr = np.load(CACHE / f"probs_{Path(which).name}_R0s_{len(y)}.npy")
torch.backends.cudnn.benchmark = True
OUT = Path(__file__).resolve().parent / a.out
summary = {"idle_check": idle_check(600), "results": {}}


def save():
    OUT.write_text(json.dumps(summary, indent=1, default=str))


def trt_plan(bs):
    p = ENG / f"det_fp16_L5_b{bs}.plan"
    if p.exists():
        return p.read_bytes()
    b = trt.Builder(LOG)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    pa = trt.OnnxParser(net, LOG)
    assert pa.parse((ENG / f"det_fp16_b{bs}.onnx").read_bytes())
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    cfg.builder_optimization_level = 5
    t0 = time.time()
    plan = bytes(b.build_serialized_network(net, cfg))
    print(f"built L5 b{bs} in {time.time()-t0:.0f}s", flush=True)
    p.write_bytes(plan)
    return plan


def acc(r):
    p = predict(r, u8, 64)
    return {**metrics(y, p), "pred_flips_vs_R0s": int(((pr > 0.5) != (p > 0.5)).sum()), "max_abs_dprob_vs_R0s": float(np.abs(pr - p).max())}


makers = {
    "fused": lambda bs: FusedRunner(which, bs),
    "fused2": lambda bs: FusedRunner2(which, bs),
    "r6f": lambda bs: Runner(which, RUNGS["R6f"], bs),
    "trt": lambda bs: TRTRunner(trt_plan(bs), bs, size),
    "trt0": lambda bs: TRTRunner((ENG / f"det_fp16_b{bs}.plan").read_bytes(), bs, size),  # default optimisation level
}
for k in a.only:
    rec = summary["results"].setdefault(k, {})
    mk = makers[k]
    if not a.skip_acc:
        r = mk(64)
        rec["accuracy"] = acc(r)
        if k == "fused":
            rec["decisions_b64"] = {n: (i, t) for n, (i, t) in r.decisions.items()}
        print(k, rec["accuracy"], flush=True)
        del r; torch.cuda.empty_cache()
    rec["perf"] = perf(mk, u8, a.seconds, k)
    print(k, rec["perf"]["latency_ms"], {b: v["img_per_s"] for b, v in rec["perf"]["sustained"].items()}, flush=True)
    save()
