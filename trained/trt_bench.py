"""TensorRT baseline for the trained detector, measured with the ladder's own harness.

Builds TensorRT engines (fp16, int8 if the installed TensorRT still supports implicit
calibration) from the plain fp32 model, then reports, in the same units as
results/trained_ladder.md: accuracy / AUC / flips vs R0s on all Test images, latency
(median of CUDA-event timings, best of 3 trials, input copy included) at batch 1/8/64,
and sustained img/s + J/img at batch 1 and 64. The PyTorch best rung (R6b) is re-run in
the same session so both sides see the same GPU conditions.

Usage (WSL):  DFM_CKPT_ROOT=/mnt/c/Users/chana/Projects/_artifacts \
              ~/trt-env/bin/python trt_bench.py --modes fp16 int8 --seconds 20
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import (CACHE, ROOT, bench, fold_conv_bn, gpu_state, idle_check, interp_of, load_cache, load_model, metrics, prep, size_of,
                     sustained, _load_one)

OUT = ROOT / "results" / "trt"
ENG = Path(os.environ.get("TRT_CACHE", Path.home() / "trt_cache"))
VAL_DIR = Path(os.environ.get("VAL_DIR", Path.home() / "data" / "MegaSub" / "Validation"))
LOG = trt.Logger(trt.Logger.WARNING)
TORCH_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
if hasattr(trt, "bfloat16"):
    TORCH_DT[trt.bfloat16] = torch.bfloat16


MODE_DT = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def export_onnx(net, bs, size, path, dtype=torch.float32):
    """TensorRT 11 is strongly typed: the engine runs in the dtypes stored in the ONNX graph."""
    if path.exists():
        return
    m = net.to(dtype)
    x = torch.randn(bs, 3, size, size, device="cuda", dtype=dtype)
    with torch.no_grad():
        torch.onnx.export(m, (x,), str(path), input_names=["x"], output_names=["logits"], opset_version=20, dynamo=True,
                          external_data=False)
    net.float()
    import onnx
    g = onnx.load(str(path)).graph
    want = {torch.float32: onnx.TensorProto.FLOAT, torch.float16: onnx.TensorProto.FLOAT16,
            torch.bfloat16: onnx.TensorProto.BFLOAT16}[dtype]
    bad = [i.name for i in g.initializer if i.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16,
                                                           onnx.TensorProto.BFLOAT16) and i.data_type != want]
    if bad:
        path.unlink()
        raise RuntimeError(f"ONNX export left {len(bad)} initializers in the wrong dtype, e.g. {bad[:3]}")


class Calib(getattr(trt, "IInt8EntropyCalibrator2", object)):
    """Entropy calibration on Validation images (never Test)."""

    def __init__(self, batches, cache_file):
        if hasattr(trt, "IInt8EntropyCalibrator2"):
            trt.IInt8EntropyCalibrator2.__init__(self)
        self.batches, self.i, self.cache_file = batches, 0, cache_file
        self.buf = torch.empty_like(batches[0])

    def get_batch_size(self):
        return self.batches[0].shape[0]

    def get_batch(self, names):
        if self.i >= len(self.batches):
            return None
        self.buf.copy_(self.batches[self.i])
        self.i += 1
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return self.cache_file.read_bytes() if self.cache_file.exists() else None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(bytes(cache))


def calib_batches(size, interp, n=512, bs=64):
    items = []
    for cls in ("Real", "Fake"):
        d = VAL_DIR / cls
        items += sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    random.Random(0).shuffle(items)
    u8 = np.stack([_load_one((p, interp, size)) for p in items[:n]])
    return [prep(u8[i:i + bs]).contiguous() for i in range(0, n, bs)]


def build_engine(onnx_path, mode, plan_path, calib=None):
    if plan_path.exists():
        return plan_path.read_bytes(), {"cached": True}
    info = {"cached": False}
    builder = trt.Builder(LOG)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, LOG)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if mode == "fp32":
        cfg.set_flag(trt.BuilderFlag.TF32)
    if mode == "int8":
        raise RuntimeError("TensorRT 11 has no implicit INT8; needs explicit Q/DQ (NVIDIA ModelOpt), not run")
    t0 = time.time()
    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError("engine build returned None")
    info["build_s"] = time.time() - t0
    plan = bytes(plan)
    plan_path.write_bytes(plan)
    return plan, info


class TRTRunner:
    def __init__(self, plan, bs, size, graph=True):
        self.engine = trt.Runtime(LOG).deserialize_cuda_engine(plan)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outn = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        self.in_dt = TORCH_DT[self.engine.get_tensor_dtype(self.inp)]
        out_dt = TORCH_DT[self.engine.get_tensor_dtype(self.outn)]
        self.static_in = torch.zeros(bs, 3, size, size, device="cuda", dtype=self.in_dt)
        self.static_out = torch.zeros(tuple(self.engine.get_tensor_shape(self.outn)), device="cuda", dtype=out_dt)
        self.ctx.set_tensor_address(self.inp, self.static_in.data_ptr())
        self.ctx.set_tensor_address(self.outn, self.static_out.data_ptr())
        self.stream = torch.cuda.Stream()
        self.graph = None
        for _ in range(3):
            self._enqueue()
        torch.cuda.synchronize()
        if graph:
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream):
                    self._enqueue()
                torch.cuda.synchronize()
                self.graph = g
            except Exception as e:  # TensorRT may refuse capture for some engines
                self.graph_error = f"{type(e).__name__}: {e}"

    def _enqueue(self):
        cur = torch.cuda.current_stream()
        self.ctx.execute_async_v3(cur.cuda_stream)

    @torch.no_grad()
    def run(self, x):
        self.static_in.copy_(x)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._enqueue()
        return self.static_out.float()


def predict(r, u8, ab):
    out = np.empty(len(u8), dtype=np.float64)
    for i in range(0, len(u8), ab):
        x = prep(u8[i:i + ab])
        m = x.shape[0]
        if m < ab:
            x = torch.cat([x, x[-1:].expand(ab - m, -1, -1, -1)])
        out[i:i + m] = r.run(x).softmax(1)[:m, 1].double().cpu().numpy()
    return out


def perf(make_runner, u8, seconds, tag):
    res = {"latency_ms": {}, "latency_trials_ms": {}, "sustained": {}, "gpu_state_start": gpu_state()}
    for bs in (1, 8, 64):
        r = make_runner(bs)
        res.setdefault("graph", {})[bs] = getattr(r, "graph", None) is not None
        x = prep(u8[:bs])
        iters = {1: 200, 8: 100, 64: 50}[bs]
        trials = [bench(lambda: r.run(x), iters=iters, warmup=10) for _ in range(3)]
        res["latency_ms"][bs], res["latency_trials_ms"][bs] = min(trials), trials
        if bs in (1, 64):
            best, attempts = None, []
            for _ in range(3):
                c = sustained(lambda z: r.run(z), x, seconds, tag=f"{tag}_{bs}")
                attempts.append({k: c[k] for k in ("img_per_s", "stable", "idle_clean", "mean_power_w", "idle_power_w")})
                ok = c["stable"] and c["idle_clean"]
                if best is None or ok or (c["img_per_s"] > best["img_per_s"] and not (best["stable"] and best["idle_clean"])):
                    best = c
                if ok:
                    break
            best["attempts"] = attempts
            res["sustained"][bs] = best
        del r
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(os.environ.get("DFM_CKPT_ROOT", "/tmp")) / "bb_runs" / "ft_effnet_b0_v2"))
    ap.add_argument("--modes", nargs="+", default=["bf16", "fp16"])
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--acc-bs", type=int, default=64)
    ap.add_argument("--no-r6b", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=600)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ENG.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    which = args.model
    size, interp = size_of(which), interp_of(which)
    u8, y = load_cache(interp, None, size)
    ref = CACHE / f"probs_{Path(which).name}_R0s_{len(y)}.npy"
    pr = np.load(ref) if ref.exists() else None
    net, _ = load_model(which)
    folded = fold_conv_bn(net.trunk)  # BN into conv in fp32 first, so the low-precision export keeps one dtype
    summary ={"trt_version": trt.__version__, "torch": torch.__version__, "model": which, "size": size,
               "bn_folded_into_conv": folded, "idle_check": idle_check(args.idle_timeout), "results": {}}

    def save():
        (OUT / "trt_results.json").write_text(json.dumps(summary, indent=1, default=str))

    for mode in args.modes:
        rec = {}
        summary["results"][mode] = rec
        try:
            plans = {}
            calib = None
            for bs in sorted({1, 8, 64, args.acc_bs}):
                onnx_p = ENG / f"det_{mode}_b{bs}.onnx"
                export_onnx(net, bs, size, onnx_p, MODE_DT.get(mode, torch.float32))
                plans[bs], rec[f"build_b{bs}"] = build_engine(onnx_p, mode, ENG / f"det_{mode}_b{bs}.plan", calib)
            r = TRTRunner(plans[args.acc_bs], args.acc_bs, size)
            t0 = time.time()
            p = predict(r, u8, args.acc_bs)
            rec["accuracy"] = {**metrics(y, p), "eval_s": time.time() - t0}
            if pr is not None:
                rec["accuracy"]["pred_flips_vs_R0s"] = int(((pr > 0.5) != (p > 0.5)).sum())
                rec["accuracy"]["max_abs_dprob_vs_R0s"] = float(np.abs(pr - p).max())
            del r
            torch.cuda.empty_cache()
            save()
            rec["perf"] = perf(lambda bs: TRTRunner(plans[bs], bs, size), u8, args.seconds, f"trt_{mode}")
        except Exception as e:
            rec["failed"] = f"{type(e).__name__}: {str(e)[:600]}"
            rec["traceback"] = traceback.format_exc()[-2000:]
        save()
        print(mode, json.dumps({k: v for k, v in rec.items() if k != "traceback"}, default=str)[:1500], flush=True)

    if not args.no_r6b:
        from ladder import RUNGS, Runner
        cfg = RUNGS["R6b"]
        try:
            summary["results"]["R6b_same_session"] = {"perf": perf(lambda bs: Runner(which, cfg, bs), u8, args.seconds, "r6b")}
        except Exception as e:
            summary["results"]["R6b_same_session"] = {"failed": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
        save()
    print("done", OUT / "trt_results.json", flush=True)


if __name__ == "__main__":
    main()
