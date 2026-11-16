"""TensorRT INT8 baseline via NVIDIA ModelOpt PTQ (explicit Q/DQ ONNX), calibrated on Validation images only.

usage:  PYTHONPATH=$HOME/mo_pkgs:... trt_int8.py quant [--n 128] [--method entropy|max]   (CPU: ONNX Q/DQ insertion + calibration)
        trt_int8.py bench [--seconds 5]                                                    (GPU: engine build, accuracy on all Test, latency)
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/mnt/c/Users/chana/Projects/inference-kernels/trained")
ap = argparse.ArgumentParser()
ap.add_argument("stage"); ap.add_argument("--n", type=int, default=128); ap.add_argument("--method", default="entropy")
ap.add_argument("--seconds", type=float, default=5); ap.add_argument("--bs", nargs="+", type=int, default=[1, 8, 64])
ap.add_argument("--tag", default="int8")
a = ap.parse_args()

import torch
from trt_bench import ENG, VAL_DIR, calib_batches, LOG, TRTRunner, perf, predict
import tensorrt as trt
from tcommon import CACHE, interp_of, load_cache, metrics, size_of

which = os.environ["DFM_CKPT_ROOT"] + "/bb_runs/ft_effnet_b0_v2"
size = size_of(which)
HERE = Path(__file__).resolve().parent
OUTJ = HERE / f"trt_{a.tag}.json"
summary = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}


def save():
    OUTJ.write_text(json.dumps(summary, indent=1, default=str))


def qonnx(bs):
    return ENG / f"det_{a.tag}_qdq_b{bs}.onnx"


if a.stage == "quant":
    from modelopt.onnx.quantization import quantize
    batches = calib_batches(size, interp_of(which), n=a.n, bs=min(a.n, 64))
    cal = torch.cat(batches).cpu().numpy()          # Validation only (VAL_DIR/Real, VAL_DIR/Fake)
    print("calibration images", len(cal), "from", VAL_DIR, flush=True)
    for bs in a.bs:
        t0 = time.time()
        src = ENG / f"det_fp32_b{bs}.onnx"
        data = cal[: (len(cal) // bs) * bs]
        quantize(onnx_path=str(src), quantize_mode="int8", calibration_data=data,
                 calibration_method=a.method, output_path=str(qonnx(bs)), high_precision_dtype="fp16",
                 calibration_eps=["cpu"], calibration_shapes=f"x:{bs}x3x{size}x{size}")
        summary.setdefault("quant", {})[bs] = {"seconds": time.time() - t0, "method": a.method, "n_calib": int(len(data)), "onnx_mb": qonnx(bs).stat().st_size / 1e6}
        print("quantized b%d in %.0fs" % (bs, time.time() - t0), flush=True)
        save()
else:
    u8, y = load_cache(interp_of(which), None, size)
    pr = np.load(CACHE / f"probs_{Path(which).name}_R0s_{len(y)}.npy")

    def plan(bs):
        p = ENG / f"det_{a.tag}_b{bs}.plan"
        if p.exists():
            return p.read_bytes()
        b = trt.Builder(LOG)
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        pa = trt.OnnxParser(net, LOG)
        if not pa.parse(qonnx(bs).read_bytes()):
            raise RuntimeError("; ".join(str(pa.get_error(i)) for i in range(pa.num_errors)))
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        t0 = time.time()
        pl = b.build_serialized_network(net, cfg)
        assert pl is not None, "TensorRT engine build failed"
        print("built int8 b%d in %.0fs" % (bs, time.time() - t0), flush=True)
        p.write_bytes(bytes(pl))
        return bytes(pl)

    mk = lambda bs: TRTRunner(plan(bs), bs, size)
    r = mk(64)
    p = predict(r, u8, 64)
    summary["accuracy"] = {**metrics(y, p), "pred_flips_vs_R0s": int(((pr > 0.5) != (p > 0.5)).sum()), "max_abs_dprob_vs_R0s": float(np.abs(pr - p).max())}
    print(summary["accuracy"], flush=True)
    del r; torch.cuda.empty_cache()
    summary["perf"] = perf(mk, u8, a.seconds, "int8")
    print(summary["perf"]["latency_ms"], flush=True)
    save()
