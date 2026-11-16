"""Per-new-frame streaming latency, one quiet session, RTX 4060 Ti. All methods fp16, CUDA graph per frame,
Mamba state (or the feature window) carried across frames. Latency = wall clock per frame incl. a device sync,
input already preprocessed on the GPU; best of 3 trials of a 200-frame median. Writes bench.json."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import (HERE, F16, FusedStream, TrunkFusedTorchHead, UnfusedStream, WindowHead, load_head,
                      load_trunk_net)
from tcommon import gpu_state, idle_check
from bench_stream import count_launches, graph_inner_time, timeit  # stream/bench_stream.py (read-only import)

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True


def trials(fn, n=3, iters=200):
    rs = [timeit(fn, iters) for _ in range(n)]
    meds = [r["median_ms"] for r in rs]
    best = min(range(n), key=lambda i: meds[i])
    return {"best_median_ms": meds[best], "trial_medians_ms": meds, "p10_ms": rs[best]["p10_ms"], "p90_ms": rs[best]["p90_ms"],
            "trial_spread": max(meds) / min(meds), "stable": max(meds) / min(meds) <= 1.10}


def run_method(label, build, x):
    obj = build()
    obj.reset()
    rec = {"label": label, "state_bytes_per_stream": obj.state_bytes() / obj.B}
    with torch.no_grad():
        obj.step(x)
    rec["launches_eager"] = count_launches(lambda: obj.step(x))
    obj.reset()
    obj.capture()
    rec["graph"] = trials(lambda: obj.step_graph(x))
    if hasattr(obj, "fn"):
        rec["switch_decisions"] = {b.name: b.sw.impl for b in obj.fn.blocks}
        rec["switch_decisions"]["stem"] = obj.fn.sw_stem.impl
    del obj
    gc.collect()
    torch.cuda.empty_cache()
    return rec


@torch.no_grad()
def stages(net, head, B, x):
    fs = FusedStream(net, head, B)
    tb = TrunkFusedTorchHead(net, head, B, fn=fs.fn)
    un = UnfusedStream(net, head, B, F16)
    fs.step(x); tb.step(x); un.step(x)
    torch.cuda.synchronize()
    hmap = fs.trunk_map(x)
    hview = hmap.permute(0, 3, 1, 2)
    feat = tb.fn.headconv(hmap).mean((1, 2))
    xproj = tb.mamba.t.proj(feat)
    r = {
        "unfused_trunk_incl_pool_ms": graph_inner_time(lambda: un.trunk_feat(x)),
        "fused_trunk_stem_blocks_ms": graph_inner_time(lambda: fs.trunk_map(x)),
        "torch_after_trunk_ms (fused headconv + mean + proj + PyTorch Mamba step)": graph_inner_time(lambda: tb.mamba(tb.fn.headconv(hmap).mean((1, 2)))),
        "torch_mamba_step_from_proj_ms": graph_inner_time(lambda: tb.mamba.from_proj(xproj)),
        "kernel_A_ms (headconv+SiLU+pool+proj)": graph_inner_time(lambda: fs.tail(hview, out=fs.stack.xbuf[0])),
        "kernel_B_ms (4 Mamba layers + norm_f + mean + MLP)": graph_inner_time(fs.stack.step_from_buf0),
        "fused_after_trunk_A_plus_B_ms": graph_inner_time(lambda: (fs.tail(hview, out=fs.stack.xbuf[0]), fs.stack.step_from_buf0())),
    }
    r["launches_after_trunk"] = {
        "fused_A_plus_B": count_launches(lambda: (fs.tail(hview, out=fs.stack.xbuf[0]), fs.stack.step_from_buf0()))["gpu_kernels"],
        "torch_headconv_mean_proj_step": count_launches(lambda: tb.mamba(tb.fn.headconv(hmap).mean((1, 2))))["gpu_kernels"],
    }
    del fs, tb, un
    gc.collect()
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-timeout", type=float, default=1800)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    a = ap.parse_args()
    res = {"torch": torch.__version__, "gpu_state_start": gpu_state()}
    res["idle_check"] = idle_check(a.idle_timeout)
    print("idle", res["idle_check"], flush=True)
    if not res["idle_check"]["passed"]:
        res["aborted"] = "GPU not idle; no timing taken"
        (HERE / "bench.json").write_text(json.dumps(res, indent=1))
        print(res["aborted"])
        return
    net, head = load_trunk_net(), load_head()
    res["runs"] = {}
    for B in a.batches:
        x = torch.randn(B, 3, 256, 256, device="cuda")
        methods = {
            "a_unfused_pytorch_trunk_plus_pytorch_mamba_step": lambda: UnfusedStream(net, head, B, F16),
            "b_fused_v2_trunk_plus_pytorch_mamba_step": lambda: TrunkFusedTorchHead(net, head, B),
            "new_fused_v2_trunk_plus_kernels_A_B": lambda: FusedStream(net, head, B),
            "c1_window30_fused_scan_mamba_recompute": lambda: WindowHead(net, head, B, 30, False),
            "c2_window30_attention_RANDOM_weights": lambda: WindowHead(net, head, B, 30, True),
        }
        r = {"methods": {}}
        for label, build in methods.items():
            m = run_method(label, build, x)
            r["methods"][label] = m
            print(B, label, round(m["graph"]["best_median_ms"], 4), m["graph"]["trial_medians_ms"],
                  "kernels", m["launches_eager"]["gpu_kernels"], flush=True)
        r["stages_graph_ms"] = stages(net, head, B, x)
        print(B, "stages", r["stages_graph_ms"], flush=True)
        r["gpu_state_after"] = gpu_state()
        res["runs"][f"B{B}"] = r
    res["gpu_state_end"] = gpu_state()
    res["idle_check_end"] = idle_check(0)
    (HERE / "bench.json").write_text(json.dumps(res, indent=1, default=str))
    print("written")


if __name__ == "__main__":
    main()
