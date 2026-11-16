import argparse
import collections
import copy
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stream"))
sys.path.insert(0, str(ROOT / "kernels"))
import selective_scan as ss_mod
from fused_dwconv_bn_silu import FusedDWConvBNSiLU
from fused_mamba_step import FusedMambaStack
from fused_tail_proj import FusedTailProj
from pipeline import StreamPipeline, TorchStreamBaseline, WindowBaseline, load_parts, make_original, to_dtype

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
RES = ROOT / "results"
DT = {"bf16": torch.bfloat16, "fp32": torch.float32}


def smi(fields):
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [x.strip() for x in out.strip().split(",")]


def wait_idle(max_wait=3600):
    t0 = time.time()
    ok = 0
    samples = []
    while time.time() - t0 < max_wait:
        u = float(smi("utilization.gpu")[0])
        samples.append(u)
        ok = ok + 1 if u <= 5 else 0
        if ok >= 3:
            return samples[-3:]
        time.sleep(5)
    raise SystemExit("GPU never idle")


def env_record():
    f = smi("name,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,utilization.gpu")
    return dict(zip(["gpu", "sm_clock_MHz", "mem_clock_MHz", "max_sm_clock_MHz", "temp_C", "power_W", "util_pct"], f))


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    n = len(ts)
    return {"median_ms": ts[n // 2], "p10_ms": ts[n // 10], "p90_ms": ts[(9 * n) // 10], "iters": n}


def graph_inner_time(fn, reps=20, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters + 5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    ts = sorted(ts[5:])
    return ts[len(ts) // 2]


def count_launches(fn, graph=False):
    from torch.profiler import ProfilerActivity, profile
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    kern, mem, cpu_launch, graph_launch = 0, 0, 0, 0
    names = collections.Counter()
    for e in p.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            if e.name.startswith("Memcpy") or e.name.startswith("Memset"):
                mem += 1
            else:
                kern += 1
                names[e.name[:60]] += 1
        elif e.name in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx"):
            cpu_launch += 1
        elif e.name in ("cudaGraphLaunch", "cuGraphLaunch"):
            graph_launch += 1
    return {"gpu_kernels": kern, "gpu_memcpy_memset": mem, "cpu_kernel_launch_calls": cpu_launch,
            "cpu_graph_launch_calls": graph_launch, "top_kernels": names.most_common(6)}


class ByteCounter(TorchDispatchMode):
    SKIP = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "detach", "alias", "lift_fresh",
            "_has_compatible_shallow_copy_type"}

    def __init__(self):
        super().__init__()
        self.bytes = 0
        self.by_op = collections.Counter()
        self.n_ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name.split("::")[1]
        if name in self.SKIP:
            return out
        rets = func._schema.returns
        if rets and all(r.alias_info is not None and not r.alias_info.is_write for r in rets):
            return out
        ins, _ = tree_flatten((args, kwargs))
        outs, _ = tree_flatten(out)
        seen = set()
        b = 0
        for i, t in enumerate(ins):
            if isinstance(t, torch.Tensor) and t.numel() > 1:
                if name in ("copy_", "fill_", "zero_") and i == 0:
                    continue
                key = (t.untyped_storage().data_ptr(), t.storage_offset(), tuple(t.shape))
                if key not in seen:
                    seen.add(key)
                    b += t.numel() * t.element_size()
        for t in outs:
            if isinstance(t, torch.Tensor) and t.numel() > 1:
                b += t.numel() * t.element_size()
        self.bytes += b
        self.by_op[name] += b
        self.n_ops += 1
        return out


def trace_bytes(fn):
    bc = ByteCounter()
    with torch.no_grad(), bc:
        fn()
    return bc


def fused_dw_bytes(trunk, fn):
    tot = [0]
    hooks = []

    def hook(mod, inp, out):
        x = inp[0]
        tot[0] += x.numel() * x.element_size() + out.numel() * out.element_size() + (mod.w.numel() + mod.b.numel()) * 4

    for m in trunk.modules():
        if isinstance(m, FusedDWConvBNSiLU):
            hooks.append(m.register_forward_hook(hook))
    fn()
    for h in hooks:
        h.remove()
    return tot[0]


class ScanBytes:
    def __init__(self):
        self.bytes = 0
        self.orig = ss_mod.selective_scan_fused

    def __enter__(self):
        def wrap(x, delta, A, Bm, Cm, D, block_e=64):
            self.bytes += (x.numel() + delta.numel() + x.numel() + Bm.numel() + Cm.numel() + A.numel() + D.numel()) * 4
            return self.orig(x, delta, A, Bm, Cm, D, block_e)
        ss_mod.selective_scan_fused = wrap
        return self

    def __exit__(self, *a):
        ss_mod.selective_scan_fused = self.orig


def kernel_a_bytes(B, cin, cout, dm, P, s, mode):
    x = B * P * cin * s
    w = cout * cin * s + cout * 4
    wp = dm * cout * s + dm * 4
    out = B * dm * 4
    pooled = 2 * B * cout * 4 if mode == "two" else 0
    prefill = out if mode == "atomic" else 0
    rmw = 2 * out if mode == "atomic" else out
    return {"x_map": x, "conv_weights": w, "proj_weights": wp, "pooled_roundtrip": pooled, "out": rmw, "prefill": prefill,
            "total": x + w + wp + pooled + rmw + prefill, "weights": w + wp, "state_rw": 0,
            "activations": x + pooled + rmw + prefill}


def kernel_b_bytes(B, D, E, N, H, s, nl, fuse_out):
    per = {}
    per["k1_x_in"] = B * D * 4
    per["k1_norm_params"] = 2 * D * 4
    per["k1_in_proj_weights"] = 2 * E * D * s
    per["k1_conv_weights_bias"] = E * 5 * 4
    per["k1_conv_state_rw"] = 2 * B * 3 * E * 4
    per["k1_u_and_silu_z_write"] = 2 * B * E * 4
    per["k1_residual_copy_write"] = B * D * 4
    per["k2_u_and_silu_z_read"] = 2 * B * E * 4
    per["k2_xproj_dt_weights"] = (2 * N * E + E * E) * s + E * 4
    per["k2_A_D"] = (E * N + E) * 4
    per["k2_ssm_state_rw"] = 2 * B * E * N * 4
    per["k2_out_proj_weights"] = D * E * s
    if fuse_out:
        per["k2_residual_atomic_rw"] = 2 * B * D * 4
    else:
        per["k2_gated_y_write"] = B * E * 4
        per["k3_gated_y_read"] = B * E * 4
        per["k3_residual_rw"] = 2 * B * D * 4
    layer = sum(per.values())
    wk = ("k1_in_proj_weights", "k1_conv_weights_bias", "k1_norm_params", "k2_xproj_dt_weights", "k2_A_D", "k2_out_proj_weights")
    layer_w = sum(per[k] for k in wk)
    layer_state = per["k1_conv_state_rw"] + per["k2_ssm_state_rw"]
    head = {"x_in": B * D * 4, "norm_f": 2 * D * 4, "ysum_rw": 2 * B * D * 4, "cnt_rw": 2 * B * 4,
            "head_weights": (H * D + 2 * H) * s + (H + 2) * 4, "logits": B * 2 * 4}
    head_w = head["norm_f"] + head["head_weights"]
    head_state = head["ysum_rw"] + head["cnt_rw"]
    total = layer * nl + sum(head.values())
    return {"per_layer_terms": per, "per_layer_total": layer, "head_terms": head, "head_total": sum(head.values()),
            "total": total, "weights": layer_w * nl + head_w, "state_rw": layer_state * nl + head_state,
            "activations": total - layer_w * nl - head_w - layer_state * nl - head_state}


def peak_run(fn, iters=30):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return {"peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20, "resident_before_run_MB": base / 2**20,
            "reserved_MB": torch.cuda.max_memory_reserved() / 2**20}


def tune():
    out = {}
    features, temporal, _ = load_parts("diloco")
    for dn in ("bf16", "fp32"):
        dtype = DT[dn]
        t = {}
        for B in (1, 8):
            fm = torch.randn(B, 320, 7, 7, device="cuda", dtype=dtype)
            o = torch.empty(B, 512, device="cuda")
            for mode in ("two", "atomic"):
                for bnt in (32, 64):
                    k = FusedTailProj(features[8][0], features[8][1], temporal.proj, dtype, mode=mode, bn_tile=bnt)
                    t[f"A_B{B}_{mode}_bn{bnt}_us"] = graph_inner_time(lambda: k(fm, out=o)) * 1e3
            xs = torch.randn(B, 512, device="cuda")
            for fo in (True, False):
                for tag, (be1, be2, bk, nw1, nw2) in (("small", (16, 16, 128, 4, 8)), ("large", (32, 32, 64, 4, 4))):
                    st = FusedMambaStack(temporal, B, dtype, fuse_out=fo, be1=be1, be2=be2, bk=bk, nw1=nw1, nw2=nw2)
                    st.xbuf[0].copy_(xs)
                    t[f"B_B{B}_fuseout{int(fo)}_{tag}_us"] = graph_inner_time(st.step_from_buf0) * 1e3
        out[dn] = t
        print(dn, json.dumps({k: round(v, 1) for k, v in t.items()}), flush=True)
    (RES / "stream_tuning.json").write_text(json.dumps(out, indent=1))


def make_methods(name, dn, B, cfg):
    features, temporal, _ = load_parts(name)
    dtype = DT[dn]
    x = torch.randn(B, 3, 224, 224, device="cuda", dtype=dtype)
    m = {}
    m["fused"] = lambda: StreamPipeline(features, temporal, B, dtype, cfg["tail_mode"], cfg["fuse_out"], True, cfg["be1"], cfg["be2"], cfg["bn_tile"])
    m["torch_stream"] = lambda: TorchStreamBaseline(features, temporal, B, dtype)
    m["window_fused_scan"] = lambda: WindowBaseline(features, temporal, B, dtype, 30, False, True, False)
    m["window_attention"] = lambda: WindowBaseline(features, temporal, B, dtype, 30, False, True, True)
    m["window_full_recompute"] = lambda: WindowBaseline(features, temporal, B, dtype, 30, True, True, False)
    return m, x


def run_method(label, build, x, iters_eager, iters_graph, graph=True):
    obj = build()
    rec = {"label": label}
    fe = lambda: obj.step(x)
    rec["eager"] = timeit(fe, iters_eager)
    rec["launches_eager"] = count_launches(fe)
    rec["memory_eager"] = peak_run(fe, 10)
    rec["state_bytes_per_stream"] = obj.state_bytes() / x.shape[0]
    if graph:
        obj.reset()
        obj.capture()
        fg = lambda: obj.step_graph(x)
        rec["graph"] = timeit(fg, iters_graph)
        rec["launches_graph"] = count_launches(fg, True)
        rec["memory_graph"] = peak_run(fg, 10)
    del obj
    gc.collect()
    torch.cuda.empty_cache()
    return rec


def analytic(name, dn, B, cfg):
    features, temporal, _ = load_parts(name)
    dtype = DT[dn]
    s = torch.tensor([], dtype=dtype).element_size()
    x = torch.randn(B, 3, 224, 224, device="cuda", dtype=dtype)
    base = TorchStreamBaseline(features, temporal, B, dtype)
    with torch.no_grad():
        fmap = base.part_trunk(x)
        tb = trace_bytes(lambda: base.part_trunk(x)).bytes
        pooled = base.part_tail(fmap)
        tail_b = trace_bytes(lambda: base.part_tail(fmap)).bytes
        rest_b = trace_bytes(lambda: base.part_rest(pooled)).bytes
    pipe = StreamPipeline(features, temporal, B, dtype, cfg["tail_mode"], cfg["fuse_out"])
    with torch.no_grad():
        ft = trace_bytes(lambda: pipe.trunk(x)).bytes
        dwb = fused_dw_bytes(pipe.trunk, lambda: pipe.trunk(x))
    conv = features[8][0]
    ka = kernel_a_bytes(B, conv.in_channels, conv.out_channels, temporal.proj.out_features, 49, s, cfg["tail_mode"])
    mix = temporal.layers[0].mixer
    kb = kernel_b_bytes(B, mix.d_model, mix.d_inner, mix.d_state, temporal.head[0].out_features, s, len(temporal.layers), cfg["fuse_out"])
    win = WindowBaseline(features, temporal, B, dtype, 30, False, True, False)
    att = WindowBaseline(features, temporal, B, dtype, 30, False, True, True)
    with torch.no_grad():
        wf = win.feat
        with ScanBytes() as sb:
            w_head = trace_bytes(lambda: win._head(wf))
        w_scan = sb.bytes
        torch.backends.mha.set_fastpath_enabled(False)
        a_head = trace_bytes(lambda: att._head(wf)).bytes
        torch.backends.mha.set_fastpath_enabled(True)
        w_trunk_tail = tb + tail_b
        wtr = WindowBaseline(features, temporal, B, dtype, 30, False, True, False)
        w_front = trace_bytes(lambda: wtr.features(x)).bytes + fused_dw_bytes(wtr.features, lambda: wtr.features(x))
        w_pool = trace_bytes(lambda: wtr.features(x).mean((2, 3))).bytes - trace_bytes(lambda: wtr.features(x)).bytes
    unfused_total = tb + tail_b + rest_b
    fused_after = ka["total"] + kb["total"]
    return {
        "elem_bytes": s,
        "unfused_trunk": tb, "unfused_tail_pool": tail_b, "unfused_proj_mamba_head": rest_b, "unfused_total": unfused_total,
        "fused_trunk": ft + dwb, "fused_kernel_A": ka, "fused_kernel_B": kb, "fused_after_trunk": fused_after,
        "fused_total": ft + dwb + fused_after,
        "unfused_after_trunk": tail_b + rest_b,
        "after_trunk_ratio": (tail_b + rest_b) / fused_after,
        "total_ratio": unfused_total / (ft + dwb + fused_after),
        "window_fused_scan_head": w_head.bytes + w_scan, "window_fused_scan_total": w_front + w_pool + w_head.bytes + w_scan,
        "window_attention_head": a_head, "window_attention_total": w_front + w_pool + a_head,
        "window_full_recompute_total": 30 * (w_front + w_pool) + w_head.bytes + w_scan,
    }


def stage_timing(name, dn, B, cfg):
    features, temporal, _ = load_parts(name)
    dtype = DT[dn]
    pipe = StreamPipeline(features, temporal, B, dtype, cfg["tail_mode"], cfg["fuse_out"], True, cfg["be1"], cfg["be2"], cfg["bn_tile"])
    x = torch.randn(B, 3, 224, 224, device="cuda", dtype=dtype)
    base = TorchStreamBaseline(features, temporal, B, dtype)
    win = WindowBaseline(features, temporal, B, dtype, 30, False, True, False)
    att = WindowBaseline(features, temporal, B, dtype, 30, False, True, True)
    with torch.no_grad():
        fmap = pipe.trunk(x)
        pooled_t = base.part_tail(base.part_trunk(x))
        r = {
            "trunk_fused_dw_ms": graph_inner_time(lambda: pipe.trunk(x)),
            "trunk_unfused_ms": graph_inner_time(lambda: base.part_trunk(x)),
            "tail_pool_proj_torch_ms": graph_inner_time(lambda: base.t.proj(base.part_tail(fmap))),
            "kernel_A_ms": graph_inner_time(lambda: pipe.tail(fmap, out=pipe.stack.xbuf[0])),
            "mamba_stack_head_torch_ms": graph_inner_time(lambda: base.part_rest(pooled_t)),
            "kernel_B_ms": graph_inner_time(pipe.stack.step_from_buf0),
        }
        r["launches_after_trunk"] = {
            "fused_A_plus_B": count_launches(lambda: (pipe.tail(fmap, out=pipe.stack.xbuf[0]), pipe.stack.step_from_buf0())),
            "torch_tail_plus_step": count_launches(lambda: base.part_rest(base.part_tail(fmap))),
            "window_fused_scan_head": count_launches(lambda: win._head(win.feat)),
            "window_attention_head": count_launches(lambda: att._head(att.feat))}
    return r


def pick_cfg(path):
    cfg = {"tail_mode": "two", "fuse_out": True, "be1": None, "be2": None, "bn_tile": None}
    p = RES / "stream_cfg.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["tune", "final", "quick", "render"])
    ap.add_argument("--no-idle", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if a.part == "tune":
        tune()
        return
    if a.part == "render":
        from render import render
        render()
        return
    cfg = pick_cfg(None)
    idle = None if a.no_idle else wait_idle()
    result = {"env_start": env_record(), "idle_samples_pct": idle, "cfg": cfg, "torch": torch.__version__,
              "note": "latency = wall clock per new frame incl. sync, input image already preprocessed on GPU"}
    runs = [("diloco", "bf16", 1), ("diloco", "bf16", 8), ("diloco", "fp32", 1), ("diloco", "fp32", 8),
            ("finetuned", "bf16", 1), ("finetuned", "bf16", 8)]
    if a.part == "quick":
        runs = runs[:1]
    result["runs"] = {}
    result["skipped"] = {}
    from pipeline import CKPTS
    for name, dn, B in runs:
        key = f"{name}_{dn}_B{B}"
        if not Path(CKPTS[name]).exists():
            result["skipped"][key] = f"checkpoint missing: {CKPTS[name]}"
            print("skip", key, result["skipped"][key], flush=True)
            continue
        print("==", key, flush=True)
        methods, x = make_methods(name, dn, B, cfg)
        r = {"methods": {}}
        for label, build in methods.items():
            naive = label == "window_full_recompute"
            r["methods"][label] = run_method(label, build, x, 30 if naive else 150, 30 if naive else 200)
            m = r["methods"][label]
            print(label, "eager", round(m["eager"]["median_ms"], 3), "graph", round(m["graph"]["median_ms"], 3),
                  "kernels", m["launches_eager"]["gpu_kernels"], m["launches_graph"]["gpu_kernels"], flush=True)
        r["stages_graph_ms"] = stage_timing(name, dn, B, cfg)
        r["bytes"] = analytic(name, dn, B, cfg)
        r["env_after"] = env_record()
        result["runs"][key] = r
        print(key, "bytes ratio after-trunk", round(r["bytes"]["after_trunk_ratio"], 2), "total", round(r["bytes"]["total_ratio"], 2), flush=True)
    result["env_end"] = env_record()
    (RES / ("stream_fused.json" if a.part == "final" else "stream_quick.json")).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
