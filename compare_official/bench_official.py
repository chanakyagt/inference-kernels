import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = Path(os.environ.get("MAMBA_OFFICIAL_REPO", Path.home() / "sota" / "mamba_official"))
OFFICIAL_BUILD = Path(os.environ.get("MAMBA_OFFICIAL_BUILD", Path.home() / "sota" / "mamba_build"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT / "cuda"))
sys.path.insert(0, str(OFFICIAL_BUILD))
from common import DETECTOR_REPO

sys.path.insert(0, str(DETECTOR_REPO))

torch.set_grad_enabled(False)
DEV = "cuda"
N = 16
SPEC_BW = 288e9


def smi():
    q = "utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,power.draw"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(", ")
    return dict(zip(["util_pct", "temp_c", "sm_mhz", "mem_mhz", "power_w"], [float(v) for v in out]))


def warm_clocks(seconds=1.5):
    m = torch.randn(4096, 4096, device=DEV)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(8):
            m @ m
        torch.cuda.synchronize()
    del m


def wait_idle(max_wait_s):
    t0 = time.time()
    ok = 0
    while time.time() - t0 < max_wait_s:
        ok = ok + 1 if smi()["util_pct"] <= 5 else 0
        if ok >= 3:
            return True
        time.sleep(5)
    return False


def load_official_triton():
    pkg = {}
    for name, sub in (("mamba_ssm", ""), ("mamba_ssm.ops", "ops"), ("mamba_ssm.ops.triton", "ops/triton")):
        m = types.ModuleType(name)
        m.__path__ = [str(OFFICIAL_REPO / "mamba_ssm" / sub)]
        sys.modules[name] = m
        pkg[name] = m
    for mod in ("softplus", "selective_state_update"):
        full = f"mamba_ssm.ops.triton.{mod}"
        spec = importlib.util.spec_from_file_location(full, OFFICIAL_REPO / "mamba_ssm" / "ops" / "triton" / f"{mod}.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
    return sys.modules["mamba_ssm.ops.triton.selective_state_update"].selective_state_update


_OURS_EXT = None


def load_ours_cuda():
    global _OURS_EXT
    if _OURS_EXT is None:
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ["CUDA_HOME"] + "/bin", os.environ["PATH"]])
        from torch.utils.cpp_extension import load
        _OURS_EXT = load(name="ours_selective_scan_cuda", sources=[str(ROOT / "cuda" / "selective_scan.cu")],
                         extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)
    return _OURS_EXT


def median_ms(ts):
    return statistics.median(ts)


def time_eager(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return median_ms(ts)


def time_graph(fn, k, iters, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / k)
    del g
    return median_ms(ts)


def time_event_gap(fn, iters, buf=None, warmup=10):
    def pre(i):
        if buf is not None:
            buf.fill_(i % 100 + 1)
        else:
            torch.cuda._sleep(400000)

    for i in range(warmup):
        pre(i)
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        pre(i)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return median_ms(ts)


def safe(label, f, errors):
    try:
        return f()
    except Exception as e:
        errors.append(f"{label}: {type(e).__name__}: {str(e)[:300]}")
        torch.cuda.synchronize()
        return None


def scan_reference(x, dt, A, Bm, Cm, D):
    Bsz, L, E = x.shape
    x, dt, Bm, Cm = x.float(), dt.float(), Bm.float(), Cm.float()
    h = torch.zeros(Bsz, E, N, device=x.device)
    ys = []
    for t in range(L):
        h = h * torch.exp(dt[:, t, :, None] * A[None]) + (dt[:, t] * x[:, t])[:, :, None] * Bm[:, t, None, :]
        ys.append((h * Cm[:, t, None, :]).sum(-1) + D[None] * x[:, t])
    return torch.stack(ys, 1)


def make_scan_inputs(B, E, T, dtype):
    g = torch.Generator(device=DEV)
    g.manual_seed(1234)
    x = torch.randn(B, T, E, device=DEV, generator=g).to(dtype)
    delta = (torch.rand(B, T, E, device=DEV, generator=g) * 0.099 + 0.001).to(dtype)
    A = (-torch.arange(1, N + 1, device=DEV, dtype=torch.float32)).repeat(E, 1).contiguous()
    Bm = torch.randn(B, T, N, device=DEV, generator=g).to(dtype)
    Cm = torch.randn(B, T, N, device=DEV, generator=g).to(dtype)
    D = torch.ones(E, device=DEV)
    return x, delta, A, Bm, Cm, D


def scan_bytes(B, E, T, dtype):
    s = 2 if dtype == torch.bfloat16 else 4
    return (3 * B * T * E + 2 * B * T * N) * s + E * N * 4 + E * 4


def to_official_layout(x, delta, Bm, Cm):
    u = x.transpose(1, 2).contiguous()
    d = delta.transpose(1, 2).contiguous()
    Bo = Bm.transpose(1, 2).contiguous().unsqueeze(1)
    Co = Cm.transpose(1, 2).contiguous().unsqueeze(1)
    return u, d, Bo, Co


def bench_scan(B, E, T, dtype, a, ops, rows, errors):
    from selective_scan import selective_scan_fused
    ext = load_ours_cuda()

    def scan_cuda(x, delta, A, Bm, Cm, D, cpb=None, s=None):
        return ext.scan(x.contiguous(), delta.contiguous(), A.float().contiguous(), Bm.contiguous(),
                        Cm.contiguous(), D.float().contiguous(), cpb or 32, s or 2)

    x, delta, A, Bm, Cm, D = make_scan_inputs(B, E, T, dtype)
    u, d, Bo, Co = to_official_layout(x, delta, Bm, Cm)
    ref = scan_reference(x, delta, A, Bm, Cm, D)
    ref_absmax = ref.abs().max().item()
    outs = {}
    fns = {}

    def ours_cuda(cpb=None, s=None):
        return lambda: scan_cuda(x, delta, A, Bm, Cm, D, cpb, s)

    def ours_tri(be=64):
        return lambda: selective_scan_fused(x, delta, A, Bm, Cm, D, block_e=be)

    best_cuda = None
    best_tri = None
    fixed = tuple(int(v) for v in a.fixed_cfg.split(",")) if a.fixed_cfg else None
    if fixed:
        best_cuda, best_tri = fixed, None
    elif not a.quick_tune:
        cfgs = [(c, s) for c in (16, 32, 64, 128, 256) for s in (1, 2, 4, 8, 16) if 32 <= c * s <= 1024]
        res = []
        for c, s in cfgs:
            t = safe(f"tune cuda {c},{s}", lambda: time_graph(ours_cuda(c, s), 5, a.tune_iters), errors)
            if t is not None:
                res.append((t, c, s))
        best_cuda = min(res)[1:] if res else (32, 2)
        res = []
        for be in (16, 32, 64, 128):
            t = safe(f"tune triton {be}", lambda: time_graph(ours_tri(be), 5, a.tune_iters), errors)
            if t is not None:
                res.append((t, be))
        best_tri = min(res)[1] if res else 64
    else:
        best_cuda, best_tri = (32, 2), 64

    if fixed:
        fns["ours_cuda_fixed"] = ours_cuda(*fixed)
    else:
        fns["ours_cuda_tuned"] = ours_cuda(*best_cuda)
        fns["ours_cuda_default"] = ours_cuda()
        fns["ours_triton_tuned"] = ours_tri(best_tri)
        fns["ours_triton_default"] = ours_tri()

    d_v = d.clone()
    states_v = torch.zeros(B, E, N, device=DEV)

    def vllm_call():
        ops["vllm_scan"](u, states_v, d_v, A, Bo.squeeze(1), Co.squeeze(1), D, None, None, False)
        return d_v

    fns["vllm_scan"] = vllm_call

    d_vr = d.clone()

    def vllm_restore():
        d_vr.copy_(d)
        ops["vllm_scan"](u, states_v, d_vr, A, Bo.squeeze(1), Co.squeeze(1), D, None, None, False)
        return d_vr

    fns["vllm_scan_with_delta_restore"] = vllm_restore

    if "official_scan" in ops:
        def official_call():
            return ops["official_scan"](u, d, A, Bo, Co, D, None, None, False)[0]

        fns["official_scan"] = official_call

        def official_layout():
            uu, dd, bb, cc = to_official_layout(x, delta, Bm, Cm)
            return ops["official_scan"](uu, dd, A, bb, cc, D, None, None, False)[0].transpose(1, 2).contiguous()

        fns["official_scan_with_layout_transposes"] = official_layout

    row = dict(batch=B, d_inner=E, T=T, dtype=str(dtype).replace("torch.", ""), d_state=N,
               cuda_best_cfg=list(best_cuda), triton_best_block_e=best_tri, ref_absmax=ref_absmax,
               analytic_min_bytes=scan_bytes(B, E, T, dtype), err_vs_fp32_ref={}, max_abs_diff_vs_official=None,
               eager_ms={}, graph_ms={}, graph_bw_GBps={}, eager_bw_GBps={})
    for name, f in fns.items():
        o = safe(f"run {name}", f, errors)
        if o is None:
            continue
        o = o.float()
        if name.startswith("vllm") or name.startswith("official"):
            o = o.transpose(1, 2) if o.shape == (B, E, T) else o
        outs[name] = o
        row["err_vs_fp32_ref"][name] = (o - ref).abs().max().item()
    if "official_scan" in outs:
        row["max_abs_diff_vs_official"] = {n: (outs[n] - outs["official_scan"]).abs().max().item()
                                           for n in ("ours_cuda_tuned", "ours_cuda_fixed", "ours_triton_tuned", "vllm_scan") if n in outs}
    if "vllm_scan" in outs:
        ok = "ours_cuda_fixed" if "ours_cuda_fixed" in outs else "ours_cuda_tuned"
        row["max_abs_diff_ours_cuda_vs_vllm"] = (outs[ok] - outs["vllm_scan"]).abs().max().item() if ok in outs else None
    if a.idle_per_row:
        row["idle_gate_met"] = wait_idle(a.idle_per_row)
        warm_clocks()
    row["smi_before"] = smi()
    k = 5
    for name, f in fns.items():
        if name not in outs:
            continue
        e = safe(f"eager {name}", lambda: time_eager(f, a.iters), errors)
        g = None
        if "with_layout" not in name and "restore" not in name:
            g = safe(f"graph {name}", lambda: time_graph(f, k, a.iters), errors)
        if e is not None:
            row["eager_ms"][name] = e
            row["eager_bw_GBps"][name] = row["analytic_min_bytes"] / (e * 1e-3) / 1e9
        if g is not None:
            row["graph_ms"][name] = g
            row["graph_bw_GBps"][name] = row["analytic_min_bytes"] / (g * 1e-3) / 1e9
    if a.cold_l2:
        buf = torch.empty(64 * 2**20, dtype=torch.uint8, device=DEV)
        row["cold_l2_ms"], row["warm_event_ms"] = {}, {}
        for name, f in fns.items():
            if name not in outs or "with_layout" in name or "restore" in name:
                continue
            c = safe(f"cold {name}", lambda: time_event_gap(f, a.iters, buf), errors)
            w = safe(f"warm_ev {name}", lambda: time_event_gap(f, a.iters, None), errors)
            if c is not None:
                row["cold_l2_ms"][name] = c
            if w is not None:
                row["warm_event_ms"][name] = w
        del buf
    row["smi_after"] = smi()
    rows.append(row)
    print(f"scan B={B} E={E} T={T} {row['dtype']}: graph {json.dumps({k: round(v, 5) for k, v in row['graph_ms'].items()})}", flush=True)
    del x, delta, Bm, Cm, u, d, Bo, Co, ref, d_v, d_vr
    torch.cuda.empty_cache()


def step_core_bytes(B, E):
    return 2 * B * E * N * 4 + 3 * B * E * 4 + E * N * 4 + E * 4 + 2 * B * N * 4


def step_core_ref(h, x, dt, A, Bm, Cm, D):
    hn = h * torch.exp(dt[..., None] * A[None]) + (dt * x)[..., None] * Bm[:, None, :]
    return hn, (hn * Cm[:, None, :]).sum(-1) + D[None] * x


def bench_step_core(B, E, a, ops, rows, errors):
    from mamba_step import ssm_step_fused
    g = torch.Generator(device=DEV)
    g.manual_seed(7)
    x = torch.randn(B, E, device=DEV, generator=g)
    dt = torch.rand(B, E, device=DEV, generator=g) * 0.099 + 0.001
    A = (-torch.arange(1, N + 1, device=DEV, dtype=torch.float32)).repeat(E, 1).contiguous()
    Bm = torch.randn(B, N, device=DEV, generator=g)
    Cm = torch.randn(B, N, device=DEV, generator=g)
    D = torch.ones(E, device=DEV)
    h0 = torch.randn(B, E, N, device=DEV, generator=g)
    zero_bias = torch.zeros(E, device=DEV)
    hr, yr = step_core_ref(h0, x, dt, A, Bm, Cm, D)
    states = {k: h0.clone() for k in ("ours", "official", "vllm")}
    out_v = torch.empty(B, E, device=DEV)
    fns = {
        "ours_triton_ssm_step": lambda: ssm_step_fused(x, dt, A, Bm, Cm, D, states["ours"]),
        "official_triton_ssu": lambda: ops["official_ssu"](states["official"], x, dt, A, Bm, Cm, D, z=None, dt_bias=None, dt_softplus=False),
        "vllm_triton_ssu": lambda: (ops["vllm_ssu"](states["vllm"], x, dt, A, Bm, Cm, D, zero_bias, z=None, dt_softplus=False, out=out_v), out_v)[1],
    }
    row = dict(batch=B, d_inner=E, d_state=N, dtype="float32", analytic_min_bytes=step_core_bytes(B, E),
               err_y_vs_ref={}, err_state_vs_ref={}, eager_ms={}, graph_ms={}, eager_bw_GBps={}, graph_bw_GBps={})
    for name, f in fns.items():
        y = safe(f"run {name}", f, errors)
        if y is None:
            continue
        key = {"ours_triton_ssm_step": "ours", "official_triton_ssu": "official", "vllm_triton_ssu": "vllm"}[name]
        row["err_y_vs_ref"][name] = (y.float() - yr).abs().max().item()
        row["err_state_vs_ref"][name] = (states[key] - hr).abs().max().item()
    row["smi_before"] = smi()
    for name, f in fns.items():
        if name not in row["err_y_vs_ref"]:
            continue
        e = safe(f"eager {name}", lambda: time_eager(f, a.iters), errors)
        gm = safe(f"graph {name}", lambda: time_graph(f, 20, a.iters), errors)
        if e is not None:
            row["eager_ms"][name] = e
            row["eager_bw_GBps"][name] = row["analytic_min_bytes"] / (e * 1e-3) / 1e9
        if gm is not None:
            row["graph_ms"][name] = gm
            row["graph_bw_GBps"][name] = row["analytic_min_bytes"] / (gm * 1e-3) / 1e9
    row["smi_after"] = smi()
    rows.append(row)
    print(f"step_core B={B} E={E}: graph {json.dumps({k: round(v, 5) for k, v in row['graph_ms'].items()})}", flush=True)


def make_temporal(E):
    from dfmamba.model import TemporalClassifier
    torch.manual_seed(0)
    return TemporalClassifier(d_model=E // 2, n_layers=1, d_state=N, feat_dim=64).to(DEV).eval()


def bench_step_fused(B, E, a, ops, rows, errors, temporal_cache):
    from fused_mamba_step import FusedMambaStack
    if E not in temporal_cache:
        temporal_cache[E] = make_temporal(E)
    temporal = temporal_cache[E]
    st = FusedMambaStack(temporal, B, torch.float32)
    p = st.layers[0]
    Dm = st.D
    g = torch.Generator(device=DEV)
    g.manual_seed(11)
    x0 = torch.randn(B, Dm, device=DEV, generator=g)
    h_init = torch.randn(B, E, N, device=DEV, generator=g) * 0.1
    cs_init = torch.randn(B, 3, E, device=DEV, generator=g) * 0.1
    st.h[0].copy_(h_init)
    st.cs[0].copy_(cs_init)
    xout = torch.empty(B, Dm, device=DEV)

    def ours():
        st.layer(0, x0, xout)
        return xout

    cs_c = cs_init.clone()
    states = {"official": h_init.clone(), "vllm": h_init.clone()}
    out_v = torch.empty(B, E, device=DEV)

    def composed(which):
        xn = F.layer_norm(x0, (Dm,), p["g"], p["be"], p["eps"])
        u, z = F.linear(xn, p["win"]).chunk(2, -1)
        win4 = torch.cat([cs_c, u.unsqueeze(1)], 1)
        cs_c.copy_(win4[:, 1:])
        c = (win4 * p["cw"].t()[None]).sum(1) + p["cb"]
        u2 = F.silu(c)
        dt = F.linear(u2, p["wdt"])
        Bm, Cm = F.linear(u2, p["wx"]).chunk(2, -1)
        if which == "official":
            y = ops["official_ssu"](states["official"], u2, dt, p["A"], Bm, Cm, p["Dp"], z=z, dt_bias=p["bdt"], dt_softplus=True)
        else:
            ops["vllm_ssu"](states["vllm"], u2, dt, p["A"], Bm, Cm, p["Dp"], p["bdt"], z=z, dt_softplus=True, out=out_v)
            y = out_v
        return x0 + F.linear(y, p["wout"])

    ref_x = None
    fns = {"ours_fused_layer": ours,
           "composed_torch_plus_official_triton_ssu": lambda: composed("official"),
           "composed_torch_plus_vllm_triton_ssu": lambda: composed("vllm")}
    wbytes = (p["win"].numel() + p["wx"].numel() + p["wdt"].numel() + p["wout"].numel()) * 4
    mb = wbytes + 2 * B * E * N * 4 + 2 * B * 3 * E * 4 + 2 * B * Dm * 4 + (E * N + E + 4 * E + E + E + 2 * Dm) * 4
    row = dict(batch=B, d_inner=E, d_model=Dm, d_state=N, dtype="float32", analytic_min_bytes=mb, weight_bytes=wbytes,
               max_abs_diff_vs_composed_official={}, eager_ms={}, graph_ms={}, eager_bw_GBps={}, graph_bw_GBps={})
    outs = {}
    for name, f in fns.items():
        st.h[0].copy_(h_init)
        st.cs[0].copy_(cs_init)
        cs_c.copy_(cs_init)
        states["official"].copy_(h_init)
        states["vllm"].copy_(h_init)
        o = safe(f"run {name}", f, errors)
        if o is not None:
            outs[name] = o.clone()
    if "composed_torch_plus_official_triton_ssu" in outs:
        ref_x = outs["composed_torch_plus_official_triton_ssu"]
        row["max_abs_diff_vs_composed_official"] = {n: (o - ref_x).abs().max().item() for n, o in outs.items()}
        row["ref_absmax"] = ref_x.abs().max().item()
    row["smi_before"] = smi()
    for name, f in fns.items():
        if name not in outs:
            continue
        e = safe(f"eager {name}", lambda: time_eager(f, a.iters), errors)
        gm = safe(f"graph {name}", lambda: time_graph(f, 10, a.iters), errors)
        if e is not None:
            row["eager_ms"][name] = e
            row["eager_bw_GBps"][name] = mb / (e * 1e-3) / 1e9
        if gm is not None:
            row["graph_ms"][name] = gm
            row["graph_bw_GBps"][name] = mb / (gm * 1e-3) / 1e9
    row["smi_after"] = smi()
    rows.append(row)
    print(f"step_fused B={B} E={E}: graph {json.dumps({k: round(v, 5) for k, v in row['graph_ms'].items()})}", flush=True)


def fmt(v, p=4):
    return "-" if v is None else f"{v:.{p}f}"


def write_markdown(res, path):
    L = []
    m = res["meta"]
    L.append("# Official-kernel comparison")
    L.append("")
    L.append(f"GPU {m['gpu']}, torch {m['torch']}, vLLM {m['vllm']}, official mamba-ssm commit {m['official_commit']}. "
             f"Timing: CUDA events, median of {m['iters']}. idle gate met: {m['idle_gate_met']}. "
             f"SM clock at start/end: {m['smi_start']['sm_mhz']:.0f}/{m['smi_end']['sm_mhz']:.0f} MHz, temp {m['smi_start']['temp_c']:.0f}/{m['smi_end']['temp_c']:.0f} C.")
    L.append("")
    L.append("GPU-time column = CUDA graph of 5 back-to-back calls divided by 5 (no CPU launch cost). Eager column = events around one Python call (includes launch cost). Ratio = official time / ours; above 1 means ours is faster. Ours = best of a config sweep (CUDA kernel threads/channels and split; Triton block size) vs the official kernels at their fixed settings.")
    L.append("")
    L.append("## Full-sequence scan (GPU time, ms)")
    L.append("")
    L.append("| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | official scan | official/ours(CUDA) | vLLM/ours(CUDA) | official/ours(Triton) | CUDA GB/s | vLLM GB/s | err ours CUDA | err vLLM | err official |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["scan"]:
        g = r["graph_ms"]
        o = g.get("official_scan")
        v = g.get("vllm_scan")
        c = g.get("ours_cuda_tuned")
        ev = r["err_vs_fp32_ref"]
        L.append(f"| {r['batch']} | {r['d_inner']} | {r['T']} | {r['dtype']} | {fmt(c)} | {fmt(g.get('ours_triton_tuned'))} | {fmt(v)} | {fmt(o)} | "
                 f"{fmt(o / c if o and c else None, 2)} | {fmt(v / c if v and c else None, 2)} | {fmt(o / g.get('ours_triton_tuned') if o and g.get('ours_triton_tuned') else None, 2)} | {fmt(r['graph_bw_GBps'].get('ours_cuda_tuned'), 1)} | {fmt(r['graph_bw_GBps'].get('vllm_scan'), 1)} | "
                 f"{ev.get('ours_cuda_tuned', float('nan')):.2e} | {ev.get('vllm_scan', float('nan')):.2e} | {ev.get('official_scan', float('nan')):.2e} |")
    L.append("")
    L.append("## Full-sequence scan (eager, ms, includes Python launch cost)")
    L.append("")
    L.append("| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | vLLM scan + delta restore | official scan | official scan + layout transposes |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in res["scan"]:
        e = r["eager_ms"]
        L.append(f"| {r['batch']} | {r['d_inner']} | {r['T']} | {r['dtype']} | {fmt(e.get('ours_cuda_tuned'))} | {fmt(e.get('ours_triton_tuned'))} | {fmt(e.get('vllm_scan'))} | "
                 f"{fmt(e.get('vllm_scan_with_delta_restore'))} | {fmt(e.get('official_scan'))} | {fmt(e.get('official_scan_with_layout_transposes'))} |")
    L.append("")
    L.append("## Single-token state update, like-for-like core (fp32, GPU time ms)")
    L.append("")
    L.append("| B | E | ours Triton | official Triton | vLLM Triton | official/ours | vLLM/ours | ours GB/s | official GB/s | vLLM GB/s | err y ours | err y official | err y vLLM |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["step_core"]:
        g = r["graph_ms"]
        o, v, c = g.get("official_triton_ssu"), g.get("vllm_triton_ssu"), g.get("ours_triton_ssm_step")
        e = r["err_y_vs_ref"]
        b = r["graph_bw_GBps"]
        L.append(f"| {r['batch']} | {r['d_inner']} | {fmt(c)} | {fmt(o)} | {fmt(v)} | {fmt(o / c if o and c else None, 2)} | {fmt(v / c if v and c else None, 2)} | "
                 f"{fmt(b.get('ours_triton_ssm_step'), 1)} | {fmt(b.get('official_triton_ssu'), 1)} | {fmt(b.get('vllm_triton_ssu'), 1)} | "
                 f"{e.get('ours_triton_ssm_step', float('nan')):.1e} | {e.get('official_triton_ssu', float('nan')):.1e} | {e.get('vllm_triton_ssu', float('nan')):.1e} |")
    L.append("")
    L.append("## Single-token state update, eager (ms)")
    L.append("")
    L.append("| B | E | ours Triton | official Triton | vLLM Triton |")
    L.append("|---|---|---|---|---|")
    for r in res["step_core"]:
        e = r["eager_ms"]
        L.append(f"| {r['batch']} | {r['d_inner']} | {fmt(e.get('ours_triton_ssm_step'))} | {fmt(e.get('official_triton_ssu'))} | {fmt(e.get('vllm_triton_ssu'))} |")
    L.append("")
    L.append("## Whole Mamba layer step: our fused unit vs a composed official pipeline (fp32, ms)")
    L.append("")
    L.append("Our unit is LayerNorm + in_proj + conv step + x_proj + dt_proj + scan step + gate + out_proj + residual in two Triton launches. The composed baseline is PyTorch ops for everything except the state update, which uses the official or vLLM Triton kernel. This compares granularity, not just the scan.")
    L.append("")
    L.append("| B | E | ours GPU | composed+official GPU | composed+vLLM GPU | ours eager | composed+official eager | composed+vLLM eager | composed-official/ours GPU | ours GB/s | max abs diff |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["step_fused"]:
        g, e = r["graph_ms"], r["eager_ms"]
        o, c = g.get("composed_torch_plus_official_triton_ssu"), g.get("ours_fused_layer")
        d = r.get("max_abs_diff_vs_composed_official", {}).get("ours_fused_layer")
        L.append(f"| {r['batch']} | {r['d_inner']} | {fmt(c)} | {fmt(o)} | {fmt(g.get('composed_torch_plus_vllm_triton_ssu'))} | {fmt(e.get('ours_fused_layer'))} | "
                 f"{fmt(e.get('composed_torch_plus_official_triton_ssu'))} | {fmt(e.get('composed_torch_plus_vllm_triton_ssu'))} | "
                 f"{fmt(o / c if o and c else None, 2)} | {fmt(r['graph_bw_GBps'].get('ours_fused_layer'), 1)} | {fmt(d, 6) if d is not None else '-'} |")
    L.append("")
    if res["errors"]:
        L.append("## Errors recorded during the run")
        L.append("")
        for e in res["errors"]:
            L.append(f"- {e}")
    Path(path).write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tune-iters", type=int, default=15)
    ap.add_argument("--quick-tune", action="store_true")
    ap.add_argument("--only", default="scan,core,fused")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--wait-idle", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--fixed-cfg", default=None)
    ap.add_argument("--idle-per-row", type=int, default=0)
    ap.add_argument("--cold-l2", action="store_true")
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--out", default=str(ROOT / "results" / "official_compare.json"))
    ap.add_argument("--md", default=str(ROOT / "results" / "official_compare.md"))
    a = ap.parse_args()

    import vllm
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn as vllm_scan
    from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update as vllm_ssu
    ops = {"vllm_scan": vllm_scan, "vllm_ssu": vllm_ssu, "official_ssu": load_official_triton()}
    errors = []
    try:
        import selective_scan_cuda as osc
        ops["official_scan"] = lambda u, d, A, Bo, Co, D, z, db, sp: osc.fwd(u, d, A, Bo, Co, D, z, db, sp)
    except Exception as e:
        errors.append(f"official selective_scan_cuda import failed: {e}")
    commit = subprocess.run(["git", "-C", str(OFFICIAL_REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

    idle = wait_idle(a.wait_idle) if a.wait_idle else False
    meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, vllm=vllm.__version__, official_commit=commit,
                iters=a.iters, idle_gate_met=idle, smi_start=smi(), spec_bw_GBps=SPEC_BW / 1e9, argv=sys.argv[1:])
    res = dict(meta=meta, scan=[], step_core=[], step_fused=[], errors=errors)

    only = set(a.only.split(","))
    if "scan" in only:
        Bs, Es, Ts = ((1,), (1024,), (30, 300)) if a.small else ((1, 4), (1024, 2048), (30, 300, 3000))
        for dtype in [{"fp32": torch.float32, "bf16": torch.bfloat16}[d] for d in a.dtypes.split(",")]:
            for B in Bs:
                for E in Es:
                    for T in Ts:
                        bench_scan(B, E, T, dtype, a, ops, res["scan"], errors)
    if "core" in only:
        for E in ((1024,) if a.small else (1024, 2048)):
            for B in ((1, 8) if a.small else (1, 8, 64)):
                bench_step_core(B, E, a, ops, res["step_core"], errors)
    if "fused" in only:
        cache = {}
        for E in ((1024,) if a.small else (1024, 2048)):
            for B in ((1, 8) if a.small else (1, 8, 64)):
                bench_step_fused(B, E, a, ops, res["step_fused"], errors, cache)
    meta["smi_end"] = smi()
    if a.merge and Path(a.out).exists():
        old = json.loads(Path(a.out).read_text())
        scan_meta = dict(meta)
        meta = old["meta"]
        meta["scan_v2"] = scan_meta
        for k in ("step_core", "step_fused"):
            if not res[k]:
                res[k] = old[k]
        res["meta"] = meta
        res["errors"] = old["errors"] + ["scan section re-run (ours_cuda added): " + e for e in errors]
    Path(a.out).write_text(json.dumps(res, indent=1))
    write_markdown(res, a.md)
    print("wrote", a.out, a.md)


if __name__ == "__main__":
    main()
