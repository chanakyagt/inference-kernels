"""Where does TensorRT fp16 beat the own fp16 path? Per-layer profile at batch 64.

TensorRT side: fp16 engine rebuilt with DETAILED profiling verbosity; IProfiler layer times; each
engine layer is mapped to PyTorch module paths through the ONNX node metadata (name_scopes).
Own side: R6f (fp16, BN folded, Triton NHWC depthwise, channels_last) run eagerly with CUDA
events around each sub-module of every MBConv block. Both are grouped by stage and by kind:
stem, expand 1x1, depthwise, squeeze-excite (SE), project 1x1, residual/other, head conv, pool+fc.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import ROOT, gpu_state, idle_check, interp_of, load_cache, load_model, prep, size_of
from trt_bench import ENG, LOG
from ladder import RUNGS, Runner
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

OUT = ROOT / "results" / "trt"
BS = 64
ITERS = 50


def classify(ref, path):
    """Map a module path like trunk.0.2.1.block.3 (or deeper) to (stage, kind)."""
    p = path.split(".")
    if p[:2] != ["trunk", "0"] or len(p) < 3:
        return ("pool+fc", "pool+fc") if path.startswith(("trunk", "head")) else ("other", "other")
    s = int(p[2])
    if s == 0:
        return ("stem", "stem")
    if s == 8:
        return ("head conv", "head conv")
    if len(p) < 6 or p[4] != "block":
        return (f"stage{s}", "residual/other")
    k = int(p[5])
    blk = ref.get_submodule(".".join(p[:5]))
    sub = blk[k]
    if isinstance(sub, SqueezeExcitation):
        return (f"stage{s}", "SE")
    if isinstance(sub, Conv2dNormActivation):
        conv = sub[0]
        if conv.groups > 1:
            return (f"stage{s}", "depthwise")
        return (f"stage{s}", "project 1x1" if k == len(blk) - 1 else "expand 1x1")
    return (f"stage{s}", "residual/other")


def onnx_node_paths(onnx_path):
    import onnx
    out = {}
    for n in onnx.load(str(onnx_path)).graph.node:
        md = {p.key: p.value for p in n.metadata_props}
        scopes = eval(md.get("pkg.torch.onnx.name_scopes", "[]"))
        mods = [s for s in scopes[1:-1] if s]
        out[n.name] = mods[-1] if mods else ""
    return out


class Prof(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.t = defaultdict(float)
        self.n = 0

    def report_layer_time(self, name, ms):
        self.t[name] += ms


def trt_profile(ref, u8):
    onnx_p = ENG / f"det_fp16_b{BS}.onnx"
    plan_p = ENG / f"det_fp16prof_b{BS}.plan"
    if not plan_p.exists():
        b = trt.Builder(LOG)
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(net, LOG)
        assert parser.parse(onnx_p.read_bytes())
        cfg = b.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        plan_p.write_bytes(bytes(b.build_serialized_network(net, cfg)))
    eng = trt.Runtime(LOG).deserialize_cuda_engine(plan_p.read_bytes())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    inp = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    outn = next(n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
    x = prep(u8[:BS]).half().contiguous()
    y = torch.empty(tuple(eng.get_tensor_shape(outn)), device="cuda", dtype=torch.float16)
    ctx.set_tensor_address(inp, x.data_ptr())
    ctx.set_tensor_address(outn, y.data_ptr())
    s = torch.cuda.Stream()
    for _ in range(10):
        ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    prof = Prof()
    ctx.profiler = prof
    for _ in range(ITERS):
        ctx.execute_async_v3(s.cuda_stream)
        s.synchronize()
    ctx.profiler = None
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    layers = {L["Name"]: L for L in info["Layers"]} if isinstance(info["Layers"][0], dict) else {}
    paths = onnx_node_paths(onnx_p)
    groups = defaultdict(float)
    rows = []
    for lname, tot in prof.t.items():
        ms = tot / ITERS
        L = layers.get(lname, {})
        text = lname + " " + str(L.get("Metadata", ""))
        nodes = re.findall(r"node_[A-Za-z0-9_]+", text)
        convs = [n for n in nodes if n.startswith("node_conv2d") or n.startswith("node_Conv")]
        key_node = (convs or nodes or [None])[0]
        if key_node is None or key_node not in paths:
            grp = ("reformat/other", "reformat/other")
        else:
            grp = classify(ref, paths[key_node])
        groups[grp] += ms
        rows.append({"layer": lname[:200], "ms": ms, "group": grp, "type": L.get("LayerType"), "tactic": L.get("TacticName")})
    return groups, rows


def own_profile(which, u8):
    cfg = dict(RUNGS["R6f"], graph=False)
    r = Runner(which, cfg, BS)
    net = r.net
    hooks, ev = [], defaultdict(list)
    targets = {}
    for name, m in net.named_modules():
        p = name.split(".")
        if name in ("trunk.0.0", "trunk.0.8", "trunk.1", "trunk.2", "head"):
            targets[name] = m
        elif len(p) == 6 and p[:2] == ["trunk", "0"] and p[4] == "block":
            targets[name] = m
        elif len(p) == 4 and p[:2] == ["trunk", "0"] and p[2] not in ("0", "8"):
            targets[name] = m  # whole MBConv block, for the residual remainder
    for name, m in targets.items():
        def pre(mod, inp, name=name):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[name].append([e, None])

        def post(mod, inp, out, name=name):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            ev[name][-1][1] = e
        hooks += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
    x = prep(u8[:BS])
    with torch.no_grad():
        for _ in range(10):
            r.run(x)
        torch.cuda.synchronize()
        ev.clear()
        s0, s1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s0.record()
        for _ in range(ITERS):
            r.run(x)
        s1.record()
        torch.cuda.synchronize()
    for h in hooks:
        h.remove()
    ms = {n: float(np.mean([a.elapsed_time(b) for a, b in v])) for n, v in ev.items()}
    ref, _ = load_model(which)
    groups = defaultdict(float)
    for name, t in ms.items():
        p = name.split(".")
        if len(p) == 4:  # whole block: residual add / stochastic depth = block - sum(children)
            kids = sum(ms.get(k, 0) for k in ms if k.startswith(name + ".block."))
            groups[(f"stage{p[2]}", "residual/other")] += max(t - kids, 0)
        else:
            groups[classify(ref, name)] += t
    return groups, ms, s0.elapsed_time(s1) / ITERS


def main():
    which = str(Path(__import__("os").environ.get("DFM_CKPT_ROOT", "/tmp")) / "bb_runs" / "ft_effnet_b0_v2")
    u8, _ = load_cache(interp_of(which), None, size_of(which))
    ref, _ = load_model(which)
    res = {"batch": BS, "iters": ITERS, "idle_check": idle_check(600), "gpu_state_start": gpu_state()}
    tg, trows = trt_profile(ref, u8)
    og, oms, own_total = own_profile(which, u8)
    keys = sorted(set(tg) | set(og), key=lambda k: (k[0], k[1]))
    table = []
    for k in keys:
        a, b = og.get(k, 0.0), tg.get(k, 0.0)
        table.append({"stage": k[0], "kind": k[1], "own_ms": a, "trt_ms": b, "own_over_trt": a / b if b > 1e-6 else None})
    kinds = defaultdict(lambda: [0.0, 0.0])
    for row in table:
        kinds[row["kind"]][0] += row["own_ms"]
        kinds[row["kind"]][1] += row["trt_ms"]
    res.update({"by_stage_kind": table,
                "by_kind": {k: {"own_ms": v[0], "trt_ms": v[1], "own_over_trt": v[0] / v[1] if v[1] else None} for k, v in kinds.items()},
                "own_total_ms_eager_hooks": own_total, "own_sum_ms": sum(og.values()), "trt_sum_ms": sum(tg.values()),
                "trt_layers": sorted(trows, key=lambda r: -r["ms"])[:60], "own_modules_ms": oms})
    (OUT / "layer_profile.json").write_text(json.dumps(res, indent=1, default=str))
    print("by kind:")
    for k, v in sorted(res["by_kind"].items(), key=lambda kv: -kv[1]["own_ms"]):
        print(f"  {k:16s} own {v['own_ms']:7.3f}  trt {v['trt_ms']:7.3f}  ratio {v['own_over_trt'] or float('nan'):.2f}")
    print("totals: own eager", round(own_total, 3), "own sum", round(res["own_sum_ms"], 3), "trt sum", round(res["trt_sum_ms"], 3))


if __name__ == "__main__":
    main()
