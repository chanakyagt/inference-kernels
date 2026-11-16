import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcommon import (size_of, interp_of, PARTS, ROOT, Net, bench, fold_conv_bn, gpu_state, idle_check, load_cache, load_model, metrics,
                     prep, sustained, find_finetuned, FROZEN_CKPT, CACHE)

from fused_dwconv_bn_silu import patch_model as patch_dw_nchw
from fused_dwconv_nhwc import patch_model as patch_dw_nhwc
from selective_scan import patch_model as patch_scan
from fp8_int8_linear import FP8Linear, Int8WeightOnlyLinear, swap_linears

RUNGS = {
    "R0s": dict(name="fp32 eager NCHW, TF32 off", strict=True),
    "R0": dict(name="fp32 eager NCHW"),
    "R1": dict(name="+ bf16 autocast", ac=True),
    "R2": dict(name="+ channels_last", ac=True, cl=True),
    "R3": dict(name="+ BN folded into conv", ac=True, cl=True, fold=True),
    "R4": dict(name="+ fused dwconv+BN+SiLU (repo, NCHW)", ac=True, cl=True, fold=True, dw="nchw"),
    "R4b": dict(name="+ fused dwconv+BN+SiLU (NHWC variant)", ac=True, cl=True, fold=True, dw="nhwc"),
    "R5": dict(name="+ fused Triton scan", ac=True, cl=True, fold=True, dw="DW", scan=True),
    "R6": dict(name="+ CUDA graph", ac=True, cl=True, fold=True, dw="DW", scan=True, graph=True),
    "R6b": dict(name="R6 with bf16 weights pre-cast (no autocast)", cl=True, fold=True, dw="DW", scan=True, graph=True, precast=True),
    "R6c": dict(name="R3 + fused scan + CUDA graph (no Triton dwconv)", ac=True, cl=True, fold=True, scan=True, graph=True),
    "R6d": dict(name="R6c with bf16 weights pre-cast (no autocast)", cl=True, fold=True, scan=True, graph=True, precast=True),
    "R6f": dict(name="R6b in fp16 instead of bf16", cl=True, fold=True, dw="DW", scan=True, graph=True, precast=True,
                dt=torch.float16),
    "R6g": dict(name="R6d in fp16 instead of bf16 (no Triton dwconv)", cl=True, fold=True, scan=True, graph=True, precast=True,
                dt=torch.float16),
    "R7": dict(name="torch.compile max-autotune on R2", ac=True, cl=True, compile=True),
    "R8": dict(name="R6c + INT8 weight-only head Linears", ac=True, cl=True, fold=True, scan=True, graph=True, q="int8"),
    "R9": dict(name="R6c + FP8 head Linears", ac=True, cl=True, fold=True, scan=True, graph=True, q="fp8"),
}
ORDER = ["R0s", "R0", "R1", "R2", "R3", "R4", "R4b", "R5", "R6", "R6b", "R6c", "R6d", "R7", "R8", "R9"]


def resolve_dw(which, dw_best):
    if dw_best != "auto":
        return dw_best
    tag = Path(which).name if which != "frozen" else "frozen"
    lat = {}
    for k, v in (("R4", "nchw"), ("R4b", "nhwc")):
        f = PARTS / f"{tag}_{k}_perf.json"
        if f.exists():
            lat[v] = json.loads(f.read_text())["latency_ms"]["64"]
    return min(lat, key=lat.get) if lat else "nhwc"


class Runner:
    def __init__(self, which, cfg, bs, dw_best="auto"):
        torch.backends.cudnn.benchmark = True
        if cfg.get("strict"):
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        net, self.interp = load_model(which)
        self.cfg, self.bs = cfg, bs
        self.size = size_of(which)
        self.meta = {}
        dw = cfg.get("dw")
        if dw == "DW":
            dw = resolve_dw(which, dw_best)
        if dw:
            self.meta["dw_blocks_fused"] = (patch_dw_nchw if dw == "nchw" else patch_dw_nhwc)(net.trunk)
            self.meta["dw_variant"] = dw
        if cfg.get("fold"):
            self.meta["conv_bn_folded"] = fold_conv_bn(net.trunk)
        if cfg.get("scan"):
            self.meta["scans_fused"] = patch_scan(net.head) if net.kind == "mamba" else 0
        if cfg.get("q"):
            cls = Int8WeightOnlyLinear if cfg["q"] == "int8" else FP8Linear
            self.meta["head_linears_quantized"] = swap_linears(net.head, cls, lambda n: True)
            self.meta["head_linears_total"] = sum(isinstance(m, nn.Linear) for m in net.head.modules()) + self.meta["head_linears_quantized"]
        if cfg.get("cl"):
            net = net.to(memory_format=torch.channels_last)
        if cfg.get("precast"):
            net = net.to(cfg.get("dt", torch.bfloat16))
            if net.kind == "mamba":
                for layer in net.head.layers:
                    layer.mixer.A_log.data = layer.mixer.A_log.data.float()
                    layer.mixer.D.data = layer.mixer.D.data.float()
        self.net = net
        self.error = None
        self.fwd = self._make_fwd()
        if cfg.get("compile"):
            self.fwd = torch.compile(self.fwd, mode="max-autotune", dynamic=False)
        self.graph = None
        with torch.no_grad():
            if cfg.get("graph"):
                self._capture()
            else:
                x = torch.randn(bs, 3, self.size, self.size, device="cuda")
                for _ in range(3):
                    self.fwd(x)
                torch.cuda.synchronize()

    def _make_fwd(self):
        cfg, net = self.cfg, self.net

        def fwd(x):
            if cfg.get("cl"):
                x = x.contiguous(memory_format=torch.channels_last)
            if cfg.get("precast"):
                x = x.to(cfg.get("dt", torch.bfloat16))
            with torch.autocast("cuda", cfg.get("dt", torch.bfloat16), enabled=bool(cfg.get("ac")), cache_enabled=not cfg.get("graph")):
                return net(x).float()
        return fwd

    def _capture(self):
        self.static_in = torch.randn(self.bs, 3, self.size, self.size, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.fwd(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = self.fwd(self.static_in)

    @torch.no_grad()
    def run(self, x):
        if self.graph is not None:
            self.static_in.copy_(x)
            self.graph.replay()
            return self.static_out
        return self.fwd(x)


@torch.no_grad()
def predict(runner, u8, ab):
    n = len(u8)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, ab):
        x = prep(u8[i:i + ab])
        m = x.shape[0]
        if m < ab:
            x = torch.cat([x, x[-1:].expand(ab - m, -1, -1, -1)])
        out[i:i + m] = runner.run(x).softmax(1)[:m, 1].double().cpu().numpy()
    return out


def acc_phase(which, key, cfg, args):
    u8, y = load_cache(interp_of(which), args.limit, size_of(which))
    r = Runner(which, cfg, args.acc_bs, args.dw)
    t0 = time.time()
    p = predict(r, u8, args.acc_bs)
    res = {"rung": key, "model": which, "name": cfg["name"], "meta": r.meta, "eval_s": time.time() - t0, **metrics(y, p)}
    tag = Path(which).name if which != "frozen" else "frozen"
    ref = CACHE / f"probs_{tag}_R0s_{len(y)}.npy"
    if key == "R0s":
        np.save(ref, p)
    elif ref.exists():
        pr = np.load(ref)
        res["pred_flips_vs_R0s"] = int(((pr > 0.5) != (p > 0.5)).sum())
        res["max_abs_dprob_vs_R0s"] = float(np.abs(pr - p).max())
        res["mean_abs_dprob_vs_R0s"] = float(np.abs(pr - p).mean())
    return res


def perf_phase(which, key, cfg, args, u8):
    res = {"rung": key, "model": which, "name": cfg["name"], "latency_ms": {}, "peak_mem_mb": {}, "sustained": {}}
    res["gpu_state_start"] = gpu_state()
    for bs in (1, 8, 64):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            r = Runner(which, cfg, bs, args.dw)
        except Exception as e:
            res["failed"] = f"{type(e).__name__}: {str(e)[:400]}"
            res["traceback"] = traceback.format_exc()[-1500:]
            return res
        res["meta"] = r.meta
        x = prep(u8[:bs])
        iters = {1: 200, 8: 100, 64: 50}[bs]
        trials = [bench(lambda: r.run(x), iters=iters, warmup=10) for _ in range(3)]
        res["latency_ms"][bs] = min(trials)
        res.setdefault("latency_trials_ms", {})[bs] = trials
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            r.run(x)
        torch.cuda.synchronize()
        res["peak_mem_mb"][bs] = torch.cuda.max_memory_allocated() / 2**20
        best = None
        attempts = []
        for att in range(3):
            cand = sustained(lambda z: r.run(z), x, args.seconds, tag=f"{key}_{bs}")
            attempts.append({k: cand[k] for k in ("img_per_s", "stable", "idle_clean", "throughput_cv", "mean_power_w", "idle_power_w")})
            ok = cand["stable"] and cand["idle_clean"]
            if best is None or ok or cand["img_per_s"] > best["img_per_s"] and not (best["stable"] and best["idle_clean"]):
                best = cand
            if ok:
                break
        best["attempts"] = attempts
        res["sustained"][bs] = best
        res["gpu_state_after_bs%d" % bs] = gpu_state()
        del r
        torch.cuda.empty_cache()
    return res


def part_path(which, key, phase):
    tag = Path(which).name if which != "frozen" else "frozen"
    PARTS.mkdir(parents=True, exist_ok=True)
    return PARTS / f"{tag}_{key}_{phase}.json"


def cmd_rung(args):
    which = args.model
    for key in args.rungs.split(","):
        cfg = RUNGS[key]
        try:
            if args.phase in ("acc", "both") and not (args.skip_done and part_path(which, key, "acc").exists()):
                res = acc_phase(which, key, cfg, args)
                part_path(which, key, "acc").write_text(json.dumps(res, indent=1))
                print(json.dumps({k: res[k] for k in ("rung", "accuracy", "auc", "eval_s")}), flush=True)
        except Exception as e:
            part_path(which, key, "acc").write_text(json.dumps({"rung": key, "failed": f"{type(e).__name__}: {str(e)[:400]}",
                                                                 "traceback": traceback.format_exc()[-1500:]}, indent=1))
            print(key, "acc FAILED", e, flush=True)
        if args.phase in ("perf", "both") and not (args.skip_done and part_path(which, key, "perf").exists()):
            chk = None if args.no_idle_check else idle_check(args.idle_timeout)
            if chk is not None and not chk["passed"]:
                print(key, "GPU not idle; skipping perf", chk, flush=True)
                continue
            u8, _ = load_cache(interp_of(which), 64, size_of(which))
            res = perf_phase(which, key, cfg, args, u8)
            res["idle_check"] = chk
            part_path(which, key, "perf").write_text(json.dumps(res, indent=1))
            print(key, json.dumps({"lat": res.get("latency_ms"), "fail": res.get("failed")}), flush=True)


class FileDS(torch.utils.data.Dataset):
    def __init__(self, names, mode, interp, size):
        self.names, self.mode, self.interp, self.size = names, mode, interp, size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.mode == "bytes":
            with open(self.names[i], "rb") as f:
                return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8)
        from PIL import Image
        res = Image.BILINEAR if self.interp == "bilinear" else Image.BICUBIC
        return torch.from_numpy(np.asarray(Image.open(self.names[i]).convert("RGB").resize((self.size, self.size), res), dtype=np.uint8).copy())


def gpu_decode_prep(blobs, interp, size):
    from torchvision.io import ImageReadMode, decode_jpeg
    imgs = decode_jpeg(blobs, device="cuda", mode=ImageReadMode.RGB)
    x = torch.stack([i.float() if i.shape[-1] == size and i.shape[-2] == size else
                     F.interpolate(i[None].float(), size=(size, size), mode=interp, antialias=True, align_corners=False)[0]
                     for i in imgs]).div_(255)
    from tcommon import MEAN, STD
    return ((x - MEAN.cuda()) / STD.cuda()).contiguous()


def pad(x, ab):
    m = x.shape[0]
    return x if m == ab else torch.cat([x, x[-1:].expand(ab - m, -1, -1, -1)])


@torch.no_grad()
def e2e_pass(kind, runner, names, interp, workers, ab, size, with_model=True):
    ds = FileDS(names, "bytes" if kind == "gpu" else "pil", interp, size)
    dl = torch.utils.data.DataLoader(ds, batch_size=ab, num_workers=workers, pin_memory=(kind == "cpu"),
                                     collate_fn=(lambda b: b) if kind == "gpu" else None)
    probs = []
    torch.cuda.synchronize()
    t0 = time.time()
    for b in dl:
        if kind == "gpu":
            x = gpu_decode_prep(b, interp, size)
        else:
            x = prep(b.numpy())
        m = x.shape[0]
        if with_model:
            probs.append(runner.run(pad(x, ab)).softmax(1)[:m, 1].double().cpu())
    torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, (torch.cat(probs).numpy() if probs else None)


@torch.no_grad()
def cmd_e2e(args):
    which = args.model
    cfg = RUNGS[args.e2e_rung]
    chk = None if args.no_idle_check else idle_check(args.idle_timeout)
    if chk is not None and not chk["passed"]:
        print("GPU not idle", chk)
        return
    interp = interp_of(which)
    SZ = size_of(which)
    names = json.loads((CACHE / "names.json").read_text())[:args.limit or None]
    y = np.load(CACHE / "labels.npy")[:len(names)] if not args.limit else None
    _, ycache = load_cache(interp, args.limit, size_of(which))
    y = ycache
    r = Runner(which, cfg, args.acc_bs, args.dw)
    res = {"model": which, "rung": args.e2e_rung, "images": len(names), "workers": args.workers, "idle_check": chk,
           "gpu_state_start": gpu_state(), "pipelines": {}}
    u8 = load_cache(interp, 64, size_of(which))[0]
    x = prep(u8[:args.acc_bs])
    t0 = time.time()
    n = 0
    while time.time() - t0 < 10:
        for _ in range(8):
            r.run(x)
            n += args.acc_bs
        torch.cuda.synchronize()
    res["model_only_img_per_s"] = n / (time.time() - t0)
    for kind in ("cpu", "gpu"):
        e2e_pass(kind, r, names[:256], interp, args.workers, args.acc_bs, SZ)
        full, dec = [], []
        probs = None
        for _ in range(args.repeats):
            dt, probs = e2e_pass(kind, r, names, interp, args.workers, args.acc_bs, SZ)
            full.append(len(names) / dt)
            dt2, _ = e2e_pass(kind, r, names, interp, args.workers, args.acc_bs, SZ, with_model=False)
            dec.append(len(names) / dt2)
        res["pipelines"][kind] = {"end_to_end_img_per_s": full, "end_to_end_median": statistics.median(full),
                                  "decode_prep_only_img_per_s": dec, "decode_prep_only_median": statistics.median(dec),
                                  **metrics(y, probs)}
        print(kind, res["pipelines"][kind]["end_to_end_median"], res["pipelines"][kind]["decode_prep_only_median"], flush=True)
    res["gpu_state_end"] = gpu_state()
    tag = Path(which).name if which != "frozen" else "frozen"
    (PARTS / f"{tag}_R10_e2e.json").write_text(json.dumps(res, indent=1))


def cmd_merge(args):
    import merge_results
    merge_results.main()


def cmd_ref(args):
    from tcommon import FrameEncoder, temporal_from_cfg
    ck = torch.load(FROZEN_CKPT, map_location="cpu")
    enc = FrameEncoder(pretrained=True).cuda().to(memory_format=torch.channels_last)
    head = temporal_from_cfg(ck["config"])
    head.load_state_dict(ck["model"])
    head = head.cuda().eval()
    for l in head.layers:
        l.mixer.scan_impl = "ref"
    u8, y = load_cache("bilinear", args.limit)
    ps = []
    with torch.no_grad():
        for i in range(0, len(u8), 256):
            x = prep(u8[i:i + 256]).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", torch.bfloat16):
                f = enc(x[:, None])
            f = f.half().float()
            ps.append(head(f).softmax(1)[:, 1].double().cpu())
    p = torch.cat(ps).numpy()
    res = {"rung": "R0ref", "name": "original extraction pipeline (bf16 trunk, fp16 features, fp32 head)", **metrics(y, p)}
    part_path("frozen", "R0ref", "acc").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rung", "e2e", "merge", "ref"])
    ap.add_argument("--model", default="frozen")
    ap.add_argument("--rungs", default=",".join(ORDER))
    ap.add_argument("--phase", default="both", choices=["acc", "perf", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--acc-bs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--dw", default="auto", choices=["nchw", "nhwc", "auto"])
    ap.add_argument("--no-idle-check", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--idle-timeout", type=float, default=1800)
    ap.add_argument("--e2e-rung", default="R6")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    {"rung": cmd_rung, "e2e": cmd_e2e, "merge": cmd_merge, "ref": cmd_ref}[args.mode](args)


if __name__ == "__main__":
    main()
