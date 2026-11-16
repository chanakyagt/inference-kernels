import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
import qlib
import qnet
import qops

BF16 = torch.bfloat16
SIZE = [224]


@torch.no_grad()
def eval_logits(net, x_u8, bs=64):
    n = len(x_u8)
    outs = []
    for i in range(0, n, bs):
        chunk = x_u8[i:i + bs]
        k = len(chunk)
        if k < bs:
            chunk = torch.cat([chunk, chunk[-1:].repeat(bs - k, 1, 1, 1)])
        outs.append(net(qlib.normalize(chunk, BF16)).float().cpu()[:k])
    return torch.cat(outs)


def build_items(speed, sens, B):
    items = []
    for name, d in speed["gemm"].items():
        v = d[str(B)]["variants"]
        tb_name = min(("bf16_cublas", "bf16_triton"), key=lambda k: v[k]["graph_ms"])
        tb = v[tb_name]["graph_ms"]
        i8_name = min(("int8_triton_bf16in", "int_mm_pipeline"), key=lambda k: v[k]["graph_ms"] if "graph_ms" in v.get(k, {}) else 1e9)
        opts = [("int8", v[i8_name]["graph_ms"], "W8A8", "tri" if i8_name == "int8_triton_bf16in" else "mm")]
        if "graph_ms" in v.get("fp8_pipeline", {}):
            opts.append(("fp8", v["fp8_pipeline"]["graph_ms"], "FP8", "tri"))
        for mode, tq, sch, impl in opts:
            gain = tb - tq
            if gain > 0:
                items.append({"layer": name, "mode": mode, "gain_ms": gain, "kl": sens["sub"][f"{name}|{sch}"]["kl"], "bf16_impl": "tri" if tb_name == "bf16_triton" else "cublas", "int8_impl": impl})
    for name, d in speed["dw"].items():
        v = d[str(B)]["variants"]
        tb_name = min(("cudnn_bf16_then_silu", "triton_bf16_nhwc_silu"), key=lambda k: v[k]["graph_ms"])
        gain = v[tb_name]["graph_ms"] - v["quant_pass_plus_int8_silu"]["graph_ms"]
        chain = v[tb_name]["graph_ms"] - v["triton_int8_nhwc_silu"]["graph_ms"]
        items.append({"layer": name, "mode": "int8", "gain_ms": max(gain, chain * 0.5), "kl": sens["sub"][f"{name}|W8A8pc"]["kl"],
                      "bf16_impl": "tri" if tb_name == "triton_bf16_nhwc_silu" else "cublas", "int8_impl": "tri", "chain_gain_ms": chain, "alone_gain_ms": gain})
    items = [i for i in items if i["gain_ms"] > 0]
    best = {}
    for it in items:
        it["ratio"] = it["gain_ms"] / (it["kl"] + 1e-8)
        if it["layer"] not in best or it["ratio"] > best[it["layer"]]["ratio"]:
            best[it["layer"]] = it
    return sorted(best.values(), key=lambda i: -i["ratio"])


def impl_tables(speed, B):
    b16, i8 = {}, {}
    for name, d in speed["gemm"].items():
        v = d[str(B)]["variants"]
        b16[name] = "tri" if v["bf16_triton"]["graph_ms"] < v["bf16_cublas"]["graph_ms"] else "cublas"
    for name, d in speed["dw"].items():
        v = d[str(B)]["variants"]
        b16[name] = "tri" if v["triton_bf16_nhwc_silu"]["graph_ms"] < v["cudnn_bf16_then_silu"]["graph_ms"] else "cublas"
    return b16


def make_cfg(items, k):
    cfg, i8 = {}, {}
    for it in items[:k]:
        cfg[it["layer"]] = it["mode"]
        i8[it["layer"]] = it["int8_impl"]
    return cfg, i8


def build_net(folded, amax, cfg, i8, b16, amax_pc):
    return qnet.QNet(folded, amax, cfg, b16, i8, amax_pc).eval()


def graph_ms(net, B, replays=30):
    x = torch.randn(B, 3, SIZE[0], SIZE[0], device="cuda", dtype=BF16)
    with torch.no_grad():
        for _ in range(3):
            net(x)
        return qops.time_graph(lambda: net(x), reps=1, replays=replays)


def eager_ms(net, B):
    x = torch.randn(B, 3, SIZE[0], SIZE[0], device="cuda", dtype=BF16)
    with torch.no_grad():
        return qops.time_eager(lambda: net(x), iters=40)


def knee(points):
    lat = [p["lat_ms"] for p in points]
    kl = [p["kl"] for p in points]
    l0, lmin = lat[0], min(lat)
    k0, kmax = kl[0], max(kl)
    best, bi = -9, 0
    for i, p in enumerate(points):
        x = (l0 - p["lat_ms"]) / max(l0 - lmin, 1e-9)
        y = (p["kl"] - k0) / max(kmax - k0, 1e-12)
        if x - y > best:
            best, bi = x - y, i
    return bi


class PowerSampler:
    def __init__(self):
        self.samples = []
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "100"],
                                     stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()
        return self

    def loop(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.time(), float(line.strip())))
            except ValueError:
                pass

    def __exit__(self, *a):
        self.proc.terminate()
        self.t.join(timeout=2)


def energy(net, B, seconds=20, idle_seconds=10):
    x = torch.randn(B, 3, SIZE[0], SIZE[0], device="cuda", dtype=BF16)
    with torch.no_grad():
        for _ in range(3):
            net(x)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            net(x)
    torch.cuda.synchronize()
    with PowerSampler() as ps:
        time.sleep(idle_seconds)
        t_idle_end = time.time()
        n = 0
        t0 = time.time()
        while time.time() - t0 < seconds:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
            n += 10
        t1 = time.time()
    idle = [p for t, p in ps.samples if t < t_idle_end - 1.0]
    act = [p for t, p in ps.samples if t0 + 1.0 <= t <= t1]
    imgs = n * B
    ip = sum(idle) / len(idle)
    ap = sum(act) / len(act)
    return {"B": B, "images": imgs, "seconds": t1 - t0, "images_per_s": imgs / (t1 - t0), "idle_W": ip, "active_W": ap,
            "n_idle_samples": len(idle), "n_active_samples": len(act),
            "mJ_per_image_idle_subtracted": (ap - ip) * (t1 - t0) / imgs * 1e3, "mJ_per_image_total": ap * (t1 - t0) / imgs * 1e3}


def model_bytes(folded, cfg):
    total = {"fp32": 0, "bf16": 0, "mix": 0, "all_int8": 0, "all_w4g64": 0}
    names = {n for n, _ in qlib.quant_layers(folded)}
    for n, mod in qlib.quant_layers(folded):
        w = mod.weight
        b = mod.bias.numel() if mod.bias is not None else 0
        kind = qlib.layer_kind(n, mod)
        total["fp32"] += (w.numel() + b) * 4
        total["bf16"] += (w.numel() + b) * 2
        elig = qnet.eligible(n, kind, w.shape[1] if w.dim() > 1 else 1, w.shape[0], "int8")
        if elig:
            total["all_int8"] += w.numel() + 4 * w.shape[0] + 4 * b
            total["all_w4g64"] += w.numel() // 2 + 2 * (w.numel() // 64 + 1) + 4 * b
        else:
            total["all_int8"] += (w.numel() + b) * 2
            total["all_w4g64"] += (w.numel() + b) * 2
        m = cfg.get(n, "bf16")
        if m == "int8":
            total["mix"] += w.numel() + 4 * w.shape[0] + 4 * b
        elif m == "fp8":
            total["mix"] += w.numel() + 4 + 2 * b
        else:
            total["mix"] += (w.numel() + b) * 2
    other_n = sum(p.numel() for n, p in folded.named_parameters() if n.rsplit(".", 1)[0] not in names)
    fp32_kept = sum(p.numel() for n, p in folded.head.layers.named_parameters() if n.endswith(("A_log", "D"))) if folded.temporal else 0
    for k, bytes_per in (("fp32", 4), ("bf16", 2), ("mix", 2), ("all_int8", 2), ("all_w4g64", 2)):
        total[k] += other_n * bytes_per + (fp32_kept * 2 if k != "fp32" else 0)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="diloco")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--speed-tag", default=None)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sens = json.loads((qlib.OUT / f"sensitivity_{a.tag}.json").read_text())
    speed = json.loads((qlib.OUT / f"layer_speed_{a.speed_tag or a.tag}.json").read_text())
    idle = None if a.no_wait else qops.wait_idle()
    before = qops.gpu_state()
    size = qlib.default_size(a.ckpt)
    SIZE[0] = size
    xt, yt = qlib.load_data("test", size)
    folded = qlib.build_model(a.ckpt)
    amax = sens["meta"]["amax"]
    amax_pc = sens["meta"]["amax_pc"]
    idx = qlib.subset_idx(yt, sens["meta"]["n_sub"])
    xs, ys = xt[idx], yt[idx]
    ref_full = torch.load(qlib.CACHE / f"ref_full_{a.tag}.pt")
    ref_sub = ref_full[idx]
    out_path = qlib.OUT / f"mixed_plan_{a.tag}.json"
    out = {"provenance": {"idle_samples": idle, "gpu_before": before, "torch": torch.__version__}, "batches": {}}
    stock = qnet.stock_bf16(folded)
    for B in (1, 64):
        if a.only and str(B) not in a.only.split(","):
            continue
        items = build_items(speed, sens, B)
        b16 = impl_tables(speed, B)
        n_steps = len(items) if not a.max_steps else min(a.max_steps, len(items))
        pts = []
        for k in range(n_steps + 1):
            cfg, i8 = make_cfg(items, k)
            net = build_net(folded, amax, cfg, i8, b16, amax_pc)
            lat = graph_ms(net, B)
            lg = eval_logits(net, xs)
            m = qlib.metrics(lg, ys, ref_sub)
            pts.append({"k": k, "added": None if k == 0 else f"{items[k - 1]['layer']}:{items[k - 1]['mode']}", "lat_ms": lat,
                        "acc": m["acc"], "auc": m["auc"], "kl": m["kl"], "flip": m["flip"]})
            print(B, k, pts[-1], flush=True)
            del net
            torch.cuda.empty_cache()
        ki = knee(pts)
        out["batches"][str(B)] = {"items": items, "points": pts, "knee_index": ki}
        out["provenance"]["gpu_mid"] = qops.gpu_state()
        out_path.write_text(json.dumps(out))
    final = {}
    for B_s, d in out["batches"].items():
        B = int(B_s)
        ki = d["knee_index"]
        items = d["items"]
        b16 = impl_tables(speed, B)
        cfg, i8 = make_cfg(items, ki)
        variants = {"stock_bf16_folded": None, "qnet_bf16": ({}, {}), "chosen_mix": (cfg, i8),
                    "all_int8_eligible": ({n: "int8" for n in list(speed["gemm"]) + list(speed["dw"])}, {})}
        res = {}
        for vname, spec in variants.items():
            net = stock if spec is None else build_net(folded, amax, spec[0], spec[1], b16, amax_pc)
            if spec is None:
                net_call = lambda x, n=net: n(x.contiguous(memory_format=torch.channels_last))
                class W(torch.nn.Module):
                    def __init__(s, n):
                        super().__init__()
                        s.n = n

                    def forward(s, x):
                        return s.n(x.contiguous(memory_format=torch.channels_last)).float()
                net = W(net).eval()
            lg = eval_logits(net, xt)
            m = qlib.metrics(lg, yt, ref_full)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            xx = torch.randn(B, 3, SIZE[0], SIZE[0], device="cuda", dtype=BF16)
            with torch.no_grad():
                net(xx)
            torch.cuda.synchronize()
            res[vname] = {"test_acc": m["acc"], "test_auc": m["auc"], "kl_vs_fp32": m["kl"], "flip_vs_fp32": m["flip"],
                          "graph_ms": graph_ms(net, B, 50), "eager_ms": eager_ms(net, B),
                          "weights_resident_MB": base / 1e6, "peak_MB_during_forward": torch.cuda.max_memory_allocated() / 1e6}
            res[vname]["energy"] = energy(net, B)
            print(B, vname, res[vname], flush=True)
            del net
            torch.cuda.empty_cache()
        final[B_s] = {"config": cfg, "int8_impl": i8, "n_quantized": len(cfg), "variants": res}
        out["final"] = final
        out_path.write_text(json.dumps(out))
    out["model_bytes"] = model_bytes(folded, final[str(max(int(b) for b in final))]["config"]) if final else None
    out["model_bytes_by_batch"] = {b: model_bytes(folded, f["config"]) for b, f in final.items()}
    out["provenance"]["gpu_after"] = qops.gpu_state()
    out["fp32_ref_full"] = sens["meta"]["ref_full"]
    out_path.write_text(json.dumps(out))
    write_md(out, a.tag)


def write_md(out, tag):
    L = [f"# Mixed-precision plan ({tag})", "", f"Provenance: {json.dumps(out['provenance'])}", ""]
    for B, d in out["batches"].items():
        L += [f"## Pareto curve, batch {B}", "",
              "Layers are added in order of (measured latency gain) / (single-layer KL cost). Every row is a real run with the real kernels: latency is a CUDA-graph replay, accuracy/AUC/KL are on the 2,000-image Test subset (KL against the FP32 model). Knee marked with ***.", "",
              "| Step | Layer added | Latency ms | Accuracy | AUC | KL vs fp32 | Prediction flips |", "|---|---|---|---|---|---|---|"]
        for p in d["points"]:
            mark = " ***" if p["k"] == d["knee_index"] else ""
            L.append(f"| {p['k']}{mark} | {p['added'] or 'none (all bf16)'} | {p['lat_ms']:.3f} | {p['acc']:.4f} | {p['auc']:.4f} | {p['kl']:.2e} | {p['flip']:.4f} |")
        L.append("")
    if "final" in out:
        L += ["## Chosen configuration, full Test set (10,905 images)", ""]
        for B, f in out["final"].items():
            L += [f"### Batch {B}: {f['n_quantized']} layers quantized", "", "| Variant | Test accuracy | Test AUC | KL vs fp32 | Graph ms | Eager ms | Images/s (20 s) | Idle W | Active W | mJ/image (idle-subtracted) | Weights MB | Peak MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for vn, r in f["variants"].items():
                e = r["energy"]
                L.append(f"| {vn} | {r['test_acc']:.4f} | {r['test_auc']:.4f} | {r['kl_vs_fp32']:.2e} | {r['graph_ms']:.3f} | {r['eager_ms']:.3f} | {e['images_per_s']:.0f} | {e['idle_W']:.1f} | {e['active_W']:.1f} | {e['mJ_per_image_idle_subtracted']:.2f} | {r['weights_resident_MB']:.1f} | {r['peak_MB_during_forward']:.0f} |")
            L += ["", "Quantized layers: " + ", ".join(f"{k}={v}" for k, v in f["config"].items()), ""]
        L += ["## Model size", "", "| Batch plan | fp32 MB | bf16 MB | chosen mix MB | all-int8 MB | all-W4 group-64 MB |", "|---|---|---|---|---|---|"]
        for b, mb in out["model_bytes_by_batch"].items():
            L.append(f"| {b} | {mb['fp32'] / 1e6:.2f} | {mb['bf16'] / 1e6:.2f} | {mb['mix'] / 1e6:.2f} | {mb['all_int8'] / 1e6:.2f} | {mb['all_w4g64'] / 1e6:.2f} |")
    (qlib.OUT / f"mixed_plan_{tag}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
