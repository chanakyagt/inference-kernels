import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="diloco")
    ap.add_argument("--n-sub", type=int, default=2000)
    ap.add_argument("--n-full", type=int, default=10)
    ap.add_argument("--bs", type=int, default=50)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    qlib.OUT.mkdir(parents=True, exist_ok=True)
    out_json = qlib.OUT / f"sensitivity_{a.tag}.json"
    state = json.loads(out_json.read_text()) if out_json.exists() else {}
    size = qlib.default_size(a.ckpt)
    qlib.build_cache(size)
    xt, yt = qlib.load_data("test", size)
    xc, _ = qlib.load_data("calib", size)
    model = qlib.build_model(a.ckpt)
    amax = qlib.calibrate(model, xc)
    amax_pc = qlib.calibrate_pc(model, xc)
    wraps = qlib.wrap_all(model, amax, amax_pc)
    idx = qlib.subset_idx(yt, a.n_sub)
    xs, ys = xt[idx], yt[idx]

    def run(x, bs=a.bs):
        return qlib.run_logits(model, x, bs)

    ref_sub = run(xs)
    state["meta"] = {"n_sub": len(idx), "calibration_images": len(xc), "amax": amax, "amax_pc": amax_pc, "ref_sub": qlib.metrics(ref_sub, ys),
                     "provenance": qlib.provenance(), "ckpt": a.ckpt or qlib.HEAD_CKPT}
    full_ref_path = qlib.CACHE / f"ref_full_{a.tag}.pt"
    if full_ref_path.exists():
        ref_full = torch.load(full_ref_path)
    else:
        ref_full = run(xt)
        torch.save(ref_full, full_ref_path)
    state["meta"]["ref_full"] = qlib.metrics(ref_full, yt)
    print("ref sub", state["meta"]["ref_sub"], "ref full", state["meta"]["ref_full"], flush=True)
    state.setdefault("sub", {})
    t0 = time.time()
    for name, w in wraps.items():
        for sch in qlib.SCHEMES + (qlib.DW_SCHEMES if name in amax_pc else []):
            key = f"{name}|{sch}"
            if key in state["sub"]:
                continue
            w.scheme = sch
            lg = run(xs)
            w.scheme = None
            m = qlib.metrics(lg, ys, ref_sub)
            m["d_acc"] = m["acc"] - state["meta"]["ref_sub"]["acc"]
            m["d_auc"] = m["auc"] - state["meta"]["ref_sub"]["auc"]
            state["sub"][key] = m
        out_json.write_text(json.dumps(state))
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)

    names = list(wraps)
    rank = {n: 0.0 for n in names}
    for sch in qlib.SCHEMES:
        order = sorted(names, key=lambda n: -state["sub"][f"{n}|{sch}"]["kl"])
        for r, n in enumerate(order):
            rank[n] += r / len(qlib.SCHEMES)
    state["mean_rank"] = rank
    top = sorted(names, key=lambda n: rank[n])[:a.n_full]
    rest = [n for n in names if n not in top]
    random.Random(0).shuffle(rest)
    rnd = rest[:a.n_full]
    state["full_sets"] = {"top": top, "random": rnd}
    state.setdefault("full", {})
    for group in (top, rnd):
        for name in group:
            for sch in qlib.SCHEMES:
                key = f"{name}|{sch}"
                if key in state["full"]:
                    continue
                wraps[name].scheme = sch
                lg = run(xt)
                wraps[name].scheme = None
                m = qlib.metrics(lg, yt, ref_full)
                m["d_acc"] = m["acc"] - state["meta"]["ref_full"]["acc"]
                m["d_auc"] = m["auc"] - state["meta"]["ref_full"]["auc"]
                state["full"][key] = m
            out_json.write_text(json.dumps(state))
            print(f"full {name} done {time.time() - t0:.0f}s", flush=True)
    for name in amax_pc:
        for sch in qlib.DW_SCHEMES:
            key = f"{name}|{sch}"
            if key in state["full"]:
                continue
            wraps[name].scheme = sch
            lg = run(xt)
            wraps[name].scheme = None
            m = qlib.metrics(lg, yt, ref_full)
            m["d_acc"] = m["acc"] - state["meta"]["ref_full"]["acc"]
            m["d_auc"] = m["auc"] - state["meta"]["ref_full"]["auc"]
            state["full"][key] = m
        out_json.write_text(json.dumps(state))
        print(f"full pc {name} done {time.time() - t0:.0f}s", flush=True)
    out_json.write_text(json.dumps(state))
    write_md(state, a.tag, {n: qlib.layer_kind(n, w.mod) for n, w in wraps.items()})


def write_md(state, tag, kinds):
    names = list(state["mean_rank"])
    order = sorted(names, key=lambda n: state["mean_rank"][n])
    sub = state["sub"]
    L = [f"# Per-layer sensitivity ({tag})", "",
         f"Each row quantizes exactly one layer (simulated in FP32) and leaves the rest untouched. KL and MSE compare the output with the FP32 model on {state['meta']['n_sub']} Test images.",
         "Lower is better. Rank = mean rank of the KL divergence over the four schemes (1 = most sensitive).", "",
         f"FP32 reference on the {state['meta']['n_sub']}-image subset: accuracy {state['meta']['ref_sub']['acc']:.4f}, AUC {state['meta']['ref_sub']['auc']:.4f}. "
         f"On the full Test set: accuracy {state['meta']['ref_full']['acc']:.4f}, AUC {state['meta']['ref_full']['auc']:.4f}.", "",
         "| Rank | Layer | Kind | KL W8A16 | KL W8A8 | KL W4A16 | KL FP8 | KL W8A8 per-channel act (depthwise only) | logit MSE W8A8 | dAcc W8A8 (pp) | dAUC W8A8 (x1e-3) | dAcc W4A16 (pp) | dAcc FP8 (pp) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(order):
        s = {sc: sub[f"{n}|{sc}"] for sc in qlib.SCHEMES}
        pc = f"{sub[n + '|W8A8pc']['kl']:.2e}" if n + '|W8A8pc' in sub else '-'
        L.append(f"| {i + 1} | {n} | {kinds.get(n, '')} | {s['W8A16']['kl']:.2e} | {s['W8A8']['kl']:.2e} | {s['W4A16']['kl']:.2e} | {s['FP8']['kl']:.2e} | {pc} | "
                 f"{s['W8A8']['logit_mse']:.2e} | {100 * s['W8A8']['d_acc']:+.2f} | {1e3 * s['W8A8']['d_auc']:+.2f} | {100 * s['W4A16']['d_acc']:+.2f} | {100 * s['FP8']['d_acc']:+.2f} |")
    fl = state.get("full", {})
    if fl:
        L += ["", "## Full Test set (10,905 images): the 10 most sensitive layers and 10 random others", "",
              "| Group | Layer | dAcc W8A16 (pp) | dAcc W8A8 | dAcc W4A16 | dAcc FP8 | dAUC W8A16 (x1e-3) | dAUC W8A8 | dAUC W4A16 | dAUC FP8 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for g in ("top", "random"):
            for n in state["full_sets"][g]:
                f = {sc: fl[f"{n}|{sc}"] for sc in qlib.SCHEMES}
                L.append(f"| {g} | {n} | " + " | ".join(f"{100 * f[sc]['d_acc']:+.2f}" for sc in qlib.SCHEMES) + " | " +
                         " | ".join(f"{1e3 * f[sc]['d_auc']:+.2f}" for sc in qlib.SCHEMES) + " |")
    (qlib.OUT / f"sensitivity_{tag}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
