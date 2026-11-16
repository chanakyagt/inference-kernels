import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
import qlib
import qnet

BF16 = torch.bfloat16


@torch.no_grad()
def real_logits(net, x_u8, bs=64):
    outs = []
    for i in range(0, len(x_u8), bs):
        outs.append(net(qlib.normalize(x_u8[i:i + bs], BF16)).float().cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="diloco")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    size = qlib.default_size(a.ckpt)
    xt, yt = qlib.load_data("test", size)
    xc, _ = qlib.load_data("calib", size)
    idx = qlib.subset_idx(yt, a.n, seed=1)
    xs = xt[idx]
    fp = qlib.build_model(a.ckpt)
    amax = qlib.calibrate(fp, xc)
    amax_pc = qlib.calibrate_pc(fp, xc)
    ref = qlib.run_logits(fp, xs, 64)
    folded = qlib.build_model(a.ckpt)
    wraps = qlib.wrap_all(fp, amax, amax_pc)
    base = qnet.QNet(folded, amax, {}, amax_pc=amax_pc).eval()
    lg_base = real_logits(base, xs)
    floor = qlib.metrics(lg_base, yt[idx], ref)
    mods = {n: m.mod for n, m in wraps.items()}
    kinds = {}
    for n, m in mods.items():
        kinds.setdefault(qlib.layer_kind(n, m), []).append(n)
    picks = [v[len(v) // 2] for k, v in sorted(kinds.items()) if k not in ("stem3x3", "mamba_conv1d")]
    rows = []
    for n in picks:
        kind = qlib.layer_kind(n, mods[n])
        for sch, mode in ((("W8A8pc" if kind.startswith("dw") else "W8A8"), "int8"), ("FP8", "fp8")):
            w = mods[n].weight
            cin = w.shape[1] if w.dim() > 1 else 1
            if not qnet.eligible(n, kind, cin, w.shape[0], mode):
                continue
            wraps[n].scheme = sch
            lf = qlib.run_logits(fp, xs, 64)
            wraps[n].scheme = None
            net = qnet.QNet(folded, amax, {n: mode}, amax_pc=amax_pc).eval()
            lr = real_logits(net, xs)
            mf, mr = qlib.metrics(lf, yt[idx], ref), qlib.metrics(lr, yt[idx], ref)
            rows.append({"layer": n, "kind": kind, "scheme": sch, "kl_fake": mf["kl"], "kl_real": mr["kl"],
                         "mean_abs_logit_diff_real_vs_fake": (lr - lf).abs().mean().item()})
            print(rows[-1], flush=True)
    for w in wraps.values():
        w.scheme = "W8A8pc" if w.amax_pc is not None else "W8A8"
    lf = qlib.run_logits(fp, xs, 64)
    for w in wraps.values():
        w.scheme = None
    cfg = {n: "int8" for n in wraps}
    lr = real_logits(qnet.QNet(folded, amax, cfg, amax_pc=amax_pc).eval(), xs)
    all_row = {"fake_all_W8A8": qlib.metrics(lf, yt[idx], ref), "real_all_int8": qlib.metrics(lr, yt[idx], ref),
               "mean_abs_logit_diff_real_vs_fake": (lr - lf).abs().mean().item()}
    out = {"n": len(idx), "bf16_floor_vs_fp32": floor, "rows": rows, "all_int8": all_row}
    (qlib.OUT / f"validate_real_{a.tag}.json").write_text(json.dumps(out, indent=1))
    L = [f"# Real kernels versus simulated quantization ({a.tag}, {len(idx)} Test images)", "",
         f"The bf16 path with no quantization already differs from FP32 by KL {floor['kl']:.2e}; that is the noise floor for the real kernels.", "",
         "| Layer | Kind | Scheme | KL simulated (fp32) | KL real kernels (bf16) | Mean abs logit diff |", "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['layer']} | {r['kind']} | {r['scheme']} | {r['kl_fake']:.2e} | {r['kl_real']:.2e} | {r['mean_abs_logit_diff_real_vs_fake']:.3f} |")
    L += ["", f"All layers int8 (W8A8): simulated KL {all_row['fake_all_W8A8']['kl']:.2e}, accuracy {all_row['fake_all_W8A8']['acc']:.4f}; real kernels KL {all_row['real_all_int8']['kl']:.2e}, accuracy {all_row['real_all_int8']['acc']:.4f}. FP32 accuracy on these images {qlib.metrics(ref, yt[idx])['acc']:.4f}."]
    (qlib.OUT / f"validate_real_{a.tag}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
