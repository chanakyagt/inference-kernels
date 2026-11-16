import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib


def layer_record(name, mod, in_shape, out_shape):
    kind = qlib.layer_kind(name, mod)
    r = {"name": name, "kind": kind, "type": type(mod).__name__}
    if isinstance(mod, nn.Linear):
        rows = 1
        macs = mod.in_features * mod.out_features
        r.update(cin=mod.in_features, cout=mod.out_features, k=1, groups=1, h=1, w=1, ho=1, wo=1)
        act_in, act_out = mod.in_features, mod.out_features
    elif isinstance(mod, nn.Conv1d):
        macs = mod.in_channels * mod.kernel_size[0] // mod.groups * mod.out_channels
        r.update(cin=mod.in_channels, cout=mod.out_channels, k=mod.kernel_size[0], groups=mod.groups, h=1, w=1, ho=1, wo=1)
        act_in, act_out = mod.in_channels, mod.out_channels
    else:
        h, w = in_shape[2], in_shape[3]
        ho, wo = out_shape[2], out_shape[3]
        macs = mod.out_channels * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1] * ho * wo
        r.update(cin=mod.in_channels, cout=mod.out_channels, k=mod.kernel_size[0], stride=mod.stride[0], groups=mod.groups,
                 h=h, w=w, ho=ho, wo=wo)
        act_in, act_out = mod.in_channels * h * w, mod.out_channels * ho * wo
    params = mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0)
    r["params"] = params
    r["flops_per_image"] = 2 * macs
    for dt, nb in (("bf16", 2), ("int8", 1)):
        for B in (1, 64):
            byt = params * nb + B * (act_in + act_out) * nb
            r[f"bytes_{dt}_b{B}"] = byt
            r[f"intensity_{dt}_b{B}"] = 2 * macs * B / byt
    return r


def inventory(model, size=224):
    shapes = {}
    hooks = []
    for name, mod in qlib.quant_layers(model):
        hooks.append(mod.register_forward_hook(lambda m, i, o, name=name: shapes.__setitem__(name, (tuple(i[0].shape), tuple(o.shape)))))
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device="cuda"))
    for h in hooks:
        h.remove()
    return [layer_record(n, m, *shapes[n]) for n, m in qlib.quant_layers(model)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default="diloco")
    a = ap.parse_args()
    model = qlib.build_model(a.ckpt)
    recs = inventory(model, qlib.default_size(a.ckpt))
    qlib.OUT.mkdir(parents=True, exist_ok=True)
    (qlib.OUT / f"inventory_{a.tag}.json").write_text(json.dumps(recs, indent=1))
    L = [f"# Layer inventory ({a.tag}, BatchNorm folded into the convolutions, bf16 byte counts)", "",
         "FLOP/byte counts weights once per batch plus input and output activations scaled by the batch. The card's ridge point is 157 FLOP/byte.", "",
         "| Layer | Kind | Cin | Cout | k | groups | In HxW | Out HxW | Params | MFLOP/img | FLOP/B b1 | FLOP/B b64 | KB bf16 b1 | MB bf16 b64 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        L.append(f"| {r['name']} | {r['kind']} | {r['cin']} | {r['cout']} | {r['k']} | {r['groups']} | {r['h']}x{r['w']} | {r['ho']}x{r['wo']} | "
                 f"{r['params']:,} | {r['flops_per_image'] / 1e6:.2f} | {r['intensity_bf16_b1']:.1f} | {r['intensity_bf16_b64']:.1f} | "
                 f"{r['bytes_bf16_b1'] / 1e3:.0f} | {r['bytes_bf16_b64'] / 1e6:.2f} |")
    kinds = {}
    for r in recs:
        k = kinds.setdefault(r["kind"], {"n": 0, "params": 0, "flops": 0, "b1": 0, "b64": 0, "bytes1": 0, "bytes64": 0})
        k["n"] += 1
        k["params"] += r["params"]
        k["flops"] += r["flops_per_image"]
        k["bytes1"] += r["bytes_bf16_b1"]
        k["bytes64"] += r["bytes_bf16_b64"]
    L += ["", "## By kind", "", "| Kind | Layers | Params | MFLOP/img | Aggregate FLOP/B b1 | Aggregate FLOP/B b64 |", "|---|---|---|---|---|---|"]
    for kname, k in sorted(kinds.items(), key=lambda kv: -kv[1]["flops"]):
        L.append(f"| {kname} | {k['n']} | {k['params']:,} | {k['flops'] / 1e6:.1f} | {k['flops'] / k['bytes1']:.1f} | {k['flops'] * 64 / k['bytes64']:.1f} |")
    tot = sum(r["flops_per_image"] for r in recs)
    L += ["", f"Total: {len(recs)} layers, {sum(r['params'] for r in recs):,} parameters, {tot / 1e9:.3f} GFLOP per image."]
    (qlib.OUT / f"inventory_{a.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-14:]))


if __name__ == "__main__":
    main()
