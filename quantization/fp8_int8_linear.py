import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
from common import DETECTOR_REPO, bench, load_detector

sys.path.insert(0, str(DETECTOR_REPO))
from fused_dwconv_bn_silu import patch_model as patch_dw
from selective_scan import patch_model as patch_scan

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max


class FP8Linear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        s = w.abs().amax().clamp(min=1e-12) / FP8_MAX
        self.register_buffer("w8", (w / s).to(FP8).t())
        self.register_buffer("sw", s.reshape(()))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=False)
        self.out_features = lin.out_features

    def forward(self, x):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        rows = x2.shape[0]
        pad = (-rows) % 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        sx = x2.abs().amax().float().clamp(min=1e-12) / FP8_MAX
        x8 = (x2.float() / sx).to(FP8)
        y = torch._scaled_mm(x8, self.w8, scale_a=sx, scale_b=self.sw, bias=self.bias, out_dtype=torch.bfloat16)
        return y[:rows].reshape(*shp[:-1], self.out_features).to(x.dtype)


class Int8WeightOnlyLinear(nn.Module):
    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127
        self.register_buffer("wq", torch.round(w / s).clamp(-127, 127).to(torch.int8))
        self.register_buffer("s", s.to(torch.bfloat16))
        self.bias = None if lin.bias is None else nn.Parameter(lin.bias.detach().to(torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return nn.functional.linear(x, self.wq.to(x.dtype) * self.s.to(x.dtype), self.bias)


def swap_linears(model, cls, where):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and where(full) and child.in_features % 16 == 0 and child.out_features % 16 == 0:
                setattr(mod, cname, cls(child))
                n += 1
    return n


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    x = torch.randn(4, 30, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
    ref32 = load_detector(torch.float32)
    cap = {}
    h = ref32.temporal.norm_f.register_forward_hook(lambda mo, i, o: cap.__setitem__("f", o.detach().float()))
    with torch.no_grad():
        y32 = ref32(x.float())
        p32 = torch.softmax(y32, 1)[:, 1]
    h.remove()
    f32 = cap["f"]
    del ref32
    results = []
    variants = [("BF16 (fused kernels)", None, None),
                ("FP8 Mamba + projection linears", FP8Linear, lambda n: n.startswith(("temporal.layers", "temporal.proj"))),
                ("FP8 all linears", FP8Linear, lambda n: True),
                ("INT8 weight-only Mamba + projection", Int8WeightOnlyLinear, lambda n: n.startswith(("temporal.layers", "temporal.proj")))]
    for name, cls, where in variants:
        m = load_detector(torch.bfloat16)
        patch_dw(m)
        patch_scan(m)
        swapped = swap_linears(m, cls, where) if cls else 0
        hk = m.temporal.norm_f.register_forward_hook(lambda mo, i, o: cap.__setitem__("f", o.detach().float()))
        with torch.no_grad():
            y = m(x)
        hk.remove()
        ferr = rel_err(cap["f"], f32)
        with torch.no_grad():
            ms = bench(lambda: m(x), iters=30)
            feats = torch.randn(4, 30, 1280, device="cuda", dtype=torch.bfloat16)
            tail_ms = bench(lambda: m.temporal(feats), iters=50)
        p = torch.softmax(y.float(), 1)[:, 1]
        lin_bytes = sum((mod.w8.numel() if hasattr(mod, "w8") else mod.wq.numel() if hasattr(mod, "wq")
                         else mod.weight.numel() * 2) for mod in m.modules()
                        if isinstance(mod, (nn.Linear, FP8Linear, Int8WeightOnlyLinear)))
        results.append({"variant": name, "linears_swapped": swapped, "mamba_out_rel_err_vs_fp32": ferr, "logit_rel_err_vs_fp32": rel_err(y, y32),
                        "max_abs_dprob_vs_fp32": (p - p32).abs().max().item(), "full_ms_4clips": ms,
                        "mamba_and_head_ms_4clips": tail_ms, "linear_weight_MB": lin_bytes / 1e6})
        print(json.dumps(results[-1]), flush=True)
        del m
    out = ROOT / "quantization" / "results"
    out.mkdir(exist_ok=True)
    (out / "quant.json").write_text(json.dumps(results, indent=1))
    L = ["# Precision vs speed (4 clips x 30 frames, untrained weights, RTX 4060 Ti)", "",
         "Error is measured against the same model in FP32. The selective-scan recurrence stays in FP32 in every row.",
         "Detection accuracy needs trained weights and a labeled test set, which this repository does not have yet,",
         "so the error columns measure numerical drift only, not accuracy loss.", "",
         "| Variant | Linears swapped | Linear weight MB | Mamba output rel. error vs FP32 | Logit rel. error | Max change in P(fake) | Full forward ms | Mamba + head ms |",
         "|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['variant']} | {r['linears_swapped']} | {r['linear_weight_MB']:.1f} | {r['mamba_out_rel_err_vs_fp32']:.2e} | {r['logit_rel_err_vs_fp32']:.2e} | "
                 f"{r['max_abs_dprob_vs_fp32']:.4f} | {r['full_ms_4clips']:.2f} | {r['mamba_and_head_ms_4clips']:.3f} |")
    (out / "quant.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
