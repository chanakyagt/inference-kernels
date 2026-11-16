import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import SqueezeExcitation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib
import qops
from int8_dwconv import bf16_dwconv_nhwc, int8_dwconv_nhwc, quantize_dw_weight

BF16 = torch.bfloat16


def eligible(name, kind, cin, cout, mode):
    if kind in ("stem3x3", "mamba_conv1d"):
        return False
    if kind.startswith("dw"):
        return mode == "int8"
    if mode == "fp8":
        return cin % 16 == 0 and cout % 16 == 0
    return True


class PWOp(nn.Module):
    def __init__(self, name, weight, bias, act, mode, amax_in, bf16_impl="cublas", int8_impl="tri", out_scale=None):
        super().__init__()
        self.name, self.act, self.mode = name, act, mode
        self.bf16_impl, self.int8_impl, self.out_scale = bf16_impl, int8_impl, out_scale
        w = weight.detach().float().reshape(weight.shape[0], -1)
        b = bias.detach().float() if bias is not None else torch.zeros(w.shape[0], device=w.device)
        self.n = w.shape[0]
        self.register_buffer("bias_f", b.contiguous())
        if mode == "bf16":
            self.register_buffer("w", w.to(BF16).contiguous())
            self.register_buffer("bias_b", b.to(BF16))
        elif mode == "int8":
            self.sx = float(amax_in) / 127
            wq, ws = qops.quant_weight_int8(w)
            self.register_buffer("wq", wq)
            self.register_buffer("wq_t", wq.t().contiguous())
            self.register_buffer("ws", ws)
        elif mode == "fp8":
            self.sx = float(amax_in) / qops.FP8_MAX
            w8, sw = qops.quant_weight_fp8(w)
            self.register_buffer("w8_t", w8.t())
            self.register_buffer("sw", sw.float())
            self.register_buffer("sx_t", torch.tensor(self.sx, dtype=torch.float32, device=w.device))
            self.register_buffer("bias_b", b.to(BF16))

    def forward(self, x):
        if self.mode == "bf16":
            if self.bf16_impl == "tri":
                return qops.gemm_bf16(x, self.w, self.bias_f, self.act)
            y = torch.addmm(self.bias_b, x, self.w.t())
            return F.silu(y) if self.act else y
        if self.mode == "int8":
            a_q = x.dtype == torch.int8
            if self.int8_impl == "mm":
                return qops.int_mm_pipeline(x, self.wq_t, self.ws, self.bias_f, self.sx, self.act, a_q)
            return qops.gemm_int8(x, self.wq, self.ws, self.bias_f, self.sx, self.act, a_q, self.out_scale)
        return qops.fp8_pipeline(x, self.w8_t, self.sw, self.bias_b, self.sx_t, self.sx, self.act)


class DWOp(nn.Module):
    def __init__(self, name, conv, act, mode, amax_in, bf16_impl="cublas"):
        super().__init__()
        self.name, self.act, self.mode, self.bf16_impl = name, act, mode, bf16_impl
        self.k, self.stride, self.pad = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        b = conv.bias.detach().float()
        self.register_buffer("bias_f", b.contiguous())
        if mode == "int8":
            sxc = (amax_in.float().clamp(min=1e-8) / 127).to(conv.weight.device)
            wq, ws = quantize_dw_weight(conv.weight)
            self.register_buffer("wq", wq)
            self.register_buffer("ws", (ws * sxc).contiguous())
            self.register_buffer("inv_sx", (1.0 / sxc).contiguous())
        else:
            self.register_buffer("w4", conv.weight.detach().to(BF16).contiguous())
            self.register_buffer("w_kc", conv.weight.detach().float().reshape(conv.weight.shape[0], -1).t().contiguous())
            self.register_buffer("bias_b", b.to(BF16))

    def forward(self, x):
        if self.mode == "int8":
            if x.dtype != torch.int8:
                x = qops.quant_act_pc(x, self.inv_sx)
            return int8_dwconv_nhwc(x, 1.0, self.wq, self.ws, self.bias_f, self.stride, self.pad, self.k, self.act)
        if self.bf16_impl == "tri":
            return bf16_dwconv_nhwc(x, self.w_kc, self.bias_f, self.stride, self.pad, self.k, self.act)
        c = x.shape[-1]
        y = F.conv2d(x.permute(0, 3, 1, 2), self.w4, self.bias_b, self.stride, self.pad, groups=c)
        y = y.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1)
        return F.silu(y) if self.act else y


class SEOp(nn.Module):
    def __init__(self, fc1, fc2):
        super().__init__()
        self.fc1, self.fc2 = fc1, fc2

    def forward(self, x):
        s = x.mean((1, 2))
        s = self.fc2(self.fc1(s))
        return x * torch.sigmoid(s)[:, None, None, :]


class Block(nn.Module):
    def __init__(self, ops, res):
        super().__init__()
        self.ops = nn.ModuleList(ops)
        self.res = res

    def forward(self, x):
        h = x
        for op in self.ops:
            if isinstance(op, PWOp):
                n, hh, ww, c = h.shape
                h = op(h.reshape(n * hh * ww, c)).reshape(n, hh, ww, -1)
            else:
                h = op(h)
        return x + h if self.res else h


class QLin(nn.Module):
    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        sh = x.shape
        return self.op(x.reshape(-1, sh[-1])).reshape(*sh[:-1], self.op.n)


class QNet(nn.Module):
    def __init__(self, folded, amax, cfg=None, bf16_impl=None, int8_impl=None, amax_pc=None):
        super().__init__()
        cfg = cfg or {}
        amax_pc = {k: torch.tensor(v) for k, v in (amax_pc or {}).items()}
        bf16_impl = bf16_impl or {}
        int8_impl = int8_impl or {}
        mods = dict(folded.named_modules())

        def mode_of(name):
            return cfg.get(name, "bf16")

        def mk_pw(name, conv, act, out_scale=None):
            m = mode_of(name)
            return PWOp(name, conv.weight, conv.bias, act, m, amax.get(name, 1.0), bf16_impl.get(name, "cublas"),
                        int8_impl.get(name, "tri"), out_scale)

        def mk_dw(name, conv, act):
            return DWOp(name, conv, act, mode_of(name), amax_pc.get(name), bf16_impl.get(name, "cublas"))

        stem = folded.trunk[0][0]
        self.stem_w = nn.Parameter(stem.weight.detach().to(BF16).contiguous(memory_format=torch.channels_last), requires_grad=False)
        self.stem_b = nn.Parameter(stem.bias.detach().to(BF16), requires_grad=False)
        self.stem_stride, self.stem_pad = stem.stride[0], stem.padding[0]
        blocks = []
        for si in range(1, 8):
            for bi, mb in enumerate(folded.trunk[si]):
                ops = []
                children = list(mb.block)
                for ci, ch in enumerate(children):
                    prefix = f"trunk.{si}.{bi}.block.{ci}"
                    if isinstance(ch, SqueezeExcitation):
                        fc1 = QLin(mk_pw(f"{prefix}.fc1", ch.fc1, True))
                        fc2 = QLin(mk_pw(f"{prefix}.fc2", ch.fc2, False))
                        ops.append(SEOp(fc1, fc2))
                        continue
                    conv = ch[0]
                    act = isinstance(ch[-1], nn.SiLU)
                    nm = f"{prefix}.0"
                    if conv.groups > 1:
                        ops.append(mk_dw(nm, conv, act))
                    else:
                        nxt = children[ci + 1] if ci + 1 < len(children) else None
                        out_scale = None
                        if mode_of(nm) == "int8" and nxt is not None and not isinstance(nxt, SqueezeExcitation) and nxt[0].groups > 1:
                            nxt_name = f"trunk.{si}.{bi}.block.{ci + 1}.0"
                            if mode_of(nxt_name) == "int8" and int8_impl.get(nm, "tri") == "tri":
                                out_scale = (amax_pc[nxt_name].float().clamp(min=1e-8) / 127).cuda()
                        ops.append(mk_pw(nm, conv, act, out_scale))
                blocks.append(Block(ops, mb.use_res_connect))
        self.blocks = nn.ModuleList(blocks)
        self.top = Block([mk_pw("trunk.8.0", folded.trunk[8][0], True)], False)
        head = copy.deepcopy(folded.head).to(BF16)
        self.temporal = folded.temporal
        if self.temporal:
            for layer in head.layers:
                layer.mixer.A_log.data = layer.mixer.A_log.data.float()
                layer.mixer.D.data = layer.mixer.D.data.float()
        for name, mod in list(head.named_modules()):
            for cname, child in list(mod.named_children()):
                full = f"head.{name}.{cname}" if name else f"head.{cname}"
                if isinstance(child, nn.Linear) and mode_of(full) != "bf16":
                    op = PWOp(full, child.weight, child.bias, False, mode_of(full), amax.get(full, 1.0), "cublas", int8_impl.get(full, "tri"))
                    setattr(mod, cname, QLin(op))
        self.head = head

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        h = F.silu(F.conv2d(x, self.stem_w, self.stem_b, self.stem_stride, self.stem_pad))
        h = h.contiguous(memory_format=torch.channels_last).permute(0, 2, 3, 1)
        for b in self.blocks:
            h = b(h)
        h = self.top(h)
        f = h.mean((1, 2))
        return self.head(f[:, None, :] if self.temporal else f).float()


def stock_bf16(folded):
    m = copy.deepcopy(folded).to(BF16)
    if folded.temporal:
        for layer in m.head.layers:
            layer.mixer.A_log.data = layer.mixer.A_log.data.float()
            layer.mixer.D.data = layer.mixer.D.data.float()
    return m.to(memory_format=torch.channels_last)
