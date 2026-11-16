"""Fused EfficientNet-B0 v2 trunk + 4-layer Mamba head, one new frame at a time.

Model = fine-tuned v2 trunk (_artifacts/bb_runs/ft_effnet_b0_v2, 256 px, bicubic) + the Mamba head trained on
frozen v2 features (deepfake-detector-mamba/results/images_megadataset_ft/single_H1, d_model 512, 4 layers).

Paths built here (all keep Mamba state across frames, one step per new frame):
  RefModel          fp32 PyTorch trunk + parallel head (the reference; T=1 per image for the Test set)
  UnfusedStream     plain PyTorch trunk (cuDNN, NCHW) + PyTorch recurrent Mamba step        [baseline a]
  TrunkFusedTorchHead  fused v2 trunk (own Triton) + headconv + PyTorch mean/proj/Mamba step   [baseline b]
  FusedStream       fused v2 trunk (stem + blocks) -> kernel A (headconv+SiLU+pool+proj) -> kernel B (4 Mamba
                    layers + norm_f + running mean + MLP head)                                  [new]
  WindowHead        fused v2 trunk once per frame + 30-frame window head recomputed (Mamba fused scan, or one
                    attention layer with RANDOM weights, as in the 11 Nov baseline)              [baseline c]
Existing code under fused_model/, stream/, kernels/ is imported, not modified.
"""
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
IK = HERE.parent
for p in (IK, IK / "kernels", IK / "trained", IK / "fused_model", IK / "stream"):
    sys.path.insert(0, str(p))

from tcommon import load_model, fold_conv_bn  # noqa: E402
from fmodel import PW, dt as F16  # noqa: E402
from fmodel2 import FusedNet2  # noqa: E402
from fused_dwconv_nhwc import patch_model as patch_dw_nhwc  # noqa: E402
from fused_tail_proj import FusedTailProj  # noqa: E402
from fused_mamba_step import FusedMambaStack  # noqa: E402
from selective_scan import patch_model as patch_scan  # noqa: E402
from dfmamba.model import TemporalClassifier  # noqa: E402

CKPT_ROOT = Path(os.environ.get("DFM_CKPT_ROOT", "/mnt/c/Users/chana/Projects/_artifacts"))
TRUNK_DIR = CKPT_ROOT / "bb_runs" / "ft_effnet_b0_v2"
HEAD_CKPT = IK.parent / "deepfake-detector-mamba" / "results" / "images_megadataset_ft" / "single_H1" / "model.pt"
SIZE = 256


def load_head(device="cuda"):
    ck = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    head = TemporalClassifier(m["d_model"], m["n_layers"], m["d_state"], m["dropout"], 2, m["temporal"], m["feat_dim"])
    head.load_state_dict(ck["model"])
    head = head.eval().to(device)
    for p in head.parameters():
        p.requires_grad_(False)
    for L in head.layers:
        L.mixer.scan_impl = "ref"
    return head


def load_trunk_net():
    """The v2 linear-head model from tcommon (fp32, unfolded); only its trunk is used."""
    net, interp = load_model(str(TRUNK_DIR))
    assert interp == "bicubic" and net.kind == "linear"
    return net


def head_dtype(head, dtype):
    h = copy.deepcopy(head)
    if dtype != torch.float32:
        h = h.to(dtype)
        for L in h.layers:
            L.mixer.A_log.data = L.mixer.A_log.data.float()
            L.mixer.D.data = L.mixer.D.data.float()
    return h


class RefModel(nn.Module):
    """fp32 reference: trunk (features+avgpool+flatten) -> head over a [B, T, 1280] feature sequence."""

    def __init__(self, net, head):
        super().__init__()
        self.trunk, self.head = net.trunk, head

    @torch.no_grad()
    def forward(self, frames):  # frames [B, T, 3, H, W]
        b, t = frames.shape[:2]
        f = self.trunk(frames.flatten(0, 1)).view(b, t, -1)
        return self.head(f)


class _Graphable:
    """step(img) -> logits [B,2]; capture() records one step in a CUDA graph."""

    def capture(self):
        self.static_in = torch.zeros(self.B, 3, SIZE, SIZE, device="cuda")
        with torch.no_grad():
            self.step(self.static_in)  # eager first call: Switch / autotune decisions happen outside the capture stream
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self.step(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = self.step(self.static_in)
        self.reset()

    @torch.no_grad()
    def step_graph(self, img):
        self.static_in.copy_(img)
        self.graph.replay()
        return self.out


class TorchMambaStep:
    """PyTorch recurrent step of the 4-layer head (same math as stream/pipeline.TorchStreamBaseline._layers)."""

    def __init__(self, head, batch, dtype):
        self.t = head_dtype(head, dtype)
        self.B, self.dtype = batch, dtype
        mix = self.t.layers[0].mixer
        self.E, self.N = mix.d_inner, mix.d_state
        D = self.t.norm_f.weight.shape[0]
        nl = len(self.t.layers)
        self.h = [torch.zeros(batch, self.E, self.N, device="cuda") for _ in range(nl)]
        self.cs = [torch.zeros(batch, 3, self.E, device="cuda", dtype=dtype) for _ in range(nl)]
        self.ysum = torch.zeros(batch, D, device="cuda")
        self.cnt = torch.zeros(batch, 1, device="cuda")

    def reset(self):
        for t in self.h + self.cs + [self.ysum, self.cnt]:
            t.zero_()

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.h + self.cs) + self.ysum.numel() * 4

    def from_proj(self, x):
        for i, L in enumerate(self.t.layers):
            m = L.mixer
            u, z = m.in_proj(L.norm(x)).chunk(2, -1)
            win = torch.cat([self.cs[i], u.unsqueeze(1)], 1)
            self.cs[i].copy_(win[:, 1:])
            c = (win * m.conv1d.weight[:, 0, :].t()[None]).sum(1) + m.conv1d.bias
            u = F.silu(c)
            dtv = F.softplus(m.dt_proj(u).float())
            Bm, Cm = m.x_proj(u).float().chunk(2, -1)
            A = -torch.exp(m.A_log.float())
            h = torch.exp(dtv[..., None] * A) * self.h[i] + dtv[..., None] * Bm[:, None, :] * u.float()[..., None]
            self.h[i].copy_(h)
            y = ((h * Cm[:, None, :]).sum(-1) + u.float() * m.D.float()).to(x.dtype)
            x = x + m.out_proj(y * F.silu(z))
        self.ysum.add_(self.t.norm_f(x).float())
        self.cnt.add_(1)
        return self.t.head((self.ysum / self.cnt).to(self.dtype)).float()

    def __call__(self, feat):  # feat [B, 1280]
        return self.from_proj(self.t.proj(feat.to(self.dtype)))


class UnfusedStream(_Graphable):
    """[a] plain PyTorch trunk (as loaded: cuDNN NCHW, BN unfolded) + PyTorch Mamba step."""

    def __init__(self, net, head, batch, dtype=torch.float16):
        self.trunk = copy.deepcopy(net.trunk).to(dtype).eval()
        self.mamba = TorchMambaStep(head, batch, dtype)
        self.B, self.dtype = batch, dtype

    def reset(self):
        self.mamba.reset()

    def state_bytes(self):
        return self.mamba.state_bytes()

    @torch.no_grad()
    def trunk_feat(self, img):
        return self.trunk(img.to(self.dtype))

    @torch.no_grad()
    def step(self, img):
        return self.mamba(self.trunk_feat(img))


def build_fused_trunk(net):
    """FusedNet2 over a BN-folded fp16 channels_last copy of the trunk (same recipe as fmodel.build)."""
    n = copy.deepcopy(net)
    patch_dw_nhwc(n.trunk)
    fold_conv_bn(n.trunk)
    n = n.to(memory_format=torch.channels_last).to(F16)
    return FusedNet2(n)


class FusedTrunkMixin:
    def trunk_map(self, img):
        """fused stem + 16 MBConv blocks -> NHWC fp16 [B, 8, 8, 320] (before the 1x1 head conv)."""
        fn = self.fn
        if fn.sw_stem.impl is None:
            fn.sw_stem.decide(lambda: fn.stem_new(img), lambda: fn.stem_old(img))
        h = (fn.stem_new if fn.sw_stem.impl == "new" else fn.stem_old)(img)
        for b in fn.blocks:
            h = b(h)
        return h


class TrunkFusedTorchHead(_Graphable, FusedTrunkMixin):
    """[b] fused v2 trunk incl. its fused 1x1 head conv, then PyTorch mean + proj + Mamba step."""

    def __init__(self, net, head, batch, fn=None):
        self.fn = fn or build_fused_trunk(net)
        self.mamba = TorchMambaStep(head, batch, F16)
        self.B = batch

    def reset(self):
        self.mamba.reset()

    def state_bytes(self):
        return self.mamba.state_bytes()

    @torch.no_grad()
    def step(self, img):
        h = self.fn.headconv(self.trunk_map(img))
        return self.mamba(h.mean((1, 2)))


class FusedStream(_Graphable, FusedTrunkMixin):
    """[new] fused v2 trunk -> kernel A -> kernel B. dtype fp16 (timed). For an fp32 check of kernels A and B, use
    FusedHeadOnFp32Trunk below (the fused trunk itself is fp16-only)."""

    def __init__(self, net, head, batch, fn=None, tail_mode="two", fuse_out=True, bn_tile=None):
        self.fn = fn or build_fused_trunk(net)
        tail = net.trunk[0][8]  # unfolded fp32 Conv2dNormActivation(320 -> 1280)
        self.tail = FusedTailProj(tail[0], tail[1], head.proj, F16, mode=tail_mode, bn_tile=bn_tile or (32 if batch <= 2 else 64))
        self.stack = FusedMambaStack(head_dtype(head, F16), batch, F16, fuse_out=fuse_out)
        self.B = batch

    def reset(self):
        self.stack.reset()

    def state_bytes(self):
        return self.stack.state_bytes()

    @torch.no_grad()
    def step(self, img):
        h = self.trunk_map(img)                       # NHWC
        self.tail(h.permute(0, 3, 1, 2), out=self.stack.xbuf[0])   # view with NHWC strides
        return self.stack.step_from_buf0()


class FusedHeadOnFp32Trunk:
    """fp32 check of kernels A and B: fp32 PyTorch trunk up to stage 7 -> kernel A fp32 -> kernel B fp32."""

    def __init__(self, net, head, batch):
        self.front = net.trunk[0][:8]
        tail = net.trunk[0][8]
        self.tail = FusedTailProj(tail[0], tail[1], head.proj, torch.float32, mode="two", bn_tile=64)
        self.stack = FusedMambaStack(head, batch, torch.float32, fuse_out=True)

    def reset(self):
        self.stack.reset()

    @torch.no_grad()
    def step(self, img):
        self.tail(self.front(img), out=self.stack.xbuf[0])
        return self.stack.step_from_buf0()


class WindowHead(_Graphable, FusedTrunkMixin):
    """[c] fused v2 trunk + fused 1x1 head conv once per frame, features kept for the last W frames, head recomputed
    over the window. attention=True swaps the Mamba head for one pre-norm Transformer layer with RANDOM weights
    (timing only, untrained; same design as the 11 Nov baseline)."""

    def __init__(self, net, head, batch, window=30, attention=False, fn=None):
        self.fn = fn or build_fused_trunk(net)
        self.B, self.W, self.attention = batch, window, attention
        t = head_dtype(head, F16)
        D = t.proj.out_features
        if attention:
            torch.manual_seed(0)
            self.pos = torch.zeros(1, window, D, device="cuda", dtype=F16)
            self.attn = nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.0, activation="gelu", batch_first=True,
                                                   norm_first=True).cuda().to(F16).eval()
        else:
            patch_scan(t)
        self.t = t
        self.feat = torch.zeros(batch, window, 1280, device="cuda", dtype=F16)

    def reset(self):
        self.feat.zero_()

    def state_bytes(self):
        return self.feat.numel() * self.feat.element_size()

    def _head(self, feats):
        t = self.t
        if self.attention:
            x = self.attn(t.proj(feats) + self.pos)
            return t.head(t.norm_f(x).mean(1))
        return t(feats)

    @torch.no_grad()
    def step(self, img):
        f = self.fn.headconv(self.trunk_map(img)).mean((1, 2))
        self.feat.copy_(torch.cat([self.feat[:, 1:], f[:, None]], 1))
        return self._head(self.feat).float()
