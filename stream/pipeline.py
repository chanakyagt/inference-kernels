import os
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kernels"))
from common import DETECTOR_REPO

sys.path.insert(0, str(DETECTOR_REPO))
from fused_dwconv_bn_silu import patch_model as patch_dw
from fused_mamba_step import FusedMambaStack, reference_step_stack
from fused_tail_proj import FusedTailProj
from selective_scan import patch_model as patch_scan
from dfmamba.model import FrameEncoder, TemporalClassifier

CKPT_ROOT = os.environ.get("DFM_CKPT_ROOT", "/tmp")
CKPTS = {
    "diloco": os.path.join(CKPT_ROOT, "img_runs", "diloco_H100", "model.pt"),
    "finetuned": os.path.join(CKPT_ROOT, "bb_runs", "ft_effnet_b0_mamba_3000", "model.pt"),
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_parts(name, device="cuda"):
    sd = torch.load(CKPTS[name], map_location="cpu")
    if name == "diloco":
        cfg = sd["config"]["model"]
        enc = FrameEncoder(pretrained=True)
        temporal = TemporalClassifier(cfg["d_model"], cfg["n_layers"], cfg["d_state"], cfg["dropout"], 2, True)
        temporal.load_state_dict(sd["model"])
        features = enc.features
        resize = "bilinear"
    else:
        from compare.backbones import build
        trunk, _, _ = build("effnet_b0", pretrained=False)
        m = sd["model"]
        trunk.load_state_dict({k[len("trunk."):]: v for k, v in m.items() if k.startswith("trunk.")})
        temporal = TemporalClassifier(d_model=256, n_layers=2, d_state=16, feat_dim=1280)
        temporal.load_state_dict({k[len("head."):]: v for k, v in m.items() if k.startswith("head.")})
        features = trunk.body[0]
        resize = "bicubic"
    return features.eval().to(device), temporal.eval().to(device), resize


def to_dtype(temporal, dtype):
    t = copy.deepcopy(temporal)
    if dtype != torch.float32:
        t = t.to(dtype)
        for L in t.layers:
            L.mixer.A_log.data = L.mixer.A_log.data.float()
            L.mixer.D.data = L.mixer.D.data.float()
    return t


class OriginalModel(nn.Module):
    def __init__(self, features, temporal):
        super().__init__()
        self.features = features
        self.temporal = temporal

    def forward(self, frames):
        b, t = frames.shape[:2]
        f = self.features(frames.flatten(0, 1)).mean((2, 3)).view(b, t, -1)
        return self.temporal(f)


def make_original(features, temporal, mode="fp32", scan="ref"):
    if mode == "fp64":
        m = OriginalModel(copy.deepcopy(features).double(), copy.deepcopy(temporal).double()).eval()
        for L in m.temporal.layers:
            L.mixer.scan_impl = scan
        return m
    dtype = torch.bfloat16 if mode == "bf16" else torch.float32
    m = OriginalModel(copy.deepcopy(features).to(dtype), to_dtype(temporal, dtype)).eval()
    for L in m.temporal.layers:
        L.mixer.scan_impl = scan
    return m


@torch.no_grad()
def run_original(model, frames, mode="fp32"):
    if mode == "bf16_autocast":
        with torch.autocast("cuda", torch.bfloat16):
            return model(frames).float()
    return model(frames.to(next(model.parameters()).dtype)).float()


class StreamPipeline:
    def __init__(self, features, temporal, batch=1, dtype=torch.float32, tail_mode="two", fuse_out=True,
                 fused_dw=True, be1=None, be2=None, bn_tile=None):
        feats = copy.deepcopy(features).to(dtype).eval()
        if fused_dw:
            patch_dw(feats)
        self.trunk = feats[:8]
        tail = feats[8]
        self.tail = FusedTailProj(tail[0], tail[1], temporal.proj, dtype, mode=tail_mode, bn_tile=bn_tile or (32 if batch <= 2 else 64))
        self.stack = FusedMambaStack(temporal, batch, dtype, fuse_out=fuse_out, be1=be1, be2=be2)
        self.dtype, self.B = dtype, batch
        self.graph = None
        self.static_in = None

    def reset(self):
        self.stack.reset()

    @torch.no_grad()
    def step(self, img):
        f = self.trunk(img.to(self.dtype))
        self.tail(f, out=self.stack.xbuf[0])
        return self.stack.step_from_buf0()

    def capture(self):
        dev = self.stack.dev
        self.static_in = torch.zeros(self.B, 3, 224, 224, device=dev, dtype=self.dtype)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.step(self.static_in)
        self.reset()

    @torch.no_grad()
    def step_graph(self, img):
        self.static_in.copy_(img)
        self.graph.replay()
        return self.out

    def state_bytes(self):
        return self.stack.state_bytes()


class TorchStreamBaseline:
    def __init__(self, features, temporal, batch=1, dtype=torch.float32):
        self.features = copy.deepcopy(features).to(dtype).eval()
        self.t = to_dtype(temporal, dtype)
        self.trunk8, self.tail8 = self.features[:8], self.features[8]
        self.dtype, self.B = dtype, batch
        dev = next(self.t.parameters()).device
        mix = self.t.layers[0].mixer
        self.dev, self.E, self.N = dev, mix.d_inner, mix.d_state
        self.graph = None
        self.reset()

    def reset(self):
        if getattr(self, "states", None) is not None:
            for t in self.states["h"] + self.states["cs"] + [self.ysum, self.cnt]:
                t.zero_()
            return
        nl = len(self.t.layers)
        dt = self.dtype
        self.states = {"h": [torch.zeros(self.B, self.E, self.N, device=self.dev) for _ in range(nl)],
                       "cs": [torch.zeros(self.B, 3, self.E, device=self.dev, dtype=dt) for _ in range(nl)]}
        D = self.t.norm_f.weight.shape[0]
        self.ysum = torch.zeros(self.B, D, device=self.dev, dtype=torch.float32)
        self.cnt = torch.zeros(self.B, 1, device=self.dev, dtype=torch.float32)

    def part_trunk(self, img):
        return self.trunk8(img.to(self.dtype))

    def part_tail(self, fmap):
        return self.tail8(fmap).mean((2, 3))

    def part_rest(self, f):
        x = self._layers(self.t.proj(f))
        self.ysum.add_(self.t.norm_f(x).float())
        self.cnt.add_(1)
        return self.t.head((self.ysum / self.cnt).to(self.dtype))

    @torch.no_grad()
    def step(self, img):
        return self.part_rest(self.part_tail(self.part_trunk(img)))

    def _layers(self, x):
        st = self.states
        for i, L in enumerate(self.t.layers):
            m = L.mixer
            u, z = m.in_proj(L.norm(x)).chunk(2, -1)
            win = torch.cat([st["cs"][i], u.unsqueeze(1)], 1)
            st["cs"][i].copy_(win[:, 1:])
            c = (win * m.conv1d.weight[:, 0, :].t()[None]).sum(1) + m.conv1d.bias
            u = F.silu(c)
            dt = F.softplus(m.dt_proj(u).float())
            Bm, Cm = m.x_proj(u).float().chunk(2, -1)
            A = -torch.exp(m.A_log.float())
            h = torch.exp(dt[..., None] * A) * st["h"][i] + dt[..., None] * Bm[:, None, :] * u.float()[..., None]
            st["h"][i].copy_(h)
            y = ((h * Cm[:, None, :]).sum(-1) + u.float() * m.D.float()).to(x.dtype)
            x = x + m.out_proj(y * F.silu(z))
        return x

    def capture(self):
        self.static_in = torch.zeros(self.B, 3, 224, 224, device=self.dev, dtype=self.dtype)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.step(self.static_in)
        self.reset()

    @torch.no_grad()
    def step_graph(self, img):
        self.static_in.copy_(img)
        self.graph.replay()
        return self.out

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.states["h"] + self.states["cs"]) + self.ysum.numel() * 4


class WindowBaseline:
    def __init__(self, features, temporal, batch=1, dtype=torch.float32, window=30, recompute_trunk=False, fused_dw=True,
                 attention=False):
        feats = copy.deepcopy(features).to(dtype).eval()
        if fused_dw:
            patch_dw(feats)
        self.features = feats
        self.dtype, self.B, self.W = dtype, batch, window
        self.recompute_trunk = recompute_trunk
        self.attention = attention
        t = to_dtype(temporal, dtype)
        D = t.proj.out_features
        if attention:
            torch.manual_seed(0)
            self.t = t
            self.pos = torch.zeros(1, window, D, device=next(t.parameters()).device, dtype=dtype)
            self.attn = nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.0, activation="gelu", batch_first=True,
                                                   norm_first=True).to(self.pos.device).to(dtype).eval()
        else:
            patch_scan(t)
            self.t = t
        self.dev = next(t.parameters()).device
        self.graph = None
        self.reset()

    def reset(self):
        n = self.W
        if getattr(self, "feat", None) is not None:
            self.feat.zero_()
            if self.frames is not None:
                self.frames.zero_()
            return
        self.feat = torch.zeros(self.B, n, 1280, device=self.dev, dtype=self.dtype)
        self.frames = torch.zeros(self.B, n, 3, 224, 224, device=self.dev, dtype=self.dtype) if self.recompute_trunk else None

    def _head(self, feats):
        t = self.t
        if self.attention:
            x = t.proj(feats) + self.pos
            x = self.attn(x)
            return t.head(t.norm_f(x).mean(1))
        return t(feats)

    @torch.no_grad()
    def step(self, img):
        img = img.to(self.dtype)
        if self.recompute_trunk:
            self.frames.copy_(torch.cat([self.frames[:, 1:], img[:, None]], 1))
            f = self.features(self.frames.flatten(0, 1)).mean((2, 3)).view(self.B, self.W, -1)
            return self._head(f)
        f = self.features(img).mean((2, 3))
        self.feat.copy_(torch.cat([self.feat[:, 1:], f[:, None]], 1))
        return self._head(self.feat)

    def capture(self):
        self.static_in = torch.zeros(self.B, 3, 224, 224, device=self.dev, dtype=self.dtype)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.step(self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self.step(self.static_in)
        self.reset()

    @torch.no_grad()
    def step_graph(self, img):
        self.static_in.copy_(img)
        self.graph.replay()
        return self.out

    def state_bytes(self):
        n = self.feat.numel() * self.feat.element_size()
        if self.frames is not None:
            n += self.frames.numel() * self.frames.element_size()
        return n


class TestImages(torch.utils.data.Dataset):
    def __init__(self, items, resize):
        self.items = items
        self.mode = {"bilinear": 2, "bicubic": 3}[resize]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image
        p, label = self.items[i]
        img = Image.open(p).convert("RGB").resize((224, 224), self.mode)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), label, i


def test_items(root=None):
    import os
    from dfmamba.extract_images import ImageFolderFaces
    root = root or os.path.expanduser("~/data/MegaDataset")
    return ImageFolderFaces(root, "Test").items


def normalize(u8, device="cuda"):
    m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (u8.to(device).float() / 255.0 - m) / s


def sample_images(resize, n, seed=0, root=None):
    import numpy as np
    items = test_items(root)
    idx = np.random.default_rng(seed).choice(len(items), n, replace=False)
    ds = TestImages([items[i] for i in idx], resize)
    xs = torch.stack([ds[i][0] for i in range(n)])
    ys = torch.tensor([ds[i][1] for i in range(n)])
    return normalize(xs), ys
