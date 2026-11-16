import os
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import DETECTOR_REPO

sys.path.insert(0, str(DETECTOR_REPO))
from dfmamba.data import MEAN, STD
from dfmamba.model import FrameEncoder, temporal_from_cfg

CACHE = Path.home() / "qcache"
OUT = ROOT / "results" / "quant_layerwise"
_CKPT_ROOT = Path(os.environ.get("DFM_CKPT_ROOT", "/tmp"))
HEAD_CKPT = str(_CKPT_ROOT / "img_runs" / "diloco_H100" / "model.pt") if (_CKPT_ROOT / "img_runs" / "diloco_H100" / "model.pt").exists() else str(Path.home() / "work" / "img_runs" / "diloco_H100" / "model.pt")
DATA = Path.home() / "data" / "MegaDataset"
SCHEMES = ["W8A16", "W8A8", "W4A16", "FP8"]
DW_SCHEMES = ["W8A8pc"]
FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
STD_T = torch.tensor(STD).view(1, 3, 1, 1)


def default_size(ckpt):
    return 224 if ckpt is None else 256


def _cname(name, size):
    return name if size == 224 else f"{name}_{size}"


def build_cache(size=224):
    CACHE.mkdir(exist_ok=True)
    if (CACHE / f"{_cname('test', size)}_u8.npy").exists() and (CACHE / f"{_cname('calib', size)}_u8.npy").exists():
        return
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from dfmamba.extract_images import ImageFolderFaces
    resample = Image.BILINEAR if size == 224 else Image.BICUBIC

    class U8(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            p, label = self.ds.items[i]
            img = Image.open(p).convert("RGB").resize((size, size), resample)
            return torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1), label

    for split, name, limit in (("Test", "test", None), ("Train", "calib", 512)):
        ds = ImageFolderFaces(DATA, split, limit=limit, seed=0)
        n = len(ds)
        arr = np.lib.format.open_memmap(CACHE / f"{_cname(name, size)}_u8.npy", mode="w+", dtype=np.uint8, shape=(n, 3, size, size))
        ys = []
        k = 0
        for x, y in DataLoader(U8(ds), batch_size=64, num_workers=4):
            arr[k:k + len(x)] = x.numpy()
            k += len(x)
            ys.append(y)
        arr.flush()
        np.save(CACHE / f"{_cname(name, size)}_y.npy", torch.cat(ys).numpy())
        (CACHE / f"{_cname(name, size)}_names.json").write_text(json.dumps([str(p) for p, _ in ds.items]))
        print(name, n, flush=True)


class MemU8:
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            return torch.from_numpy(self.arr[key.numpy()])
        return torch.from_numpy(np.array(self.arr[key]))


def load_data(name, size=224):
    arr = np.load(CACHE / f"{_cname(name, size)}_u8.npy", mmap_mode="r")
    y = torch.from_numpy(np.load(CACHE / f"{_cname(name, size)}_y.npy"))
    return (MemU8(arr) if name == "test" else torch.from_numpy(np.array(arr))), y


def normalize(xu8, dtype=torch.float32):
    x = xu8.cuda(non_blocking=True).float() / 255.0
    x = (x - MEAN_T.cuda()) / STD_T.cuda()
    return x.to(dtype)


class Detector(nn.Module):
    def __init__(self, trunk, head, temporal=True):
        super().__init__()
        self.trunk = trunk
        self.head = head
        self.temporal = temporal

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.trunk(x), 1).flatten(1)
        return self.head(f[:, None, :] if self.temporal else f)


def fold_bn_trunk(trunk):
    n = 0
    for mod in trunk.modules():
        if isinstance(mod, nn.Sequential) and len(mod) >= 2 and isinstance(mod[0], nn.Conv2d) and isinstance(mod[1], nn.BatchNorm2d):
            conv, bn = mod[0], mod[1]
            s = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
            w = conv.weight.detach().double() * s.view(-1, 1, 1, 1)
            b = bn.bias.detach().double() - bn.running_mean.double() * s
            if conv.bias is not None:
                b = b + conv.bias.detach().double() * s
            new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                            groups=conv.groups, bias=True)
            new.weight.data = w.float()
            new.bias.data = b.float()
            mod[0] = new
            mod[1] = nn.Identity()
            n += 1
    return n


def build_model(ckpt=None, fold=True):
    temporal = True
    if ckpt is None:
        c = torch.load(HEAD_CKPT, map_location="cpu")
        enc = FrameEncoder(pretrained=True)
        head = temporal_from_cfg(c["config"])
        head.load_state_dict(c["model"])
    else:
        c = torch.load(ckpt, map_location="cpu")
        enc = FrameEncoder(pretrained=False)
        sd = c["model"]
        enc.features.load_state_dict({k[len("trunk.body.0."):]: v for k, v in sd.items() if k.startswith("trunk.body.0.")})
        if "config" in c:
            head = temporal_from_cfg(c["config"])
            head.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
        else:
            temporal = False
            head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 2))
            head.load_state_dict({k[len("head."):]: v for k, v in sd.items() if k.startswith("head.")})
    if temporal:
        for layer in head.layers:
            layer.mixer.scan_impl = "ref"
    m = Detector(enc.features, head, temporal).eval().cuda().float()
    if fold:
        fold_bn_trunk(m.trunk)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def quant_layers(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            out.append((name, mod))
    return out


def layer_kind(name, mod):
    if isinstance(mod, nn.Linear):
        if name == "head.1":
            return "head_fc"
        if name.startswith("head.layers"):
            return "mamba_" + name.split(".")[-1]
        return "head_proj" if name == "head.proj" else "head_mlp"
    if isinstance(mod, nn.Conv1d):
        return "mamba_conv1d"
    if ".fc1" in name or ".fc2" in name:
        return "se_fc"
    if mod.groups > 1 and mod.groups == mod.in_channels:
        return f"dw{mod.kernel_size[0]}x{mod.kernel_size[0]}"
    if mod.kernel_size == (3, 3):
        return "stem3x3"
    return "pw1x1"


def qdq_weight(w, scheme):
    w = w.float()
    if scheme in ("W8A16", "W8A8", "W8A8pc"):
        flat = w.reshape(w.shape[0], -1)
        s = flat.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
        return (torch.round(flat / s).clamp(-127, 127) * s).reshape(w.shape)
    if scheme == "W4A16":
        flat = w.reshape(w.shape[0], -1)
        n = flat.shape[1]
        pad = (-n) % 64
        if pad:
            flat = F.pad(flat, (0, pad))
        g = flat.reshape(flat.shape[0], -1, 64)
        s = g.abs().amax(2, keepdim=True).clamp(min=1e-12) / 7
        q = (torch.round(g / s).clamp(-8, 7) * s).reshape(flat.shape[0], -1)
        return q[:, :n].reshape(w.shape)
    if scheme == "FP8":
        s = w.abs().amax().clamp(min=1e-12) / FP8_MAX
        return (w / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).float() * s
    raise ValueError(scheme)


def qdq_act(x, scheme, amax, amax_pc=None):
    if scheme == "W8A8pc":
        s = (amax_pc / 127).view(1, -1, *([1] * (x.dim() - 2)))
        return torch.round(x / s).clamp(-127, 127) * s
    if scheme == "W8A8":
        s = amax / 127
        return torch.round(x / s).clamp(-127, 127) * s
    if scheme == "FP8":
        s = amax / FP8_MAX
        return (x / s).clamp(-FP8_MAX, FP8_MAX).to(FP8).float() * s
    return x


class QWrap(nn.Module):
    def __init__(self, mod, amax, amax_pc=None):
        super().__init__()
        self.mod = mod
        self.amax = float(amax)
        self.amax_pc = None if amax_pc is None else torch.tensor(amax_pc, dtype=torch.float32, device="cuda").clamp(min=1e-8)
        self.scheme = None
        self.cache = {}

    def weight_for(self, scheme):
        if scheme not in self.cache:
            self.cache[scheme] = qdq_weight(self.mod.weight, scheme)
        return self.cache[scheme]

    def forward(self, x):
        if self.scheme is None:
            return self.mod(x)
        w = self.weight_for(self.scheme)
        x = qdq_act(x, self.scheme, self.amax, self.amax_pc)
        m = self.mod
        if isinstance(m, nn.Linear):
            return F.linear(x, w, m.bias)
        if isinstance(m, nn.Conv1d):
            return F.conv1d(x, w, m.bias, m.stride, m.padding, m.dilation, m.groups)
        return F.conv2d(x, w, m.bias, m.stride, m.padding, m.dilation, m.groups)


def set_module(root, name, new):
    parts = name.split(".")
    p = root
    for q in parts[:-1]:
        p = getattr(p, q)
    setattr(p, parts[-1], new)


def calibrate(model, calib_u8, bs=64):
    amax = {}
    hooks = []
    layers = quant_layers(model)
    for name, mod in layers:
        def hook(m, inp, out, name=name):
            amax.setdefault(name, []).append(inp[0].detach().abs().max().item())
        hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for i in range(0, len(calib_u8), bs):
            model(normalize(calib_u8[i:i + bs]))
    for h in hooks:
        h.remove()
    return {k: float(np.mean(v)) for k, v in amax.items()}


def calibrate_pc(model, calib_u8, bs=64):
    acc = {}
    hooks = []
    for name, mod in quant_layers(model):
        if isinstance(mod, nn.Conv2d) and mod.groups > 1 and mod.groups == mod.in_channels:
            def hook(m, inp, out, name=name):
                acc.setdefault(name, []).append(inp[0].detach().abs().amax(dim=(0, 2, 3)).cpu())
            hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for i in range(0, len(calib_u8), bs):
            model(normalize(calib_u8[i:i + bs]))
    for h in hooks:
        h.remove()
    return {k: torch.stack(v).mean(0).tolist() for k, v in acc.items()}


def wrap_all(model, amax, amax_pc=None):
    wraps = {}
    amax_pc = amax_pc or {}
    for name, mod in quant_layers(model):
        w = QWrap(mod, amax[name], amax_pc.get(name))
        set_module(model, name, w)
        wraps[name] = w
    return wraps


def auc_score(scores, labels):
    scores = scores.double().cpu()
    labels = labels.cpu()
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    sorted_s = scores[order]
    r = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    uniq, inv, cnt = torch.unique_consecutive(sorted_s, return_inverse=True, return_counts=True)
    csum = torch.cumsum(cnt, 0).double()
    avg = csum - (cnt.double() - 1) / 2
    ranks[order] = avg[inv]
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return ((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item()


@torch.no_grad()
def run_logits(model, x_u8, bs=125):
    outs = []
    for i in range(0, len(x_u8), bs):
        outs.append(model(normalize(x_u8[i:i + bs])).float().cpu())
    return torch.cat(outs)


def metrics(logits, labels, ref=None):
    pred = logits.argmax(1)
    d = {"acc": (pred == labels).float().mean().item(), "auc": auc_score(logits[:, 1] - logits[:, 0], labels)}
    if ref is not None:
        lp, lr = F.log_softmax(logits, 1), F.log_softmax(ref, 1)
        d["kl"] = (lr.exp() * (lr - lp)).sum(1).mean().item()
        d["logit_mse"] = ((logits - ref) ** 2).mean().item()
        d["flip"] = (pred != ref.argmax(1)).float().mean().item()
    return d


def subset_idx(y, n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = []
    for c in (0, 1):
        ii = (y == c).nonzero().flatten()
        k = int(round(n * len(ii) / len(y)))
        idx.append(ii[torch.randperm(len(ii), generator=g)[:k]])
    return torch.sort(torch.cat(idx)).values


def provenance():
    import subprocess
    q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"nvidia_smi": q, "torch": torch.__version__}
