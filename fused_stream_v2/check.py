"""Correctness of the fused EfficientNet v2 + Mamba streaming path on all 10,905 Test images, plus a 30-frame
streaming check. Each Test image is a stream of length 1 (state reset), which is exactly how the head was trained
and evaluated (one frame per image). Writes correctness.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import (HERE, F16, FusedHeadOnFp32Trunk, FusedStream, RefModel, TrunkFusedTorchHead, UnfusedStream,
                      load_head, load_trunk_net)
from tcommon import load_cache, metrics, prep

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
AB = 64


def probs_of(step_fn, reset_fn, u8):
    out = np.empty(len(u8), dtype=np.float64)
    for i in range(0, len(u8), AB):
        x = prep(u8[i:i + AB])
        m = x.shape[0]
        if m < AB:
            x = torch.cat([x, x[-1:].expand(AB - m, -1, -1, -1)])
        if reset_fn:
            reset_fn()
        with torch.no_grad():
            lg = step_fn(x)
        out[i:i + m] = lg.float().softmax(1)[:m, 1].double().cpu().numpy()
    return out


def compare(p, ref, y):
    r = metrics(y, p)
    r["flips_vs_ref_fp32"] = int(((p > 0.5) != (ref > 0.5)).sum())
    r["max_abs_dprob_vs_ref_fp32"] = float(np.abs(p - ref).max())
    return r


@torch.no_grad()
def main():
    t0 = time.time()
    res = {}
    head = load_head()
    # 0. the head on the stored v2 features (reproduces the 89.2% of single_H1/test_metrics.json)
    d = torch.load(Path.home() / "work" / "mega_feats_ft" / "test.pt", map_location="cpu", weights_only=False)
    ps = []
    for i in range(0, len(d["labels"]), 512):
        ps.append(head(d["feats"][i:i + 512].float().cuda()).softmax(1)[:, 1].double().cpu())
    res["head_on_stored_features"] = metrics(d["labels"].numpy(), torch.cat(ps).numpy())
    print("stored feats", res["head_on_stored_features"], flush=True)

    net = load_trunk_net()
    u8, y = load_cache("bicubic", None, 256)
    print("images", len(u8), flush=True)
    ref = RefModel(net, head)
    p_ref = probs_of(lambda x: ref(x[:, None]), None, u8)
    res["ref_fp32_end_to_end"] = metrics(y, p_ref)
    print("ref", res["ref_fp32_end_to_end"], flush=True)

    k32 = FusedHeadOnFp32Trunk(net, head, AB)
    res["fp32_trunk_plus_fused_kernels_A_B_fp32"] = compare(probs_of(k32.step, k32.reset, u8), p_ref, y)
    print("A+B fp32", res["fp32_trunk_plus_fused_kernels_A_B_fp32"], flush=True)
    del k32

    fs = FusedStream(net, head, AB)
    res["fused_path_fp16"] = compare(probs_of(fs.step, fs.reset, u8), p_ref, y)
    print("fused fp16", res["fused_path_fp16"], flush=True)
    del fs
    tb = TrunkFusedTorchHead(net, head, AB)
    res["fused_trunk_torch_head_fp16"] = compare(probs_of(tb.step, tb.reset, u8), p_ref, y)
    print("fused trunk + torch head fp16", res["fused_trunk_torch_head_fp16"], flush=True)
    del tb
    un = UnfusedStream(net, head, AB, F16)
    p_un = probs_of(un.step, un.reset, u8)
    res["unfused_pytorch_fp16"] = compare(p_un, p_ref, y)
    print("unfused fp16", res["unfused_pytorch_fp16"], flush=True)
    del un
    torch.cuda.empty_cache()

    # 30-frame streams: state carried across frames, compared with the parallel head over the same prefix
    rng = np.random.default_rng(0)
    S, T = 4, 30
    idx = rng.choice(len(u8), S * T, replace=False)
    frames = prep(u8[np.sort(idx)]).view(S, T, 3, 256, 256)
    seq = {}
    for tag, mk in (("fused_kernels_A_B_fp32_on_fp32_trunk", lambda: FusedHeadOnFp32Trunk(net, head, S)),
                    ("fused_path_fp16", lambda: FusedStream(net, head, S))):
        st = mk()
        st.reset()
        diffs = []
        for t in range(T):
            out = st.step(frames[:, t].contiguous()).float()
            if t + 1 in (1, 10, 30):
                r = ref(frames[:, :t + 1]).float()
                diffs.append({"frames": t + 1, "max_abs_logit_diff_vs_parallel_fp32": float((out - r).abs().max()),
                              "max_abs_logit": float(r.abs().max())})
        seq[tag] = diffs
        print(tag, diffs, flush=True)
        del st
    res["stream_30_frames_vs_parallel"] = {"streams": S, "frames": T, "checks": seq}
    res["n_images"] = int(len(y))
    res["wall_s"] = round(time.time() - t0, 1)
    (HERE / "correctness.json").write_text(json.dumps(res, indent=1))
    print("done", res["wall_s"], "s")


if __name__ == "__main__":
    main()
