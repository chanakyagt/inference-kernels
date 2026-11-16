import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bp in (8, 16, 32, 64):
        for bc in (32, 64, 128):
            out.append(triton.Config({"BP": bp, "BC": bc}, num_warps=4 if bp * bc >= 2048 else 2))
    return out


def _prune(configs, named_args, **kwargs):
    cap = max(8, triton.next_power_of_2(named_args["P"]))
    keep = [c for c in configs if c.kwargs["BP"] <= cap]
    return keep or configs[:1]


@triton.autotune(configs=_configs(), key=["C", "P", "K", "S", "ACT", "IS_Q"], warmup=5, rep=20, prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _dw_nhwc(x_ptr, w_ptr, ws_ptr, b_ptr, y_ptr, sx,
             C, H, W, Ho, Wo, P,
             K: tl.constexpr, S: tl.constexpr, PAD: tl.constexpr, ACT: tl.constexpr, IS_Q: tl.constexpr,
             BP: tl.constexpr, BC: tl.constexpr):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)
    p = pid_p * BP + tl.arange(0, BP)
    c = pid_c * BC + tl.arange(0, BC)
    pm = p < P
    cm = c < C
    n = p // (Ho * Wo)
    r = p % (Ho * Wo)
    oh = r // Wo
    ow = r % Wo
    acc = tl.zeros((BP, BC), dtype=tl.float32)
    for kh in tl.static_range(K):
        ih = oh * S - PAD + kh
        mh = pm & (ih >= 0) & (ih < H)
        for kw in tl.static_range(K):
            iw = ow * S - PAD + kw
            mm = mh & (iw >= 0) & (iw < W)
            base = ((n * H + ih) * W + iw) * C
            v = tl.load(x_ptr + base[:, None] + c[None, :], mask=mm[:, None] & cm[None, :], other=0).to(tl.float32)
            wv = tl.load(w_ptr + (kh * K + kw) * C + c, mask=cm, other=0).to(tl.float32)
            acc += v * wv[None, :]
    ws = tl.load(ws_ptr + c, mask=cm, other=0.0)
    b = tl.load(b_ptr + c, mask=cm, other=0.0)
    acc = acc * (ws * sx)[None, :] + b[None, :]
    if ACT:
        acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + p[:, None] * C + c[None, :], acc.to(y_ptr.dtype.element_ty), mask=pm[:, None] & cm[None, :])


def quantize_dw_weight(w):
    w = w.float().reshape(w.shape[0], -1)
    s = w.abs().amax(1).clamp(min=1e-12) / 127
    q = torch.round(w / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q.t().contiguous(), s


def int8_dwconv_nhwc(x_q, sx, w_q, w_scale, bias, stride, padding, k, act=True, out_dtype=torch.bfloat16):
    N, H, W, C = x_q.shape
    Ho = (H + 2 * padding - k) // stride + 1
    Wo = (W + 2 * padding - k) // stride + 1
    y = torch.empty((N, Ho, Wo, C), device=x_q.device, dtype=out_dtype)
    P = N * Ho * Wo
    grid = lambda meta: (triton.cdiv(P, meta["BP"]), triton.cdiv(C, meta["BC"]))
    _dw_nhwc[grid](x_q, w_q, w_scale, bias, y, float(sx), C, H, W, Ho, Wo, P, K=k, S=stride, PAD=padding, ACT=act, IS_Q=x_q.dtype == torch.int8)
    return y


def bf16_dwconv_nhwc(x, w, bias, stride, padding, k, act=True):
    N, H, W, C = x.shape
    ones = torch.ones(C, device=x.device, dtype=torch.float32)
    return int8_dwconv_nhwc(x, 1.0, w, ones, bias, stride, padding, k, act, out_dtype=x.dtype)


class Int8DWConv(torch.nn.Module):
    def __init__(self, conv, act_amax, act=True):
        super().__init__()
        wq, ws = quantize_dw_weight(conv.weight)
        self.register_buffer("wq", wq)
        self.register_buffer("ws", ws)
        self.register_buffer("bias", conv.bias.detach().float().contiguous())
        self.sx = float(act_amax) / 127
        self.k = conv.kernel_size[0]
        self.stride = conv.stride[0]
        self.padding = conv.padding[0]
        self.act = act

    def forward_q(self, xq):
        return int8_dwconv_nhwc(xq, self.sx, self.wq, self.ws, self.bias, self.stride, self.padding, self.k, self.act)

    def forward(self, x):
        xq = torch.round(x.permute(0, 2, 3, 1).float() / self.sx).clamp(-127, 127).to(torch.int8).contiguous()
        return self.forward_q(xq).permute(0, 3, 1, 2)
