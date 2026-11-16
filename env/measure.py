import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import bench


def main():
    n = 256 * 2**20
    a = torch.empty(n, dtype=torch.float32, device="cuda").uniform_()
    b = torch.empty_like(a)
    copy_ms = bench(lambda: b.copy_(a), iters=30)
    read_ms = bench(lambda: a.sum(), iters=30)
    m = 8192
    x = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    y = torch.randn(m, m, dtype=torch.bfloat16, device="cuda")
    mm_ms = bench(lambda: x @ y, iters=10)
    x8 = x.to(torch.float8_e4m3fn)
    y8 = y.t().contiguous().to(torch.float8_e4m3fn).t()
    one = torch.ones((), device="cuda")
    f8_ms = bench(lambda: torch._scaled_mm(x8, y8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16), iters=10)
    p = torch.cuda.get_device_properties(0)
    res = {"gpu": p.name, "sms": p.multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
           "copy_GBps": round(2 * a.numel() * 4 / copy_ms / 1e6, 1),
           "read_GBps": round(a.numel() * 4 / read_ms / 1e6, 1),
           "bf16_dense_TFLOPs": round(2 * m**3 / mm_ms / 1e9, 1),
           "fp8_dense_TFLOPs": round(2 * m**3 / f8_ms / 1e9, 1)}
    try:
        import triton
        res["triton"] = triton.__version__
    except ImportError:
        pass
    print(json.dumps(res, indent=2))
    (Path(__file__).resolve().parent / "ceilings.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
