import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
os.environ["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), os.environ["CUDA_HOME"] + "/bin", os.environ["PATH"]])

_ext = None


def load(verbose=False):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load as _load
        _ext = _load(name="selective_scan_cuda", sources=[str(HERE / "selective_scan.cu")],
                     extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++20"],
                     extra_cflags=["-O3", "-std=c++20"], verbose=verbose)
    return _ext


DEFAULT_CONFIG = (32, 2)


def scan_cuda(x, delta, A, Bm, Cm, D, cpb=None, s=None):
    cpb = cpb or DEFAULT_CONFIG[0]
    s = s or DEFAULT_CONFIG[1]
    return load().scan(x.contiguous(), delta.contiguous(), A.float().contiguous(), Bm.contiguous(),
                       Cm.contiguous(), D.float().contiguous(), cpb, s)


if __name__ == "__main__":
    load(verbose=True)
    print("built")
