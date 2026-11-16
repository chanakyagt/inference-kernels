"""One streaming frame under Nsight Compute: DRAM bytes and kernel time per new frame.
usage (WSL, under ncu --profile-from-start off): python ncu_frame.py <method> <batch>
The model is built and warmed (5 frames) outside the profiled range; exactly one frame (eager, same kernels as the
CUDA-graph benchmark) is profiled between cudaProfilerStart/Stop."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v2 import F16, FusedStream, TrunkFusedTorchHead, UnfusedStream, load_head, load_trunk_net

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

METHODS = {
    "unfused": lambda net, head, B: UnfusedStream(net, head, B, F16),
    "fused_trunk_torch_mamba": lambda net, head, B: TrunkFusedTorchHead(net, head, B),
    "fused_AB": lambda net, head, B: FusedStream(net, head, B),
}


@torch.no_grad()
def main():
    method, B = sys.argv[1], int(sys.argv[2])
    torch.manual_seed(0)
    net, head = load_trunk_net(), load_head()
    obj = METHODS[method](net, head, B)
    x = torch.randn(B, 3, 256, 256, device="cuda")
    obj.reset()
    for _ in range(5):
        obj.step(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    obj.step(x)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("done", method, B, flush=True)


if __name__ == "__main__":
    main()
