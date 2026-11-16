# How the official selective-scan kernel was built

The official `mamba-ssm` kernel (commit e9594ce) was compiled from `csrc/selective_scan/` with PyTorch's `torch.utils.cpp_extension.load`, as extension name `selective_scan_cuda`, into `~/sota/mamba_build`. The command and flags below come from the build script and from the compiler line in the build log (`~/sota/build.log`, 11 files compiled).

Environment: `CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.9`.

Sources: `selective_scan.cpp`, `selective_scan_fwd_{fp32,fp16,bf16}.cu`, and the backward files.

Host compiler flags: `-O3 -std=c++20`.

nvcc flags: `-O3 -std=c++20 -gencode=arch=compute_89,code=sm_89 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -lineinfo`, plus the `-U__CUDA_NO_HALF_OPERATORS__`, `-U__CUDA_NO_HALF_CONVERSIONS__`, `-U__CUDA_NO_BFLOAT16_OPERATORS__`, `-U__CUDA_NO_BFLOAT16_CONVERSIONS__`, `-U__CUDA_NO_BFLOAT162_OPERATORS__` and `-U__CUDA_NO_BFLOAT162_CONVERSIONS__` flags.

So the official kernel is built optimised, for this GPU's architecture (sm_89), with fast math. `--use_fast_math` lets the compiler use faster, slightly less exact arithmetic.

This record was added after the benchmark run. The build files carry a timestamp of 11 November, 16:20, and `results/official_compare.json` does not store the flags in its `meta`; adding them there is pending.
