#!/bin/bash
# usage (WSL): bash run.sh check.py [args]  |  bash run.sh bench.py [args]
export DFM_CKPT_ROOT=/mnt/c/Users/chana/Projects/_artifacts DFM_SCRATCH=$HOME/trt_scratch TRITON_CACHE_AUTOTUNING=1
mkdir -p "$DFM_SCRATCH"
cd /mnt/c/Users/chana/Projects/inference-kernels/fused_stream_v2
exec ~/trt-env/bin/python "$@"
