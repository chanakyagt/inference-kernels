#!/bin/bash
# usage (WSL): bash run.sh script.py [args]   -- runs with the trt-env python from trained/
export DFM_CKPT_ROOT=/mnt/c/Users/chana/Projects/_artifacts DFM_SCRATCH=$HOME/trt_scratch TRITON_CACHE_AUTOTUNING=1
export PYTHONPATH=${EXTRA_PP:+$EXTRA_PP:}/mnt/c/Users/chana/Projects/inference-kernels/fused_model:/mnt/c/Users/chana/Projects/inference-kernels/trained
cd /mnt/c/Users/chana/Projects/inference-kernels/trained
exec ~/trt-env/bin/python "$@"
