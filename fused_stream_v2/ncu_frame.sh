#!/bin/bash
# Nsight Compute DRAM bytes per new frame: unfused vs fused trunk + PyTorch Mamba vs fused A+B; batch 1 and 8.
# cache-control none = L2 kept warm between kernels (as in a real stream); all = L2 flushed before every kernel.
export DFM_CKPT_ROOT=/mnt/c/Users/chana/Projects/_artifacts DFM_SCRATCH=$HOME/trt_scratch TRITON_CACHE_AUTOTUNING=1
cd /mnt/c/Users/chana/Projects/inference-kernels/fused_stream_v2
OUT=ncu_results; mkdir -p $OUT
M=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum
for cc in none all; do for B in 1 8; do for m in unfused fused_trunk_torch_mamba fused_AB; do
  timeout 1200 /usr/local/cuda/bin/ncu --profile-from-start off --cache-control $cc --clock-control base --metrics $M --csv \
    ~/trt-env/bin/python ncu_frame.py $m $B > $OUT/${m}_B${B}_${cc}.csv 2> $OUT/${m}_B${B}_${cc}.err
  echo "$cc B$B $m rc=$?"
done; done; done
