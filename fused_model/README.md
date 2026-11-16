# Fused fp16 EfficientNet-B0 (own Triton kernels) vs TensorRT fp16

Status: FINAL numbers are in the section "FINAL quiet same-session run" at the end (bench_quiet.json): quote only those. The earlier tables below were taken while other jobs were running and are kept for history only. 13 Nov 2026, RTX 4060 Ti, WSL2. Earlier raw data: `bench_all.json`, `bench_v2.json`.

## What was fused (MEASURED to run; fmodel.py)
- Expand 1x1: Triton GEMM, bias + SiLU in the epilogue (per-shape autotune of BM/BN/BK/warps/stages).
- Depthwise: existing Triton NHWC kernel (kernels/fused_dwconv_nhwc.py), called on NHWC directly (no permutes).
- SE: still PyTorch (mean + 2 small linears).
- Project 1x1: Triton GEMM, SE gate multiplied in while loading A, bias + residual add in the epilogue.
- Stem, head conv, fc: stem = cuDNN + PyTorch SiLU; head conv = same GEMM path with cuDNN fallback.
- Each 1x1 layer is timed against the unfused cuDNN path at build time and falls back if Triton is slower. At b64 only the s2b0 project fell back (see `decisions_b64` in the JSON).
- Not done: SE-pool in the depthwise epilogue, stem epilogue, whole SE tail kernel.

## Correctness (MEASURED)
Per block vs unfused R6f block on real Test activations (b64): max abs diff 0.03-0.25 with activation max 28-270 (fp16 rounding level). Whole model on all 10,905 Test images below.

## Results (same session, TensorRT rebuilt at builder_optimization_level=5)
| Version | Acc / AUC (10,905 Test) | flips vs fp32 R0s | lat b1 / b8 / b64 (ms) | img/s sustained b1 / b64 |
|---|---|---|---|---|
| Fused (own) | 0.8961 / 0.9588 | 1 | 0.472 / 1.290 / 12.71 | 2,048 / 5,025 |
| R6f (own, unfused) | 0.8962 / 0.9588 | 0 | 0.622 / 1.927 / 22.02 | 1,622 / 2,828 |
| TensorRT fp16, opt level 5 | 0.8960 / 0.9587 | 2 | 0.591 / 1.319 / 16.57 | 1,380 / 3,890 |

- Fused vs TensorRT (this session): 1.25x faster at b1, 1.02x at b8, 1.30x at b64 (1.29x sustained).
- Caveat: the earlier quiet-machine TensorRT fp16 (default level) run measured 0.413 / 1.123 / 15.06 ms. TensorRT in this session is slower than that, probably CPU contention and/or level-5 tactic selection; against the earlier numbers the fused model would win only at b64 (12.71 vs 15.06) and lose at b1 (0.472 vs 0.413). The quiet-machine re-run decides b1.
- J/img not measured in this pass (power fields are in the JSON if the sustained runs were clean; check `idle_clean`).

## Commands (WSL; from Git Bash set MSYS_NO_PATHCONV=1)
    bash run.sh $PWD/check_blocks.py 64
    bash run.sh $PWD/bench_all.py --seconds 20
`FM_FORCE_IMPL=triton|cudnn` forces the 1x1 implementation for ablations. First run autotunes for several minutes (TRITON_CACHE_AUTOTUNING=1 caches it).

## v2: batch-1 fusion (fmodel2.py) - PARTIAL, PRELIMINARY (WSL died mid-run, see end)
b1 profile of v1 (MEASURED, CUDA graph): 141 kernels, 402 us summed kernel time vs 472 us graph latency. SE chain (reduce, 2 gemv, silu, sigmoid) = 163 us of it; pw_kernel 162 us; dw 56 us.
v2 (FusedRunner2, maker `fused2` in bench_all.py; v1 untouched): dw kernel writes SE-pool partial sums; one SE-tail kernel per image; stem as one Triton kernel (fp32 NCHW in, bias+SiLU, NHWC out); mean+fc head kernel. Each piece is timed against the old path at build and falls back if slower (FM2_FORCE=new|old for ablations).
Block-level accuracy (check_blocks2.py, b64, forced new, MEASURED): stem max diff 0.125 (act max 177); blocks 0.03-0.25 (same as v1); SE gate max diff <=1e-3; head logits 1e-3; whole-model logits 5e-3 (max logit 1.7). Whole-model 10,905-image flips NOT yet run.
b1 result with per-piece fallback: 96 kernels (was 141), summed kernel time 348 us (was 402). The fused SE tail lost to the old chain in 7 of 16 blocks (the 7x7 stages and headtail: new 11-21 us vs old 11-15 us per block incl. dw) because one program per image is latency-bound (~5 us); the 3 best-case micro-benchmarks give 2.4-3.2 us vs 9.6-13 us old chain at larger-map shapes. Graph latency of v2 not measured.
Next: bench_se.py (config sweep, includes 256/512 BC) was cut off; re-run `bench_all.py --only fused2 --seconds 5 --out bench_v2.json`.

## TensorRT INT8 via ModelOpt - PARTIAL
Full `uv pip install nvidia-modelopt[onnx]` dry-run: 60 packages incl. torch 2.15, CUDA 13 libs, onnxruntime-gpu = several GB (> 300 MB), NOT installed into trt-env. Instead installed modelopt 0.48.0 + small deps with --no-deps into ~/mo_pkgs (165 MB on disk, <100 MB download; used via EXTRA_PP=$HOME/mo_pkgs bash run.sh ...). Import works. trt_int8.py (stages quant / bench) written; calibration = 128 Validation images, entropy, fp16 high-precision. Quantization of the b64 ONNX started (ModelOpt found 122 nodes to quantize) and the WSL VM then died (free showed 15.2 of 15.8 GB used: ModelOpt histogram calibration + other jobs). No INT8 accuracy or latency exists yet.

## v2 results after WSL restart (MEASURED, PRELIMINARY - machine shared with other jobs; raw: bench_v2.json)
| Version | Acc / AUC (10,905 Test) | flips vs R0s | lat b1 / b8 / b64 (ms) | img/s sustained b1 / b64 |
|---|---|---|---|---|
| Fused v2 | 0.8961 / 0.9588 | 1 | 0.361 / 1.120 / 10.17 | 2,874 / 6,305 |
| Fused v1 (earlier, busy session) | 0.8961 / 0.9588 | 1 | 0.472 / 1.290 / 12.71 | 2,048 / 5,025 |
| TensorRT fp16 default, earlier quiet | - | - | 0.413 / 1.123 / 15.06 | - |
SE-tail config sweep widened (BC up to 256): no change (0.361 b1 both runs), so autotune already finds the best; hard-coding not needed. v1 vs v2 were not run back-to-back, so the v1 column is not a controlled comparison. INT8 quant not started (waiting for quiet machine).

## FINAL quiet same-session run (13 Nov 2026, 17:4x; idle gate passed 1/1/2 %; all sustained runs stable + clean idle; raw: bench_quiet.json)
| Version | Acc / AUC | flips vs R0s | lat b1 / b8 / b64 (ms) | img/s sustained b1 / b64 | J/img net b1 / b64 |
|---|---|---|---|---|---|
| **Fused v2 (own Triton)** | 0.8961 / 0.9588 | 1 | **0.344 / 1.114 / 10.17** | **2,810 / 6,007** | 0.0273 / 0.0245 |
| Fused v1 (own Triton) | 0.8961 / 0.9588 | 1 | 0.426 / 1.249 / 13.34 | 2,330 / 4,862 | 0.0306 / 0.0278 |
| TensorRT 11.4 fp16, default optimisation level | 0.8961 / 0.9588 | 1 | 0.407 / 1.122 / 15.35 | 2,356 / 4,171 | 0.0281 / 0.0255 |
| TensorRT 11.4 fp16, builder_optimization_level 5 | 0.8960 / 0.9587 | 2 | 0.559 / 1.275 / 16.03 | 1,774 / 3,988 | 0.0390 / 0.0288 |
| R6f (own, unfused) | 0.8962 / 0.9588 | 0 | 0.534 / 1.859 / 23.38 | 1,846 / 2,883 | 0.0407 / 0.0435 |

Fused v2 vs the best TensorRT (default level): 1.18x lower latency at b1, tie at b8 (1.01x), 1.51x at b64; 1.44x sustained throughput at b64, 1.19x at b1; energy per image equal within noise (0.0245 vs 0.0255 J at b64); accuracy identical (1 flip each). Level 5 was slower than the default level on this model. Latency = best of 3 trials of a 50-200-iteration median; TensorRT default b1 trials were 0.629 / 0.632 / 0.407 ms (best used, as for all rows).
