# Fused EfficientNet-B0 v2 + Mamba head: streaming path (15 Nov 2026)

One per-frame video path for the fine-tuned detector:
- Trunk: the fused v2 EfficientNet-B0 kernels (`fused_model/fmodel2.py`, own Triton: stem, expand GEMM + bias + SiLU, NHWC depthwise with SE-pool partial sums, SE tail, project GEMM with SE-gate prologue and residual epilogue), fp16.
- Head: the v2 Mamba head (`deepfake-detector-mamba/results/images_megadataset_ft/single_H1/model.pt`, d_model 512, 4 Mamba layers), trained on frozen v2 trunk features.
- Hand-off: the 11 Nov fused kernels A (head conv + SiLU + pool + projection) and B (4 Mamba recurrent steps + final norm + mean + MLP), now fed by the v2 trunk.

Mamba state is carried across frames (constant per stream); each Test image is a stream of length 1, as in training.

Files: `model_v2.py` (paths), `check.py` -> `correctness.json`, `bench.py` -> `bench.json`, `bench.log`. Run in WSL: `bash run.sh check.py`, `bash run.sh bench.py`.

## Correctness (all 10,905 Test images; `correctness.json`)

| Path | Acc | AUC | Flips vs fp32 reference | Max prob. difference |
|---|---|---|---|---|
| fp32 reference (unfused, end to end) | 89.10% | 0.9368 | - | - |
| fp32 trunk + fused kernels A, B (fp32) | 89.10% | 0.9368 | 0 | 5.1e-7 |
| Fused path fp16 (the timed path) | 89.12% | 0.9370 | 6 | 0.065 |
| Unfused PyTorch fp16 | 89.10% | 0.9369 | 4 | 0.110 |
| Head on stored features (reproduces single_H1) | 89.15% | 0.9363 | - | - |

The end-to-end 89.10% differs from the stored-feature 89.15% (and the 89.2% in single_H1) by image preprocessing at feature extraction, not by the kernels (the fp32 fused path matches the fp32 reference exactly).
Streaming check (4 streams x 30 frames, recurrent vs parallel scan): fp32 fused max logit difference 1.4e-6; fp16 fused 0.014.

## Timing (one quiet session, RTX 4060 Ti, fp16, CUDA graph per frame, best of 3 trials of a 200-frame median; `bench.json`)

Idle gate before the run: passed (4/1/0%). All 10 rows stable (trial spread <= 1.095). The end-of-run idle check read 77% on its first sample (the benchmark's own teardown) and is not a sign of another job.

Per new frame, ms:

| Method | Batch 1 | Batch 8 | GPU kernels per frame (b1 / b8) | State per stream |
|---|---|---|---|---|
| a. Unfused PyTorch trunk + PyTorch Mamba step | 1.030 | 3.050 | 504 / 507 | 289 KB |
| b. Fused v2 trunk + PyTorch Mamba step | 0.638 | 1.543 | 217 / 205 | 289 KB |
| **New: fused v2 trunk + kernels A, B** | **0.527** | **1.415** | **94 / 77** | 313 KB |
| c1. 30-frame window, Mamba recomputed with fused scan | 0.722 | 1.680 | 193 / 170 | 77 KB (window) |
| c2. 30-frame window attention, RANDOM weights | 0.423 | 1.355 | 101 / 84 | 77 KB (window) |

- New vs unfused (a): 1.96x at batch 1, 2.15x at batch 8.
- New vs fused trunk + PyTorch Mamba (b): 1.21x at batch 1, 1.09x at batch 8.
- **The fused Mamba path is not the fastest option.** The window-attention baseline (c2, untrained random weights, so its accuracy is unknown) is 1.25x faster at batch 1 and 1.04x faster at batch 8. Its cost grows with the window (fixed here at 30 frames); the Mamba path's per-frame cost and state do not.

Stage times inside a CUDA graph (ms):

| Stage | Batch 1 | Batch 8 |
|---|---|---|
| Unfused trunk incl. pool | 0.752 | 2.634 |
| Fused v2 trunk | 0.344 | 1.089 |
| After the trunk, PyTorch (head conv + mean + proj + Mamba step) | 0.288 (137 kernels) | 0.323 (141 kernels) |
| After the trunk, fused A + B | 0.066 (11 kernels) | 0.178 (11 kernels) |
| of which kernel A / kernel B | 0.007 / 0.058 | 0.020 / 0.153 |

After-trunk stage: 4.4x faster at batch 1, 1.8x at batch 8; 137 -> 11 kernels.

## Supported claim

On every video frame, fusing the hand-off from EfficientNet into the Mamba head cut the stage after the backbone 4.4 times (137 GPU kernels to 11); with the fused backbone, the whole detector ran at 0.53 ms per frame, 2.0x faster than unfused PyTorch, with fp32 outputs unchanged. Do not claim it is the fastest streaming option (the untrained window-attention baseline is faster).

## DRAM traffic per frame, Nsight Compute counters (15 Nov 2026; `ncu_frame.py`, `ncu_frame.sh`, `ncu_summarize.py`, `ncu_results/`, `ncu_summary.json`, `ncu_after_trunk.json`)

One new frame per run, after 5 warm-up frames, fp16, eager (the same kernels the CUDA-graph benchmark launches), base clocks, one run per point. Metrics: dram__bytes_read.sum + dram__bytes_write.sum, summed over every kernel in the frame.

**Quoted mode: `--cache-control all`** (L2 flushed before every kernel). Each kernel's count is the data it must fetch from DRAM with nothing cached, so the sum is the frame's compulsory traffic when no kernel can reuse another's data from L2. A real stream with a warm 32 MB L2 moves less.

| Path | Batch 1 kernels / MB per frame | Batch 8 kernels / MB per frame |
|---|---|---|
| Unfused PyTorch | 507 / 111.0 | 507 / 612.5 |
| Fused v2 trunk + PyTorch Mamba | 201 / 57.9 | 205 / 217.8 |
| **Fused v2 trunk + kernels A, B** | **75 / 53.4** | **75 / 196.7** |

- Whole frame, fused vs unfused: 2.1x less DRAM traffic at batch 1, 3.1x at batch 8.
- The trunk kernels are identical in the last two rows (same 64 kernels, 28 MB at b1, 168-169 MB at b8). After the trunk: PyTorch 137 kernels / 30.0 MB vs fused A+B 11 kernels / 25.1 MB at batch 1 (-16%); 141 kernels / 48.8 MB vs 11 / 28.3 MB at batch 8 (-42%).
- At batch 1 the after-trunk bytes are close to the floor set by the head's weights (Mamba head 11,503,874 parameters = 23.0 MB in fp16, so fused A+B moves about 2 MB beyond its weights; PyTorch about 7 MB), so the 4.4x after-trunk speed-up there comes mostly from 137 -> 11 launches, not from bytes; at batch 8 the removed activation traffic is the larger share.

**Not quoted: `--cache-control none`** (L2 warm). Totals are inconsistent across paths (for example fused trunk + PyTorch Mamba 24.2 MB vs fused A+B 52.2 MB at batch 1, with 18 MB of writes on one and 0.4 MB on the other): write-backs of dirty L2 lines are counted on whichever later kernel evicts them, so per-frame sums depend on what happened to be in L2. Raw files are kept in `ncu_results/*_none.csv`.
