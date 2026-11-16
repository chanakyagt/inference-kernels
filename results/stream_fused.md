# Fused CNN-to-SSM streaming path: measurements

GPU NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130. Idle check before timing (utilisation percent, 3 samples 5 s apart): [5.0, 5.0, 5.0]. Start: SM clock 225 MHz, memory clock 405 MHz, 39 C, 9.60 W. End: SM clock 2805 MHz, 58 C, 79.02 W. Latency is wall clock per new frame including a device sync; the input image is already preprocessed and on the GPU.

## diloco_bf16_B1

| Method | Eager ms (p10-p90) | CUDA graph ms (p10-p90) | GPU kernels per frame | CPU kernel launches (eager) | Peak MB eager | Peak MB graph | State bytes per stream |
|---|---|---|---|---|---|---|---|
| Fused path (kernels A + B) | 5.527 (4.882-7.153) | 0.904 (0.831-0.967) | 313 | 313 | 96 | 92 | 313348 |
| Baseline 1: unfused PyTorch recurrent step | 9.571 (8.607-11.522) | 1.097 (1.068-1.212) | 506 | 506 | 104 | 116 | 288768 |
| Baseline 2a: window head recomputed with fused scan, trunk once per frame | 9.338 (8.596-10.262) | 1.177 (1.051-1.549) | 411 | 411 | 112 | 124 | 76800 |
| Baseline 3: 1-layer attention over the 30-frame window, trunk once per frame | 8.656 (7.865-9.999) | 1.328 (0.707-1.447) | 326 | 326 | 128 | 139 | 76800 |
| Baseline 2b: trunk recomputed on all 30 frames, then fused-scan head | 11.531 (10.637-12.615) | 8.409 (8.203-8.510) | 437 | 437 | 282 | 149 | 9108480 |

Stage times inside a CUDA graph (ms): trunk_fused_dw_ms 0.6017, trunk_unfused_ms 0.7425, tail_pool_proj_torch_ms 0.0200, kernel_A_ms 0.0081, mamba_stack_head_torch_ms 0.2656, kernel_B_ms 0.0593.

GPU kernels after the trunk: fused_A_plus_B 11, torch_tail_plus_step 140, window_fused_scan_head 102, window_attention_head 17.

Analytic DRAM MiB per frame (element size 2 B). Trunk: unfused 96.02, with fused depthwise kernel 78.67. After the trunk: unfused 29.52, fused 23.65 (weights plus state that must move anyway: 23.52; activation traffic: unfused 6.01, fused 0.14). Whole frame: unfused 125.54, fused 102.32; window baselines 2a 115.44, 2b 2441.83, 3 89.31.

## diloco_bf16_B8

| Method | Eager ms (p10-p90) | CUDA graph ms (p10-p90) | GPU kernels per frame | CPU kernel launches (eager) | Peak MB eager | Peak MB graph | State bytes per stream |
|---|---|---|---|---|---|---|---|
| Fused path (kernels A + B) | 9.118 (8.441-10.282) | 2.277 (2.238-2.896) | 330 | 330 | 194 | 155 | 313348 |
| Baseline 1: unfused PyTorch recurrent step | 11.737 (10.958-12.905) | 3.026 (2.985-3.666) | 529 | 529 | 193 | 172 | 288768 |
| Baseline 2a: window head recomputed with fused scan, trunk once per frame | 11.281 (10.637-12.444) | 2.611 (2.572-2.886) | 434 | 434 | 200 | 178 | 76800 |
| Baseline 3: 1-layer attention over the 30-frame window, trunk once per frame | 9.101 (8.390-10.272) | 2.156 (2.133-2.458) | 344 | 344 | 214 | 192 | 76800 |
| Baseline 2b: trunk recomputed on all 30 frames, then fused-scan head | 81.767 (81.577-81.928) | 80.756 (80.496-80.895) | 441 | 441 | 1441 | 263 | 9108480 |

Stage times inside a CUDA graph (ms): trunk_fused_dw_ms 1.9477, trunk_unfused_ms 2.5317, tail_pool_proj_torch_ms 0.0486, kernel_A_ms 0.0213, mamba_stack_head_torch_ms 0.3042, kernel_B_ms 0.1564.

GPU kernels after the trunk: fused_A_plus_B 11, torch_tail_plus_step 146, window_fused_scan_head 106, window_attention_head 16.

Analytic DRAM MiB per frame (element size 2 B). Trunk: unfused 718.59, with fused depthwise kernel 578.06. After the trunk: unfused 69.05, fused 28.80 (weights plus state that must move anyway: 27.70; activation traffic: unfused 41.35, fused 1.10). Whole frame: unfused 787.64, fused 606.86; window baselines 2a 720.08, 2b 17681.06, 3 608.51.

## diloco_fp32_B1

| Method | Eager ms (p10-p90) | CUDA graph ms (p10-p90) | GPU kernels per frame | CPU kernel launches (eager) | Peak MB eager | Peak MB graph | State bytes per stream |
|---|---|---|---|---|---|---|---|
| Fused path (kernels A + B) | 6.699 (6.342-7.749) | 1.812 (1.487-1.911) | 293 | 293 | 193 | 184 | 313348 |
| Baseline 1: unfused PyTorch recurrent step | 8.706 (8.110-9.693) | 1.744 (1.609-2.134) | 433 | 433 | 240 | 246 | 313344 |
| Baseline 2a: window head recomputed with fused scan, trunk once per frame | 8.862 (8.148-10.061) | 1.659 (1.623-2.111) | 383 | 383 | 248 | 254 | 153600 |
| Baseline 3: 1-layer attention over the 30-frame window, trunk once per frame | 6.876 (6.524-7.825) | 1.346 (1.330-1.477) | 310 | 310 | 268 | 275 | 153600 |
| Baseline 2b: trunk recomputed on all 30 frames, then fused-scan head | 17.578 (17.518-17.729) | 17.085 (16.997-17.183) | 365 | 365 | 571 | 288 | 18216960 |

Stage times inside a CUDA graph (ms): trunk_fused_dw_ms 1.1354, trunk_unfused_ms 1.2364, tail_pool_proj_torch_ms 0.0277, kernel_A_ms 0.0410, mamba_stack_head_torch_ms 0.3208, kernel_B_ms 0.2194.

GPU kernels after the trunk: fused_A_plus_B 11, torch_tail_plus_step 119, window_fused_scan_head 95, window_attention_head 22.

Analytic DRAM MiB per frame (element size 4 B). Trunk: unfused 191.74, with fused depthwise kernel 156.45. After the trunk: unfused 53.13, fused 46.22 (weights plus state that must move anyway: 46.05; activation traffic: unfused 7.08, fused 0.17). Whole frame: unfused 244.87, fused 202.66; window baselines 2a 221.11, 2b 4847.43, 3 177.71.

## diloco_fp32_B8

| Method | Eager ms (p10-p90) | CUDA graph ms (p10-p90) | GPU kernels per frame | CPU kernel launches (eager) | Peak MB eager | Peak MB graph | State bytes per stream |
|---|---|---|---|---|---|---|---|
| Fused path (kernels A + B) | 6.426 (6.048-7.247) | 4.178 (4.145-4.388) | 286 | 286 | 319 | 244 | 313348 |
| Baseline 1: unfused PyTorch recurrent step | 8.045 (7.704-8.817) | 4.808 (4.753-5.033) | 437 | 437 | 365 | 298 | 313344 |
| Baseline 2a: window head recomputed with fused scan, trunk once per frame | 8.203 (7.712-9.697) | 4.653 (4.616-4.890) | 379 | 379 | 364 | 297 | 153600 |
| Baseline 3: 1-layer attention over the 30-frame window, trunk once per frame | 6.557 (6.176-7.149) | 4.008 (3.969-4.235) | 300 | 300 | 377 | 310 | 153600 |
| Baseline 2b: trunk recomputed on all 30 frames, then fused-scan head | 161.555 (161.179-161.887) | 161.124 (160.714-161.679) | 370 | 370 | 2812 | 443 | 18216960 |

Stage times inside a CUDA graph (ms): trunk_fused_dw_ms 3.6011, trunk_unfused_ms 4.2118, tail_pool_proj_torch_ms 0.0839, kernel_A_ms 0.1650, mamba_stack_head_torch_ms 0.4464, kernel_B_ms 0.3205.

GPU kernels after the trunk: fused_A_plus_B 11, torch_tail_plus_step 130, window_fused_scan_head 98, window_attention_head 19.

Analytic DRAM MiB per frame (element size 4 B). Trunk: unfused 1436.89, with fused depthwise kernel 1155.23. After the trunk: unfused 99.75, fused 51.57 (weights plus state that must move anyway: 50.23; activation traffic: unfused 49.52, fused 1.34). Whole frame: unfused 1536.64, fused 1206.80; window baselines 2a 1377.27, 2b 35272.78, 3 1217.96.

## Correctness

### Kernel A (tail + pool + proj) against PyTorch fp32, real trunk maps, max abs error of the d_model output

| Checkpoint | fp32 two-kernel B=1 / 8 / 64 | bf16 two-kernel B=1 / 8 / 64 | PyTorch bf16 tail | output max abs |
|---|---|---|---|---|
| diloco | 9.54e-07 / 1.91e-06 / 3.81e-06 | 5.16e-03 / 8.62e-03 / 8.99e-03 | 1.09e-01 | 19.86 |

### Kernel B stepped over T frames against the full-sequence parallel scan, worst error over all steps

| Checkpoint | T | variant | hidden max abs | hidden max abs / max hidden | logit max abs |
|---|---|---|---|---|---|
| diloco | 30 | fp32_fuseout1 | 1.07e-02 | 1.07e-06 | 2.86e-06 |
| diloco | 30 | fp32_fuseout0 | 9.77e-03 | 9.73e-07 | 1.19e-06 |
| diloco | 30 | bf16_fuseout1 | 3.01e+01 | 3.00e-03 | 1.36e-02 |
| diloco | 30 | PyTorch bf16 full sequence (for scale) | 1.60e+02 | | 1.09e-01 |
| diloco | 300 | fp32_fuseout1 | 1.56e-02 | 6.99e-07 | 1.85e-06 |
| diloco | 300 | fp32_fuseout0 | 1.46e-02 | 6.55e-07 | 2.35e-06 |
| diloco | 300 | bf16_fuseout1 | 6.20e+01 | 2.77e-03 | 1.36e-02 |
| diloco | 300 | PyTorch bf16 full sequence (for scale) | 3.88e+02 | | 9.36e-02 |
| diloco | 3000 | fp32_fuseout1 | 4.49e-02 | 1.17e-06 | 9.36e-06 |
| diloco | 3000 | fp32_fuseout0 | 3.71e-02 | 9.67e-07 | 9.48e-06 |
| diloco | 3000 | bf16_fuseout1 | 1.77e+02 | 4.61e-03 | 1.36e-02 |
| diloco | 3000 | PyTorch bf16 full sequence (for scale) | 3.64e+02 | | 2.49e-01 |

### End to end, 200 real images as 1-frame sequences, and a 30-image sequence

| Checkpoint | Variant | logit max abs vs fp32 original | flips | note |
|---|---|---|---|---|
| diloco | orig_bf16_vs_orig_fp32 | 1.76e+00 | 3 | |
| diloco | fp32_tail_two_fuseout1_B1 | 6.96e-05 | 0 | |
| diloco | fp32_tail_atomic_fuseout1_B1 | 7.15e-05 | 0 | |
| diloco | fp32_tail_two_fuseout0_B1 | 7.10e-05 | 0 | |
| diloco | fp32_B8_batched | 1.08e-04 | 0 | |
| diloco | fp32_cuda_graph_B1 | 7.01e-05 | 0 | |
| diloco | bf16_tail_two_fuseout1_B1 | 1.57e+00 | 3 | |
| diloco | bf16_tail_atomic_fuseout1_B1 | 1.57e+00 | 3 | |
| diloco | bf16_tail_two_fuseout0_B1 | 1.57e+00 | 3 | |
| diloco | bf16_B8_batched | 1.41e+00 | 2 | |
| diloco | bf16_cuda_graph_B1 | 1.57e+00 | 3 | |
| diloco | seq30_fp32 | 3.70e-06 | 0 | |
| diloco | seq30_bf16 | 2.28e-01 | 1 | |
| diloco | bf16 original vs fp32 original | 1.76e+00 | 3 | the original's own bf16 drift |

### Full test set (10905 images, each a 1-frame sequence)

| Checkpoint | Pipeline | accuracy |
|---|---|---|
| diloco | orig_fp64 | 0.71444 (7791/10905) |
| diloco | orig_fp32 | 0.71444 (7791/10905) |
| diloco | orig_bf16_autocast | 0.71444 (7791/10905) |
| diloco | orig_bf16_pure | 0.71343 (7780/10905) |
| diloco | fused_fp32 | 0.71444 (7791/10905) |
| diloco | fused_bf16 | 0.71472 (7794/10905) |
| finetuned | orig_fp64 | 0.86208 (9401/10905) |
| finetuned | orig_fp32 | 0.86208 (9401/10905) |
| finetuned | orig_bf16_autocast | 0.86162 (9396/10905) |
| finetuned | orig_bf16_pure | 0.86236 (9404/10905) |
| finetuned | fused_fp32 | 0.86208 (9401/10905) |
| finetuned | fused_bf16 | 0.86226 (9403/10905) |

| Checkpoint | Comparison | prediction flips | logit max abs | logit mean abs |
|---|---|---|---|---|
| diloco | fused_fp32_vs_orig_fp32 | 0 | 0.0002 | 0.00000 |
| diloco | fused_bf16_vs_orig_bf16_pure | 68 | 2.4864 | 0.05249 |
| diloco | fused_bf16_vs_orig_bf16_autocast | 119 | 6.6699 | 0.11433 |
| diloco | fused_bf16_vs_orig_fp32 | 183 | 7.9273 | 0.15545 |
| diloco | orig_bf16_pure_vs_orig_fp32 | 179 | 7.9385 | 0.15577 |
| diloco | orig_bf16_autocast_vs_orig_fp32 | 136 | 7.5870 | 0.12402 |
| diloco | orig_fp32_vs_orig_fp64 | 0 | 0.0002 | 0.00000 |
| diloco | fused_fp32_vs_orig_fp64 | 0 | 0.0002 | 0.00000 |
| diloco | orig_bf16_pure_vs_orig_fp64 | 179 | 7.9385 | 0.15577 |
| diloco | fused_bf16_vs_orig_fp64 | 183 | 7.9273 | 0.15545 |
| finetuned | fused_fp32_vs_orig_fp32 | 0 | 0.0000 | 0.00000 |
| finetuned | fused_bf16_vs_orig_bf16_pure | 15 | 0.4413 | 0.02432 |
| finetuned | fused_bf16_vs_orig_bf16_autocast | 27 | 1.0878 | 0.04299 |
| finetuned | fused_bf16_vs_orig_fp32 | 48 | 0.9747 | 0.05687 |
| finetuned | orig_bf16_pure_vs_orig_fp32 | 41 | 1.0602 | 0.05939 |
| finetuned | orig_bf16_autocast_vs_orig_fp32 | 35 | 0.9135 | 0.04745 |
| finetuned | orig_fp32_vs_orig_fp64 | 0 | 0.0000 | 0.00000 |
| finetuned | fused_fp32_vs_orig_fp64 | 0 | 0.0000 | 0.00000 |
| finetuned | orig_bf16_pure_vs_orig_fp64 | 41 | 1.0602 | 0.05939 |
| finetuned | fused_bf16_vs_orig_fp64 | 48 | 0.9747 | 0.05687 |
