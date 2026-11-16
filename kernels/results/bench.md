# Kernel benchmarks (NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130, bfloat16, median of repeated runs)

## Depthwise conv + BatchNorm + SiLU, the 16 EfficientNet-B0 depthwise blocks at 30 frames

Useful bytes = input + output + weights, read or written once. Achieved bandwidth = useful bytes / time.

| Input (N,C,H,W) | k | s | PyTorch eager ms | torch.compile ms | Fused Triton ms | Speedup vs eager | Eager GB/s | Fused GB/s | Fused % of 288 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| 30x32x112x112 | 3 | 1 | 0.828 | 0.584 | 0.219 | 3.78x | 58 | 220 | 76% |
| 30x96x112x112 | 3 | 2 | 0.644 | 1.093 | 0.392 | 1.64x | 140 | 230 | 80% |
| 30x144x56x56 | 3 | 1 | 0.907 | 0.700 | 0.300 | 3.03x | 60 | 181 | 63% |
| 30x144x56x56 | 5 | 2 | 0.270 | 1.880 | 0.152 | 1.78x | 125 | 224 | 78% |
| 30x240x28x28 | 5 | 1 | 0.378 | 0.153 | 0.220 | 1.72x | 60 | 103 | 36% |
| 30x240x28x28 | 3 | 2 | 0.103 | 0.124 | 0.065 | 1.60x | 137 | 219 | 76% |
| 30x480x14x14 | 3 | 1 | 0.142 | 0.121 | 0.090 | 1.59x | 79 | 126 | 44% |
| 30x480x14x14 | 3 | 1 | 0.154 | 0.114 | 0.091 | 1.69x | 73 | 124 | 43% |
| 30x480x14x14 | 5 | 1 | 0.201 | 0.134 | 0.135 | 1.48x | 56 | 84 | 29% |
| 30x672x14x14 | 5 | 1 | 0.269 | 0.137 | 0.181 | 1.49x | 59 | 88 | 30% |
| 30x672x14x14 | 5 | 1 | 0.277 | 0.133 | 0.182 | 1.52x | 57 | 87 | 30% |
| 30x672x14x14 | 5 | 2 | 0.105 | 0.387 | 0.076 | 1.39x | 94 | 131 | 46% |
| 30x1152x7x7 | 5 | 1 | 0.158 | 0.161 | 0.106 | 1.49x | 43 | 65 | 23% |
| 30x1152x7x7 | 5 | 1 | 0.160 | 0.141 | 0.107 | 1.49x | 43 | 64 | 22% |
| 30x1152x7x7 | 5 | 1 | 0.158 | 0.139 | 0.098 | 1.60x | 44 | 70 | 24% |
| 30x1152x7x7 | 3 | 1 | 0.118 | 0.156 | 0.074 | 1.60x | 58 | 92 | 32% |

Total over 16 blocks: eager 4.87 ms, torch.compile 6.16 ms, fused 2.49 ms (1.96x vs eager).

## Selective scan, one Mamba layer (d_inner 1024, d_state 16)

| Batch | Frames | PyTorch loop ms | Fused Triton ms | Speedup | Loop: (B,L,E,N) tensors MB | Fused: MB read+written |
|---|---|---|---|---|---|---|
| 1 | 30 | 4.018 | 0.302 | 13.3x | 7.9 | 0.37 |
| 4 | 30 | 4.417 | 0.338 | 13.1x | 31.5 | 1.49 |
| 1 | 300 | 40.263 | 0.420 | 95.9x | 78.6 | 3.72 |
| 1 | 3000 | 399.713 | 3.440 | 116.2x | 786.4 | 37.25 |

## Whole detector forward (EfficientNet-B0 + 4 Mamba layers + head), 30 frames per clip

| Clips | Variant | ms | Clips/s |
|---|---|---|---|
| 1 | baseline | 22.58 | 44.3 |
| 1 | baseline + CUDA graph | 11.27 | 88.7 |
| 1 | fused dwconv | 23.12 | 43.3 |
| 1 | fused scan | 11.26 | 88.8 |
| 1 | both | 9.62 | 104.0 |
| 1 | both + CUDA graph | 7.98 | 125.3 |
| 4 | baseline | 57.21 | 69.9 |
| 4 | baseline + CUDA graph | 50.90 | 78.6 |
| 4 | fused dwconv | 45.20 | 88.5 |
| 4 | fused scan | 50.75 | 78.8 |
| 4 | both | 39.07 | 102.4 |
| 4 | both + CUDA graph | 37.96 | 105.4 |
