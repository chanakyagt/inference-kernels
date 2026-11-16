# Per-layer profile: 4 clip(s) x 30 frames, bfloat16, NVIDIA GeForce RTX 4060 Ti

- End-to-end forward: **57.38 ms**; EfficientNet-B0 part alone: 49.76 ms
- Sum of every op timed in isolation: 68.08 ms (differs from end-to-end because isolated timing has no launch overlap and skips un-hooked ops such as residual adds and the squeeze-excitation multiply)
- Ceilings used: 288 GB/s, 45.1 TFLOP/s (measured BF16 tensor-core matmul), ridge 157 FLOP/byte

## Grouped by kind

| Section | Kind | Ops | Time (ms) | Share | GFLOP | MB moved | FLOP/byte | Achieved GB/s | Bound |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | selective scan (python loop) | 4 | 18.220 | 26.8% | 1.09 | 128.1 | 8.5 | 7 | memory |
| EfficientNet-B0 | BatchNorm2d | 49 | 13.830 | 20.3% | 3.23 | 3234.3 | 1.0 | 234 | memory |
| EfficientNet-B0 | SiLU | 49 | 9.720 | 14.3% | 2.96 | 2955.8 | 1.0 | 304 | memory |
| EfficientNet-B0 | conv 1x1 | 32 | 8.051 | 11.8% | 81.25 | 1666.2 | 48.8 | 207 | memory |
| EfficientNet-B0 | conv dw 3x3 | 7 | 6.577 | 9.7% | 3.12 | 944.6 | 3.3 | 144 | memory |
| EfficientNet-B0 | conv dw 5x5 | 9 | 5.908 | 8.7% | 5.16 | 518.5 | 10.0 | 88 | memory |
| EfficientNet-B0 | AdaptiveAvgPool2d | 17 | 1.956 | 2.9% | 0.28 | 571.1 | 0.5 | 292 | memory |
| EfficientNet-B0 | conv 1x1 (SE) | 32 | 1.558 | 2.3% | 0.15 | 5.8 | 26.2 | 4 | memory |
| Mamba | Linear | 16 | 0.774 | 1.1% | 2.55 | 28.2 | 90.5 | 36 | memory |
| EfficientNet-B0 | conv 3x3 | 1 | 0.640 | 0.9% | 2.60 | 132.5 | 19.6 | 207 | memory |
| EfficientNet-B0 | Sigmoid | 16 | 0.361 | 0.5% | 0.00 | 4.3 | 1.0 | 12 | memory |
| Mamba | Conv1d | 4 | 0.193 | 0.3% | 0.00 | 2.1 | 2.1 | 11 | memory |
| Mamba | LayerNorm | 4 | 0.128 | 0.2% | 0.00 | 1.0 | 1.0 | 8 | memory |
| Head | Linear | 2 | 0.070 | 0.1% | 0.00 | 0.3 | 3.9 | 4 | memory |
| Projection | Linear | 1 | 0.039 | 0.1% | 0.16 | 1.7 | 90.3 | 45 | memory |
| Head | LayerNorm | 1 | 0.034 | 0.0% | 0.00 | 0.2 | 1.0 | 7 | memory |
| Head | GELU | 1 | 0.024 | 0.0% | 0.00 | 0.0 | 0.2 | 0 | memory |

## Every op

| Op | Kind | Output shape | Time (ms) | GFLOP | MB moved | FLOP/byte | GB/s | % of peak BW |
|---|---|---|---|---|---|---|---|---|
| encoder.features.0.0 | conv 3x3 | 120x32x112x112 | 0.6400 | 2.601 | 132.47 | 19.6 | 207 | 72% |
| encoder.features.0.1 | BatchNorm2d | 120x32x112x112 | 0.7721 | 0.193 | 192.68 | 1.0 | 250 | 87% |
| encoder.features.0.2 | SiLU | 120x32x112x112 | 0.7803 | 0.193 | 192.68 | 1.0 | 247 | 86% |
| encoder.features.1.0.block.0.0 | conv dw 3x3 | 120x32x112x112 | 1.6886 | 0.867 | 192.68 | 4.5 | 114 | 40% |
| encoder.features.1.0.block.0.1 | BatchNorm2d | 120x32x112x112 | 0.7759 | 0.193 | 192.68 | 1.0 | 248 | 86% |
| encoder.features.1.0.block.0.2 | SiLU | 120x32x112x112 | 0.7772 | 0.193 | 192.68 | 1.0 | 248 | 86% |
| encoder.features.1.0.block.1.avgpool | AdaptiveAvgPool2d | 120x32x1x1 | 0.3584 | 0.048 | 96.35 | 0.5 | 269 | 93% |
| encoder.features.1.0.block.1.fc1 | conv 1x1 (SE) | 120x8x1x1 | 0.0454 | 0.000 | 0.01 | 6.1 | 0 | 0% |
| encoder.features.1.0.block.1.activation | SiLU | 120x8x1x1 | 0.0174 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.1.0.block.1.fc2 | conv 1x1 (SE) | 120x32x1x1 | 0.0399 | 0.000 | 0.01 | 6.0 | 0 | 0% |
| encoder.features.1.0.block.1.scale_activation | Sigmoid | 120x32x1x1 | 0.0195 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.1.0.block.2.0 | conv 1x1 | 120x16x112x112 | 0.6031 | 1.541 | 144.51 | 10.7 | 240 | 83% |
| encoder.features.1.0.block.2.1 | BatchNorm2d | 120x16x112x112 | 0.3949 | 0.096 | 96.34 | 1.0 | 244 | 85% |
| encoder.features.2.0.block.0.0 | conv 1x1 | 120x96x112x112 | 1.3289 | 4.624 | 337.19 | 13.7 | 254 | 88% |
| encoder.features.2.0.block.0.1 | BatchNorm2d | 120x96x112x112 | 2.2702 | 0.578 | 578.03 | 1.0 | 255 | 88% |
| encoder.features.2.0.block.0.2 | SiLU | 120x96x112x112 | 2.3439 | 0.578 | 578.03 | 1.0 | 247 | 86% |
| encoder.features.2.0.block.1.0 | conv dw 3x3 | 120x96x56x56 | 1.6281 | 0.650 | 361.27 | 1.8 | 222 | 77% |
| encoder.features.2.0.block.1.1 | BatchNorm2d | 120x96x56x56 | 0.5888 | 0.145 | 144.51 | 1.0 | 245 | 85% |
| encoder.features.2.0.block.1.2 | SiLU | 120x96x56x56 | 0.5878 | 0.145 | 144.51 | 1.0 | 246 | 85% |
| encoder.features.2.0.block.2.avgpool | AdaptiveAvgPool2d | 120x96x1x1 | 0.2703 | 0.036 | 72.28 | 0.5 | 267 | 93% |
| encoder.features.2.0.block.2.fc1 | conv 1x1 (SE) | 120x4x1x1 | 0.0410 | 0.000 | 0.02 | 3.7 | 1 | 0% |
| encoder.features.2.0.block.2.activation | SiLU | 120x4x1x1 | 0.0182 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.2.0.block.2.fc2 | conv 1x1 (SE) | 120x96x1x1 | 0.0461 | 0.000 | 0.02 | 3.7 | 1 | 0% |
| encoder.features.2.0.block.2.scale_activation | Sigmoid | 120x96x1x1 | 0.0205 | 0.000 | 0.05 | 1.0 | 2 | 1% |
| encoder.features.2.0.block.3.0 | conv 1x1 | 120x24x56x56 | 0.3591 | 1.734 | 90.32 | 19.2 | 251 | 87% |
| encoder.features.2.0.block.3.1 | BatchNorm2d | 120x24x56x56 | 0.1352 | 0.036 | 36.13 | 1.0 | 267 | 93% |
| encoder.features.2.1.block.0.0 | conv 1x1 | 120x144x56x56 | 0.5161 | 2.601 | 126.45 | 20.6 | 245 | 85% |
| encoder.features.2.1.block.0.1 | BatchNorm2d | 120x144x56x56 | 0.8724 | 0.217 | 216.76 | 1.0 | 248 | 86% |
| encoder.features.2.1.block.0.2 | SiLU | 120x144x56x56 | 0.8888 | 0.217 | 216.76 | 1.0 | 244 | 85% |
| encoder.features.2.1.block.1.0 | conv dw 3x3 | 120x144x56x56 | 1.9229 | 0.975 | 216.76 | 4.5 | 113 | 39% |
| encoder.features.2.1.block.1.1 | BatchNorm2d | 120x144x56x56 | 0.8775 | 0.217 | 216.76 | 1.0 | 247 | 86% |
| encoder.features.2.1.block.1.2 | SiLU | 120x144x56x56 | 0.8817 | 0.217 | 216.76 | 1.0 | 246 | 85% |
| encoder.features.2.1.block.2.avgpool | AdaptiveAvgPool2d | 120x144x1x1 | 0.4024 | 0.054 | 108.41 | 0.5 | 269 | 94% |
| encoder.features.2.1.block.2.fc1 | conv 1x1 (SE) | 120x6x1x1 | 0.0410 | 0.000 | 0.04 | 5.5 | 1 | 0% |
| encoder.features.2.1.block.2.activation | SiLU | 120x6x1x1 | 0.0174 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.2.1.block.2.fc2 | conv 1x1 (SE) | 120x144x1x1 | 0.0443 | 0.000 | 0.04 | 5.5 | 1 | 0% |
| encoder.features.2.1.block.2.scale_activation | Sigmoid | 120x144x1x1 | 0.0208 | 0.000 | 0.07 | 1.0 | 3 | 1% |
| encoder.features.2.1.block.3.0 | conv 1x1 | 120x24x56x56 | 0.4966 | 2.601 | 126.45 | 20.6 | 255 | 88% |
| encoder.features.2.1.block.3.1 | BatchNorm2d | 120x24x56x56 | 0.1342 | 0.036 | 36.13 | 1.0 | 269 | 93% |
| encoder.features.3.0.block.0.0 | conv 1x1 | 120x144x56x56 | 0.5147 | 2.601 | 126.45 | 20.6 | 246 | 85% |
| encoder.features.3.0.block.0.1 | BatchNorm2d | 120x144x56x56 | 0.8602 | 0.217 | 216.76 | 1.0 | 252 | 88% |
| encoder.features.3.0.block.0.2 | SiLU | 120x144x56x56 | 0.8796 | 0.217 | 216.76 | 1.0 | 246 | 86% |
| encoder.features.3.0.block.1.0 | conv dw 5x5 | 120x144x28x28 | 0.8448 | 0.677 | 135.48 | 5.0 | 160 | 56% |
| encoder.features.3.0.block.1.1 | BatchNorm2d | 120x144x28x28 | 0.2427 | 0.054 | 54.19 | 1.0 | 223 | 78% |
| encoder.features.3.0.block.1.2 | SiLU | 120x144x28x28 | 0.0948 | 0.054 | 54.19 | 1.0 | 572 | 198% |
| encoder.features.3.0.block.2.avgpool | AdaptiveAvgPool2d | 120x144x1x1 | 0.0276 | 0.014 | 27.13 | 0.5 | 981 | 341% |
| encoder.features.3.0.block.2.fc1 | conv 1x1 (SE) | 120x6x1x1 | 0.0450 | 0.000 | 0.04 | 5.5 | 1 | 0% |
| encoder.features.3.0.block.2.activation | SiLU | 120x6x1x1 | 0.0186 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.3.0.block.2.fc2 | conv 1x1 (SE) | 120x144x1x1 | 0.0459 | 0.000 | 0.04 | 5.5 | 1 | 0% |
| encoder.features.3.0.block.2.scale_activation | Sigmoid | 120x144x1x1 | 0.0196 | 0.000 | 0.07 | 1.0 | 4 | 1% |
| encoder.features.3.0.block.3.0 | conv 1x1 | 120x40x28x28 | 0.1290 | 1.084 | 34.63 | 31.3 | 268 | 93% |
| encoder.features.3.0.block.3.1 | BatchNorm2d | 120x40x28x28 | 0.0430 | 0.015 | 15.05 | 1.0 | 350 | 122% |
| encoder.features.3.1.block.0.0 | conv 1x1 | 120x240x28x28 | 0.2007 | 1.806 | 52.70 | 34.3 | 263 | 91% |
| encoder.features.3.1.block.0.1 | BatchNorm2d | 120x240x28x28 | 0.3868 | 0.090 | 90.32 | 1.0 | 233 | 81% |
| encoder.features.3.1.block.0.2 | SiLU | 120x240x28x28 | 0.3666 | 0.090 | 90.32 | 1.0 | 246 | 86% |
| encoder.features.3.1.block.1.0 | conv dw 5x5 | 120x240x28x28 | 1.2622 | 1.129 | 90.33 | 12.5 | 72 | 25% |
| encoder.features.3.1.block.1.1 | BatchNorm2d | 120x240x28x28 | 0.3912 | 0.090 | 90.32 | 1.0 | 231 | 80% |
| encoder.features.3.1.block.1.2 | SiLU | 120x240x28x28 | 0.3747 | 0.090 | 90.32 | 1.0 | 241 | 84% |
| encoder.features.3.1.block.2.avgpool | AdaptiveAvgPool2d | 120x240x1x1 | 0.1792 | 0.023 | 45.22 | 0.5 | 252 | 88% |
| encoder.features.3.1.block.2.fc1 | conv 1x1 (SE) | 120x10x1x1 | 0.0443 | 0.001 | 0.06 | 8.9 | 1 | 1% |
| encoder.features.3.1.block.2.activation | SiLU | 120x10x1x1 | 0.0258 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.3.1.block.2.fc2 | conv 1x1 (SE) | 120x240x1x1 | 0.0451 | 0.001 | 0.07 | 8.8 | 1 | 1% |
| encoder.features.3.1.block.2.scale_activation | Sigmoid | 120x240x1x1 | 0.0236 | 0.000 | 0.12 | 1.0 | 5 | 2% |
| encoder.features.3.1.block.3.0 | conv 1x1 | 120x40x28x28 | 0.2314 | 1.806 | 52.70 | 34.3 | 228 | 79% |
| encoder.features.3.1.block.3.1 | BatchNorm2d | 120x40x28x28 | 0.0603 | 0.015 | 15.05 | 1.0 | 250 | 87% |
| encoder.features.4.0.block.0.0 | conv 1x1 | 120x240x28x28 | 0.1987 | 1.806 | 52.70 | 34.3 | 265 | 92% |
| encoder.features.4.0.block.0.1 | BatchNorm2d | 120x240x28x28 | 0.4024 | 0.090 | 90.32 | 1.0 | 224 | 78% |
| encoder.features.4.0.block.0.2 | SiLU | 120x240x28x28 | 0.3728 | 0.090 | 90.32 | 1.0 | 242 | 84% |
| encoder.features.4.0.block.1.0 | conv dw 3x3 | 120x240x14x14 | 0.2765 | 0.102 | 56.45 | 1.8 | 204 | 71% |
| encoder.features.4.0.block.1.1 | BatchNorm2d | 120x240x14x14 | 0.0561 | 0.023 | 22.58 | 1.0 | 403 | 140% |
| encoder.features.4.0.block.1.2 | SiLU | 120x240x14x14 | 0.0184 | 0.023 | 22.58 | 1.0 | 1225 | 425% |
| encoder.features.4.0.block.2.avgpool | AdaptiveAvgPool2d | 120x240x1x1 | 0.0276 | 0.006 | 11.35 | 0.5 | 410 | 143% |
| encoder.features.4.0.block.2.fc1 | conv 1x1 (SE) | 120x10x1x1 | 0.0515 | 0.001 | 0.06 | 8.9 | 1 | 0% |
| encoder.features.4.0.block.2.activation | SiLU | 120x10x1x1 | 0.0203 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.4.0.block.2.fc2 | conv 1x1 (SE) | 120x240x1x1 | 0.0502 | 0.001 | 0.07 | 8.8 | 1 | 0% |
| encoder.features.4.0.block.2.scale_activation | Sigmoid | 120x240x1x1 | 0.0225 | 0.000 | 0.12 | 1.0 | 5 | 2% |
| encoder.features.4.0.block.3.0 | conv 1x1 | 120x80x14x14 | 0.0908 | 0.903 | 15.09 | 59.8 | 166 | 58% |
| encoder.features.4.0.block.3.1 | BatchNorm2d | 120x80x14x14 | 0.0594 | 0.008 | 7.53 | 1.0 | 127 | 44% |
| encoder.features.4.1.block.0.0 | conv 1x1 | 120x480x14x14 | 0.1321 | 1.806 | 26.42 | 68.4 | 200 | 69% |
| encoder.features.4.1.block.0.1 | BatchNorm2d | 120x480x14x14 | 0.2054 | 0.045 | 45.16 | 1.0 | 220 | 76% |
| encoder.features.4.1.block.0.2 | SiLU | 120x480x14x14 | 0.0317 | 0.045 | 45.16 | 1.0 | 1423 | 494% |
| encoder.features.4.1.block.1.0 | conv dw 3x3 | 120x480x14x14 | 0.4156 | 0.203 | 45.17 | 4.5 | 109 | 38% |
| encoder.features.4.1.block.1.1 | BatchNorm2d | 120x480x14x14 | 0.2037 | 0.045 | 45.16 | 1.0 | 222 | 77% |
| encoder.features.4.1.block.1.2 | SiLU | 120x480x14x14 | 0.0317 | 0.045 | 45.16 | 1.0 | 1423 | 494% |
| encoder.features.4.1.block.2.avgpool | AdaptiveAvgPool2d | 120x480x1x1 | 0.0307 | 0.011 | 22.69 | 0.5 | 739 | 257% |
| encoder.features.4.1.block.2.fc1 | conv 1x1 (SE) | 120x20x1x1 | 0.0446 | 0.002 | 0.14 | 16.5 | 3 | 1% |
| encoder.features.4.1.block.2.activation | SiLU | 120x20x1x1 | 0.0193 | 0.000 | 0.01 | 1.0 | 0 | 0% |
| encoder.features.4.1.block.2.fc2 | conv 1x1 (SE) | 120x480x1x1 | 0.0474 | 0.002 | 0.14 | 16.4 | 3 | 1% |
| encoder.features.4.1.block.2.scale_activation | Sigmoid | 120x480x1x1 | 0.0226 | 0.000 | 0.23 | 1.0 | 10 | 4% |
| encoder.features.4.1.block.3.0 | conv 1x1 | 120x80x14x14 | 0.1474 | 1.806 | 26.42 | 68.4 | 179 | 62% |
| encoder.features.4.1.block.3.1 | BatchNorm2d | 120x80x14x14 | 0.0583 | 0.008 | 7.53 | 1.0 | 129 | 45% |
| encoder.features.4.2.block.0.0 | conv 1x1 | 120x480x14x14 | 0.1338 | 1.806 | 26.42 | 68.4 | 197 | 69% |
| encoder.features.4.2.block.0.1 | BatchNorm2d | 120x480x14x14 | 0.2038 | 0.045 | 45.16 | 1.0 | 222 | 77% |
| encoder.features.4.2.block.0.2 | SiLU | 120x480x14x14 | 0.0318 | 0.045 | 45.16 | 1.0 | 1421 | 493% |
| encoder.features.4.2.block.1.0 | conv dw 3x3 | 120x480x14x14 | 0.4127 | 0.203 | 45.17 | 4.5 | 109 | 38% |
| encoder.features.4.2.block.1.1 | BatchNorm2d | 120x480x14x14 | 0.2160 | 0.045 | 45.16 | 1.0 | 209 | 73% |
| encoder.features.4.2.block.1.2 | SiLU | 120x480x14x14 | 0.0317 | 0.045 | 45.16 | 1.0 | 1423 | 494% |
| encoder.features.4.2.block.2.avgpool | AdaptiveAvgPool2d | 120x480x1x1 | 0.0307 | 0.011 | 22.69 | 0.5 | 739 | 257% |
| encoder.features.4.2.block.2.fc1 | conv 1x1 (SE) | 120x20x1x1 | 0.0461 | 0.002 | 0.14 | 16.5 | 3 | 1% |
| encoder.features.4.2.block.2.activation | SiLU | 120x20x1x1 | 0.0186 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.4.2.block.2.fc2 | conv 1x1 (SE) | 120x480x1x1 | 0.0462 | 0.002 | 0.14 | 16.4 | 3 | 1% |
| encoder.features.4.2.block.2.scale_activation | Sigmoid | 120x480x1x1 | 0.0221 | 0.000 | 0.23 | 1.0 | 10 | 4% |
| encoder.features.4.2.block.3.0 | conv 1x1 | 120x80x14x14 | 0.1352 | 1.806 | 26.42 | 68.4 | 195 | 68% |
| encoder.features.4.2.block.3.1 | BatchNorm2d | 120x80x14x14 | 0.0481 | 0.008 | 7.53 | 1.0 | 156 | 54% |
| encoder.features.5.0.block.0.0 | conv 1x1 | 120x480x14x14 | 0.1300 | 1.806 | 26.42 | 68.4 | 203 | 71% |
| encoder.features.5.0.block.0.1 | BatchNorm2d | 120x480x14x14 | 0.2017 | 0.045 | 45.16 | 1.0 | 224 | 78% |
| encoder.features.5.0.block.0.2 | SiLU | 120x480x14x14 | 0.0317 | 0.045 | 45.16 | 1.0 | 1423 | 494% |
| encoder.features.5.0.block.1.0 | conv dw 5x5 | 120x480x14x14 | 0.6410 | 0.564 | 45.18 | 12.5 | 70 | 24% |
| encoder.features.5.0.block.1.1 | BatchNorm2d | 120x480x14x14 | 0.2019 | 0.045 | 45.16 | 1.0 | 224 | 78% |
| encoder.features.5.0.block.1.2 | SiLU | 120x480x14x14 | 0.0358 | 0.045 | 45.16 | 1.0 | 1260 | 437% |
| encoder.features.5.0.block.2.avgpool | AdaptiveAvgPool2d | 120x480x1x1 | 0.0307 | 0.011 | 22.69 | 0.5 | 739 | 257% |
| encoder.features.5.0.block.2.fc1 | conv 1x1 (SE) | 120x20x1x1 | 0.0441 | 0.002 | 0.14 | 16.5 | 3 | 1% |
| encoder.features.5.0.block.2.activation | SiLU | 120x20x1x1 | 0.0183 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.5.0.block.2.fc2 | conv 1x1 (SE) | 120x480x1x1 | 0.0428 | 0.002 | 0.14 | 16.4 | 3 | 1% |
| encoder.features.5.0.block.2.scale_activation | Sigmoid | 120x480x1x1 | 0.0205 | 0.000 | 0.23 | 1.0 | 11 | 4% |
| encoder.features.5.0.block.3.0 | conv 1x1 | 120x112x14x14 | 0.1362 | 2.529 | 27.96 | 90.5 | 205 | 71% |
| encoder.features.5.0.block.3.1 | BatchNorm2d | 120x112x14x14 | 0.0461 | 0.011 | 10.54 | 1.0 | 229 | 79% |
| encoder.features.5.1.block.0.0 | conv 1x1 | 120x672x14x14 | 0.2171 | 3.540 | 37.03 | 95.6 | 171 | 59% |
| encoder.features.5.1.block.0.1 | BatchNorm2d | 120x672x14x14 | 0.2724 | 0.063 | 63.23 | 1.0 | 232 | 81% |
| encoder.features.5.1.block.0.2 | SiLU | 120x672x14x14 | 0.1136 | 0.063 | 63.22 | 1.0 | 556 | 193% |
| encoder.features.5.1.block.1.0 | conv dw 5x5 | 120x672x14x14 | 0.8824 | 0.790 | 63.26 | 12.5 | 72 | 25% |
| encoder.features.5.1.block.1.1 | BatchNorm2d | 120x672x14x14 | 0.2703 | 0.063 | 63.23 | 1.0 | 234 | 81% |
| encoder.features.5.1.block.1.2 | SiLU | 120x672x14x14 | 0.1157 | 0.063 | 63.22 | 1.0 | 546 | 190% |
| encoder.features.5.1.block.2.avgpool | AdaptiveAvgPool2d | 120x672x1x1 | 0.0522 | 0.016 | 31.77 | 0.5 | 608 | 211% |
| encoder.features.5.1.block.2.fc1 | conv 1x1 (SE) | 120x28x1x1 | 0.0544 | 0.005 | 0.21 | 22.0 | 4 | 1% |
| encoder.features.5.1.block.2.activation | SiLU | 120x28x1x1 | 0.0219 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.5.1.block.2.fc2 | conv 1x1 (SE) | 120x672x1x1 | 0.0522 | 0.005 | 0.21 | 21.8 | 4 | 1% |
| encoder.features.5.1.block.2.scale_activation | Sigmoid | 120x672x1x1 | 0.0236 | 0.000 | 0.32 | 1.0 | 14 | 5% |
| encoder.features.5.1.block.3.0 | conv 1x1 | 120x112x14x14 | 0.1853 | 3.540 | 37.03 | 95.6 | 200 | 69% |
| encoder.features.5.1.block.3.1 | BatchNorm2d | 120x112x14x14 | 0.0491 | 0.011 | 10.54 | 1.0 | 215 | 75% |
| encoder.features.5.2.block.0.0 | conv 1x1 | 120x672x14x14 | 0.2212 | 3.540 | 37.03 | 95.6 | 167 | 58% |
| encoder.features.5.2.block.0.1 | BatchNorm2d | 120x672x14x14 | 0.2794 | 0.063 | 63.23 | 1.0 | 226 | 79% |
| encoder.features.5.2.block.0.2 | SiLU | 120x672x14x14 | 0.1167 | 0.063 | 63.22 | 1.0 | 542 | 188% |
| encoder.features.5.2.block.1.0 | conv dw 5x5 | 120x672x14x14 | 0.8827 | 0.790 | 63.26 | 12.5 | 72 | 25% |
| encoder.features.5.2.block.1.1 | BatchNorm2d | 120x672x14x14 | 0.2746 | 0.063 | 63.23 | 1.0 | 230 | 80% |
| encoder.features.5.2.block.1.2 | SiLU | 120x672x14x14 | 0.1137 | 0.063 | 63.22 | 1.0 | 556 | 193% |
| encoder.features.5.2.block.2.avgpool | AdaptiveAvgPool2d | 120x672x1x1 | 0.0430 | 0.016 | 31.77 | 0.5 | 739 | 257% |
| encoder.features.5.2.block.2.fc1 | conv 1x1 (SE) | 120x28x1x1 | 0.0533 | 0.005 | 0.21 | 22.0 | 4 | 1% |
| encoder.features.5.2.block.2.activation | SiLU | 120x28x1x1 | 0.0205 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.5.2.block.2.fc2 | conv 1x1 (SE) | 120x672x1x1 | 0.0563 | 0.005 | 0.21 | 21.8 | 4 | 1% |
| encoder.features.5.2.block.2.scale_activation | Sigmoid | 120x672x1x1 | 0.0247 | 0.000 | 0.32 | 1.0 | 13 | 5% |
| encoder.features.5.2.block.3.0 | conv 1x1 | 120x112x14x14 | 0.1864 | 3.540 | 37.03 | 95.6 | 199 | 69% |
| encoder.features.5.2.block.3.1 | BatchNorm2d | 120x112x14x14 | 0.0492 | 0.011 | 10.54 | 1.0 | 214 | 74% |
| encoder.features.6.0.block.0.0 | conv 1x1 | 120x672x14x14 | 0.2250 | 3.540 | 37.03 | 95.6 | 165 | 57% |
| encoder.features.6.0.block.0.1 | BatchNorm2d | 120x672x14x14 | 0.2775 | 0.063 | 63.23 | 1.0 | 228 | 79% |
| encoder.features.6.0.block.0.2 | SiLU | 120x672x14x14 | 0.1146 | 0.063 | 63.22 | 1.0 | 552 | 192% |
| encoder.features.6.0.block.1.0 | conv dw 5x5 | 120x672x7x7 | 0.2570 | 0.198 | 39.55 | 5.0 | 154 | 53% |
| encoder.features.6.0.block.1.1 | BatchNorm2d | 120x672x7x7 | 0.0942 | 0.016 | 15.81 | 1.0 | 168 | 58% |
| encoder.features.6.0.block.1.2 | SiLU | 120x672x7x7 | 0.0201 | 0.016 | 15.81 | 1.0 | 786 | 273% |
| encoder.features.6.0.block.2.avgpool | AdaptiveAvgPool2d | 120x672x1x1 | 0.0509 | 0.004 | 8.06 | 0.5 | 158 | 55% |
| encoder.features.6.0.block.2.fc1 | conv 1x1 (SE) | 120x28x1x1 | 0.0573 | 0.005 | 0.21 | 22.0 | 4 | 1% |
| encoder.features.6.0.block.2.activation | SiLU | 120x28x1x1 | 0.0229 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.6.0.block.2.fc2 | conv 1x1 (SE) | 120x672x1x1 | 0.0563 | 0.005 | 0.21 | 21.8 | 4 | 1% |
| encoder.features.6.0.block.2.scale_activation | Sigmoid | 120x672x1x1 | 0.0236 | 0.000 | 0.32 | 1.0 | 14 | 5% |
| encoder.features.6.0.block.3.0 | conv 1x1 | 120x192x7x7 | 0.0891 | 1.517 | 10.42 | 145.6 | 117 | 41% |
| encoder.features.6.0.block.3.1 | BatchNorm2d | 120x192x7x7 | 0.0532 | 0.005 | 4.52 | 1.0 | 85 | 29% |
| encoder.features.6.1.block.0.0 | conv 1x1 | 120x1152x7x7 | 0.1444 | 2.601 | 16.25 | 160.1 | 113 | 39% |
| encoder.features.6.1.block.0.1 | BatchNorm2d | 120x1152x7x7 | 0.1413 | 0.027 | 27.10 | 1.0 | 192 | 67% |
| encoder.features.6.1.block.0.2 | SiLU | 120x1152x7x7 | 0.0215 | 0.027 | 27.10 | 1.0 | 1260 | 438% |
| encoder.features.6.1.block.1.0 | conv dw 5x5 | 120x1152x7x7 | 0.3768 | 0.339 | 27.15 | 12.5 | 72 | 25% |
| encoder.features.6.1.block.1.1 | BatchNorm2d | 120x1152x7x7 | 0.1379 | 0.027 | 27.10 | 1.0 | 197 | 68% |
| encoder.features.6.1.block.1.2 | SiLU | 120x1152x7x7 | 0.0205 | 0.027 | 27.10 | 1.0 | 1323 | 459% |
| encoder.features.6.1.block.2.avgpool | AdaptiveAvgPool2d | 120x1152x1x1 | 0.0840 | 0.007 | 13.82 | 0.5 | 165 | 57% |
| encoder.features.6.1.block.2.fc1 | conv 1x1 (SE) | 120x48x1x1 | 0.0534 | 0.013 | 0.40 | 33.3 | 7 | 3% |
| encoder.features.6.1.block.2.activation | SiLU | 120x48x1x1 | 0.0377 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.6.1.block.2.fc2 | conv 1x1 (SE) | 120x1152x1x1 | 0.0481 | 0.013 | 0.40 | 33.1 | 8 | 3% |
| encoder.features.6.1.block.2.scale_activation | Sigmoid | 120x1152x1x1 | 0.0225 | 0.001 | 0.55 | 1.0 | 25 | 9% |
| encoder.features.6.1.block.3.0 | conv 1x1 | 120x192x7x7 | 0.1505 | 2.601 | 16.25 | 160.1 | 108 | 37% |
| encoder.features.6.1.block.3.1 | BatchNorm2d | 120x192x7x7 | 0.0481 | 0.005 | 4.52 | 1.0 | 94 | 33% |
| encoder.features.6.2.block.0.0 | conv 1x1 | 120x1152x7x7 | 0.1444 | 2.601 | 16.25 | 160.1 | 113 | 39% |
| encoder.features.6.2.block.0.1 | BatchNorm2d | 120x1152x7x7 | 0.1423 | 0.027 | 27.10 | 1.0 | 190 | 66% |
| encoder.features.6.2.block.0.2 | SiLU | 120x1152x7x7 | 0.0205 | 0.027 | 27.10 | 1.0 | 1323 | 459% |
| encoder.features.6.2.block.1.0 | conv dw 5x5 | 120x1152x7x7 | 0.3789 | 0.339 | 27.15 | 12.5 | 72 | 25% |
| encoder.features.6.2.block.1.1 | BatchNorm2d | 120x1152x7x7 | 0.1395 | 0.027 | 27.10 | 1.0 | 194 | 67% |
| encoder.features.6.2.block.1.2 | SiLU | 120x1152x7x7 | 0.0286 | 0.027 | 27.10 | 1.0 | 946 | 328% |
| encoder.features.6.2.block.2.avgpool | AdaptiveAvgPool2d | 120x1152x1x1 | 0.0963 | 0.007 | 13.82 | 0.5 | 144 | 50% |
| encoder.features.6.2.block.2.fc1 | conv 1x1 (SE) | 120x48x1x1 | 0.0604 | 0.013 | 0.40 | 33.3 | 7 | 2% |
| encoder.features.6.2.block.2.activation | SiLU | 120x48x1x1 | 0.0349 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.6.2.block.2.fc2 | conv 1x1 (SE) | 120x1152x1x1 | 0.0512 | 0.013 | 0.40 | 33.1 | 8 | 3% |
| encoder.features.6.2.block.2.scale_activation | Sigmoid | 120x1152x1x1 | 0.0287 | 0.001 | 0.55 | 1.0 | 19 | 7% |
| encoder.features.6.2.block.3.0 | conv 1x1 | 120x192x7x7 | 0.1440 | 2.601 | 16.25 | 160.1 | 113 | 39% |
| encoder.features.6.2.block.3.1 | BatchNorm2d | 120x192x7x7 | 0.0553 | 0.005 | 4.52 | 1.0 | 82 | 28% |
| encoder.features.6.3.block.0.0 | conv 1x1 | 120x1152x7x7 | 0.1485 | 2.601 | 16.25 | 160.1 | 109 | 38% |
| encoder.features.6.3.block.0.1 | BatchNorm2d | 120x1152x7x7 | 0.1556 | 0.027 | 27.10 | 1.0 | 174 | 60% |
| encoder.features.6.3.block.0.2 | SiLU | 120x1152x7x7 | 0.0348 | 0.027 | 27.10 | 1.0 | 778 | 270% |
| encoder.features.6.3.block.1.0 | conv dw 5x5 | 120x1152x7x7 | 0.3820 | 0.339 | 27.15 | 12.5 | 71 | 25% |
| encoder.features.6.3.block.1.1 | BatchNorm2d | 120x1152x7x7 | 0.1372 | 0.027 | 27.10 | 1.0 | 198 | 69% |
| encoder.features.6.3.block.1.2 | SiLU | 120x1152x7x7 | 0.0212 | 0.027 | 27.10 | 1.0 | 1277 | 443% |
| encoder.features.6.3.block.2.avgpool | AdaptiveAvgPool2d | 120x1152x1x1 | 0.0838 | 0.007 | 13.82 | 0.5 | 165 | 57% |
| encoder.features.6.3.block.2.fc1 | conv 1x1 (SE) | 120x48x1x1 | 0.0543 | 0.013 | 0.40 | 33.3 | 7 | 3% |
| encoder.features.6.3.block.2.activation | SiLU | 120x48x1x1 | 0.0208 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.6.3.block.2.fc2 | conv 1x1 (SE) | 120x1152x1x1 | 0.0512 | 0.013 | 0.40 | 33.1 | 8 | 3% |
| encoder.features.6.3.block.2.scale_activation | Sigmoid | 120x1152x1x1 | 0.0236 | 0.001 | 0.55 | 1.0 | 23 | 8% |
| encoder.features.6.3.block.3.0 | conv 1x1 | 120x192x7x7 | 0.1403 | 2.601 | 16.25 | 160.1 | 116 | 40% |
| encoder.features.6.3.block.3.1 | BatchNorm2d | 120x192x7x7 | 0.0512 | 0.005 | 4.52 | 1.0 | 88 | 31% |
| encoder.features.7.0.block.0.0 | conv 1x1 | 120x1152x7x7 | 0.1412 | 2.601 | 16.25 | 160.1 | 115 | 40% |
| encoder.features.7.0.block.0.1 | BatchNorm2d | 120x1152x7x7 | 0.1350 | 0.027 | 27.10 | 1.0 | 201 | 70% |
| encoder.features.7.0.block.0.2 | SiLU | 120x1152x7x7 | 0.0212 | 0.027 | 27.10 | 1.0 | 1279 | 444% |
| encoder.features.7.0.block.1.0 | conv dw 3x3 | 120x1152x7x7 | 0.2322 | 0.122 | 27.12 | 4.5 | 117 | 41% |
| encoder.features.7.0.block.1.1 | BatchNorm2d | 120x1152x7x7 | 0.1382 | 0.027 | 27.10 | 1.0 | 196 | 68% |
| encoder.features.7.0.block.1.2 | SiLU | 120x1152x7x7 | 0.0205 | 0.027 | 27.10 | 1.0 | 1323 | 459% |
| encoder.features.7.0.block.2.avgpool | AdaptiveAvgPool2d | 120x1152x1x1 | 0.0838 | 0.007 | 13.82 | 0.5 | 165 | 57% |
| encoder.features.7.0.block.2.fc1 | conv 1x1 (SE) | 120x48x1x1 | 0.0514 | 0.013 | 0.40 | 33.3 | 8 | 3% |
| encoder.features.7.0.block.2.activation | SiLU | 120x48x1x1 | 0.0195 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.7.0.block.2.fc2 | conv 1x1 (SE) | 120x1152x1x1 | 0.0475 | 0.013 | 0.40 | 33.1 | 8 | 3% |
| encoder.features.7.0.block.2.scale_activation | Sigmoid | 120x1152x1x1 | 0.0223 | 0.001 | 0.55 | 1.0 | 25 | 9% |
| encoder.features.7.0.block.3.0 | conv 1x1 | 120x320x7x7 | 0.2047 | 4.335 | 18.05 | 240.2 | 88 | 31% |
| encoder.features.7.0.block.3.1 | BatchNorm2d | 120x320x7x7 | 0.0587 | 0.008 | 7.53 | 1.0 | 128 | 45% |
| encoder.features.8.0 | conv 1x1 | 120x1280x7x7 | 0.2253 | 4.817 | 19.64 | 245.3 | 87 | 30% |
| encoder.features.8.1 | BatchNorm2d | 120x1280x7x7 | 0.1609 | 0.030 | 30.12 | 1.0 | 187 | 65% |
| encoder.features.8.2 | SiLU | 120x1280x7x7 | 0.0236 | 0.030 | 30.11 | 1.0 | 1278 | 444% |
| encoder.pool | AdaptiveAvgPool2d | 120x1280x1x1 | 0.1044 | 0.008 | 15.36 | 0.5 | 147 | 51% |
| temporal.proj | Linear | 4x30x512 | 0.0389 | 0.157 | 1.74 | 90.3 | 45 | 16% |
| temporal.layers.0.norm | LayerNorm | 4x30x512 | 0.0328 | 0.000 | 0.25 | 1.0 | 8 | 3% |
| temporal.layers.0.mixer.in_proj | Linear | 4x30x2048 | 0.0340 | 0.252 | 2.71 | 92.8 | 80 | 28% |
| temporal.layers.0.mixer.conv1d | Conv1d | 4x1024x33 | 0.0471 | 0.001 | 0.53 | 2.1 | 11 | 4% |
| temporal.layers.0.mixer.dt_proj | Linear | 4x30x1024 | 0.0623 | 0.252 | 2.59 | 97.1 | 42 | 14% |
| temporal.layers.0.mixer.x_proj | Linear | 4x30x32 | 0.0573 | 0.008 | 0.32 | 24.7 | 6 | 2% |
| temporal.layers.0.mixer.out_proj | Linear | 4x30x512 | 0.0325 | 0.126 | 1.42 | 88.8 | 44 | 15% |
| temporal.layers.1.norm | LayerNorm | 4x30x512 | 0.0308 | 0.000 | 0.25 | 1.0 | 8 | 3% |
| temporal.layers.1.mixer.in_proj | Linear | 4x30x2048 | 0.0345 | 0.252 | 2.71 | 92.8 | 79 | 27% |
| temporal.layers.1.mixer.conv1d | Conv1d | 4x1024x33 | 0.0481 | 0.001 | 0.53 | 2.1 | 11 | 4% |
| temporal.layers.1.mixer.dt_proj | Linear | 4x30x1024 | 0.0625 | 0.252 | 2.59 | 97.1 | 41 | 14% |
| temporal.layers.1.mixer.x_proj | Linear | 4x30x32 | 0.0566 | 0.008 | 0.32 | 24.7 | 6 | 2% |
| temporal.layers.1.mixer.out_proj | Linear | 4x30x512 | 0.0356 | 0.126 | 1.42 | 88.8 | 40 | 14% |
| temporal.layers.2.norm | LayerNorm | 4x30x512 | 0.0320 | 0.000 | 0.25 | 1.0 | 8 | 3% |
| temporal.layers.2.mixer.in_proj | Linear | 4x30x2048 | 0.0358 | 0.252 | 2.71 | 92.8 | 76 | 26% |
| temporal.layers.2.mixer.conv1d | Conv1d | 4x1024x33 | 0.0490 | 0.001 | 0.53 | 2.1 | 11 | 4% |
| temporal.layers.2.mixer.dt_proj | Linear | 4x30x1024 | 0.0665 | 0.252 | 2.59 | 97.1 | 39 | 14% |
| temporal.layers.2.mixer.x_proj | Linear | 4x30x32 | 0.0625 | 0.008 | 0.32 | 24.7 | 5 | 2% |
| temporal.layers.2.mixer.out_proj | Linear | 4x30x512 | 0.0389 | 0.126 | 1.42 | 88.8 | 36 | 13% |
| temporal.layers.3.norm | LayerNorm | 4x30x512 | 0.0328 | 0.000 | 0.25 | 1.0 | 8 | 3% |
| temporal.layers.3.mixer.in_proj | Linear | 4x30x2048 | 0.0356 | 0.252 | 2.71 | 92.8 | 76 | 26% |
| temporal.layers.3.mixer.conv1d | Conv1d | 4x1024x33 | 0.0483 | 0.001 | 0.53 | 2.1 | 11 | 4% |
| temporal.layers.3.mixer.dt_proj | Linear | 4x30x1024 | 0.0655 | 0.252 | 2.59 | 97.1 | 40 | 14% |
| temporal.layers.3.mixer.x_proj | Linear | 4x30x32 | 0.0593 | 0.008 | 0.32 | 24.7 | 5 | 2% |
| temporal.layers.3.mixer.out_proj | Linear | 4x30x512 | 0.0348 | 0.126 | 1.42 | 88.8 | 41 | 14% |
| temporal.norm_f | LayerNorm | 4x30x512 | 0.0338 | 0.000 | 0.25 | 1.0 | 7 | 3% |
| temporal.head.0 | Linear | 4x256 | 0.0355 | 0.001 | 0.27 | 3.9 | 8 | 3% |
| temporal.head.1 | GELU | 4x256 | 0.0236 | 0.000 | 0.00 | 0.2 | 0 | 0% |
| temporal.head.3 | Linear | 4x2 | 0.0348 | 0.000 | 0.00 | 1.3 | 0 | 0% |
| temporal.layers.0.mixer.scan | selective scan (python loop) | 4x30x1024 | 4.3476 | 0.273 | 32.01 | 8.5 | 7 | 3% |
| temporal.layers.1.mixer.scan | selective scan (python loop) | 4x30x1024 | 4.5114 | 0.273 | 32.01 | 8.5 | 7 | 2% |
| temporal.layers.2.mixer.scan | selective scan (python loop) | 4x30x1024 | 4.6159 | 0.273 | 32.01 | 8.5 | 7 | 2% |
| temporal.layers.3.mixer.scan | selective scan (python loop) | 4x30x1024 | 4.7452 | 0.273 | 32.01 | 8.5 | 7 | 2% |
