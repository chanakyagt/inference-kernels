# Per-layer profile: 1 clip(s) x 30 frames, bfloat16, NVIDIA GeForce RTX 4060 Ti

- End-to-end forward: **24.57 ms**; EfficientNet-B0 part alone: 10.91 ms
- Sum of every op timed in isolation: 35.88 ms (differs from end-to-end because isolated timing has no launch overlap and skips un-hooked ops such as residual adds and the squeeze-excitation multiply)
- Ceilings used: 288 GB/s, 45.1 TFLOP/s (measured BF16 tensor-core matmul), ridge 157 FLOP/byte

## Grouped by kind

| Section | Kind | Ops | Time (ms) | Share | GFLOP | MB moved | FLOP/byte | Achieved GB/s | Bound |
|---|---|---|---|---|---|---|---|---|---|
| Mamba | selective scan (python loop) | 4 | 20.094 | 56.0% | 0.27 | 32.2 | 8.5 | 2 | memory |
| EfficientNet-B0 | BatchNorm2d | 49 | 4.288 | 11.9% | 0.81 | 808.7 | 1.0 | 189 | memory |
| EfficientNet-B0 | conv 1x1 | 32 | 2.418 | 6.7% | 20.31 | 421.3 | 48.2 | 174 | memory |
| EfficientNet-B0 | SiLU | 49 | 1.919 | 5.3% | 0.74 | 738.9 | 1.0 | 385 | memory |
| EfficientNet-B0 | conv dw 3x3 | 7 | 1.698 | 4.7% | 0.78 | 236.2 | 3.3 | 139 | memory |
| EfficientNet-B0 | conv 1x1 (SE) | 32 | 1.618 | 4.5% | 0.04 | 2.4 | 15.7 | 1 | memory |
| EfficientNet-B0 | conv dw 5x5 | 9 | 1.544 | 4.3% | 1.29 | 129.9 | 9.9 | 84 | memory |
| Mamba | Linear | 16 | 0.721 | 2.0% | 0.64 | 23.0 | 27.7 | 32 | memory |
| EfficientNet-B0 | AdaptiveAvgPool2d | 17 | 0.503 | 1.4% | 0.07 | 142.8 | 0.5 | 284 | memory |
| EfficientNet-B0 | Sigmoid | 16 | 0.380 | 1.1% | 0.00 | 1.1 | 1.0 | 3 | memory |
| Mamba | Conv1d | 4 | 0.209 | 0.6% | 0.00 | 0.6 | 1.9 | 3 | memory |
| EfficientNet-B0 | conv 3x3 | 1 | 0.171 | 0.5% | 0.65 | 33.1 | 19.6 | 194 | memory |
| Mamba | LayerNorm | 4 | 0.140 | 0.4% | 0.00 | 0.3 | 1.0 | 2 | memory |
| Head | Linear | 2 | 0.074 | 0.2% | 0.00 | 0.3 | 1.0 | 4 | memory |
| Projection | Linear | 1 | 0.042 | 0.1% | 0.04 | 1.4 | 27.7 | 34 | memory |
| Head | LayerNorm | 1 | 0.036 | 0.1% | 0.00 | 0.1 | 1.0 | 2 | memory |
| Head | GELU | 1 | 0.028 | 0.1% | 0.00 | 0.0 | 0.2 | 0 | memory |

## Every op

| Op | Kind | Output shape | Time (ms) | GFLOP | MB moved | FLOP/byte | GB/s | % of peak BW |
|---|---|---|---|---|---|---|---|---|
| encoder.features.0.0 | conv 3x3 | 30x32x112x112 | 0.1707 | 0.650 | 33.12 | 19.6 | 194 | 67% |
| encoder.features.0.1 | BatchNorm2d | 30x32x112x112 | 0.2120 | 0.048 | 48.17 | 1.0 | 227 | 79% |
| encoder.features.0.2 | SiLU | 30x32x112x112 | 0.0337 | 0.048 | 48.17 | 1.0 | 1428 | 496% |
| encoder.features.1.0.block.0.0 | conv dw 3x3 | 30x32x112x112 | 0.4321 | 0.217 | 48.17 | 4.5 | 111 | 39% |
| encoder.features.1.0.block.0.1 | BatchNorm2d | 30x32x112x112 | 0.2089 | 0.048 | 48.17 | 1.0 | 231 | 80% |
| encoder.features.1.0.block.0.2 | SiLU | 30x32x112x112 | 0.0345 | 0.048 | 48.17 | 1.0 | 1395 | 484% |
| encoder.features.1.0.block.1.avgpool | AdaptiveAvgPool2d | 30x32x1x1 | 0.0236 | 0.012 | 24.09 | 0.5 | 1023 | 355% |
| encoder.features.1.0.block.1.fc1 | conv 1x1 (SE) | 30x8x1x1 | 0.0440 | 0.000 | 0.00 | 5.2 | 0 | 0% |
| encoder.features.1.0.block.1.activation | SiLU | 30x8x1x1 | 0.0184 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.1.0.block.1.fc2 | conv 1x1 (SE) | 30x32x1x1 | 0.0471 | 0.000 | 0.00 | 5.2 | 0 | 0% |
| encoder.features.1.0.block.1.scale_activation | Sigmoid | 30x32x1x1 | 0.0208 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.1.0.block.2.0 | conv 1x1 | 30x16x112x112 | 0.0983 | 0.385 | 36.13 | 10.7 | 368 | 128% |
| encoder.features.1.0.block.2.1 | BatchNorm2d | 30x16x112x112 | 0.0462 | 0.024 | 24.08 | 1.0 | 522 | 181% |
| encoder.features.2.0.block.0.0 | conv 1x1 | 30x96x112x112 | 0.3420 | 1.156 | 84.30 | 13.7 | 246 | 86% |
| encoder.features.2.0.block.0.1 | BatchNorm2d | 30x96x112x112 | 0.5907 | 0.145 | 144.51 | 1.0 | 245 | 85% |
| encoder.features.2.0.block.0.2 | SiLU | 30x96x112x112 | 0.5847 | 0.145 | 144.51 | 1.0 | 247 | 86% |
| encoder.features.2.0.block.1.0 | conv dw 3x3 | 30x96x56x56 | 0.4198 | 0.163 | 90.32 | 1.8 | 215 | 75% |
| encoder.features.2.0.block.1.1 | BatchNorm2d | 30x96x56x56 | 0.1638 | 0.036 | 36.13 | 1.0 | 221 | 77% |
| encoder.features.2.0.block.1.2 | SiLU | 30x96x56x56 | 0.0264 | 0.036 | 36.13 | 1.0 | 1368 | 475% |
| encoder.features.2.0.block.2.avgpool | AdaptiveAvgPool2d | 30x96x1x1 | 0.0266 | 0.009 | 18.07 | 0.5 | 679 | 236% |
| encoder.features.2.0.block.2.fc1 | conv 1x1 (SE) | 30x4x1x1 | 0.0420 | 0.000 | 0.01 | 3.4 | 0 | 0% |
| encoder.features.2.0.block.2.activation | SiLU | 30x4x1x1 | 0.0176 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.2.0.block.2.fc2 | conv 1x1 (SE) | 30x96x1x1 | 0.0461 | 0.000 | 0.01 | 3.3 | 0 | 0% |
| encoder.features.2.0.block.2.scale_activation | Sigmoid | 30x96x1x1 | 0.0205 | 0.000 | 0.01 | 1.0 | 1 | 0% |
| encoder.features.2.0.block.3.0 | conv 1x1 | 30x24x56x56 | 0.0522 | 0.434 | 22.58 | 19.2 | 432 | 150% |
| encoder.features.2.0.block.3.1 | BatchNorm2d | 30x24x56x56 | 0.0614 | 0.009 | 9.03 | 1.0 | 147 | 51% |
| encoder.features.2.1.block.0.0 | conv 1x1 | 30x144x56x56 | 0.1157 | 0.650 | 31.62 | 20.6 | 273 | 95% |
| encoder.features.2.1.block.0.1 | BatchNorm2d | 30x144x56x56 | 0.2376 | 0.054 | 54.19 | 1.0 | 228 | 79% |
| encoder.features.2.1.block.0.2 | SiLU | 30x144x56x56 | 0.0993 | 0.054 | 54.19 | 1.0 | 546 | 189% |
| encoder.features.2.1.block.1.0 | conv dw 3x3 | 30x144x56x56 | 0.4956 | 0.244 | 54.19 | 4.5 | 109 | 38% |
| encoder.features.2.1.block.1.1 | BatchNorm2d | 30x144x56x56 | 0.2519 | 0.054 | 54.19 | 1.0 | 215 | 75% |
| encoder.features.2.1.block.1.2 | SiLU | 30x144x56x56 | 0.1005 | 0.054 | 54.19 | 1.0 | 539 | 187% |
| encoder.features.2.1.block.2.avgpool | AdaptiveAvgPool2d | 30x144x1x1 | 0.0256 | 0.014 | 27.10 | 0.5 | 1060 | 368% |
| encoder.features.2.1.block.2.fc1 | conv 1x1 (SE) | 30x6x1x1 | 0.0510 | 0.000 | 0.01 | 4.8 | 0 | 0% |
| encoder.features.2.1.block.2.activation | SiLU | 30x6x1x1 | 0.0246 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.2.1.block.2.fc2 | conv 1x1 (SE) | 30x144x1x1 | 0.0533 | 0.000 | 0.01 | 4.7 | 0 | 0% |
| encoder.features.2.1.block.2.scale_activation | Sigmoid | 30x144x1x1 | 0.0246 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.2.1.block.3.0 | conv 1x1 | 30x24x56x56 | 0.0911 | 0.650 | 31.62 | 20.6 | 347 | 120% |
| encoder.features.2.1.block.3.1 | BatchNorm2d | 30x24x56x56 | 0.0543 | 0.009 | 9.03 | 1.0 | 166 | 58% |
| encoder.features.3.0.block.0.0 | conv 1x1 | 30x144x56x56 | 0.1270 | 0.650 | 31.62 | 20.6 | 249 | 86% |
| encoder.features.3.0.block.0.1 | BatchNorm2d | 30x144x56x56 | 0.2468 | 0.054 | 54.19 | 1.0 | 220 | 76% |
| encoder.features.3.0.block.0.2 | SiLU | 30x144x56x56 | 0.0963 | 0.054 | 54.19 | 1.0 | 563 | 195% |
| encoder.features.3.0.block.1.0 | conv dw 5x5 | 30x144x28x28 | 0.2109 | 0.169 | 33.88 | 5.0 | 161 | 56% |
| encoder.features.3.0.block.1.1 | BatchNorm2d | 30x144x28x28 | 0.0522 | 0.014 | 13.55 | 1.0 | 259 | 90% |
| encoder.features.3.0.block.1.2 | SiLU | 30x144x28x28 | 0.0184 | 0.014 | 13.55 | 1.0 | 735 | 255% |
| encoder.features.3.0.block.2.avgpool | AdaptiveAvgPool2d | 30x144x1x1 | 0.0274 | 0.003 | 6.78 | 0.5 | 248 | 86% |
| encoder.features.3.0.block.2.fc1 | conv 1x1 (SE) | 30x6x1x1 | 0.0512 | 0.000 | 0.01 | 4.8 | 0 | 0% |
| encoder.features.3.0.block.2.activation | SiLU | 30x6x1x1 | 0.0379 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.3.0.block.2.fc2 | conv 1x1 (SE) | 30x144x1x1 | 0.0508 | 0.000 | 0.01 | 4.7 | 0 | 0% |
| encoder.features.3.0.block.2.scale_activation | Sigmoid | 30x144x1x1 | 0.0234 | 0.000 | 0.02 | 1.0 | 1 | 0% |
| encoder.features.3.0.block.3.0 | conv 1x1 | 30x40x28x28 | 0.0481 | 0.271 | 8.67 | 31.3 | 180 | 63% |
| encoder.features.3.0.block.3.1 | BatchNorm2d | 30x40x28x28 | 0.0584 | 0.004 | 3.76 | 1.0 | 64 | 22% |
| encoder.features.3.1.block.0.0 | conv 1x1 | 30x240x28x28 | 0.0726 | 0.452 | 13.19 | 34.2 | 182 | 63% |
| encoder.features.3.1.block.0.1 | BatchNorm2d | 30x240x28x28 | 0.0625 | 0.023 | 22.58 | 1.0 | 362 | 126% |
| encoder.features.3.1.block.0.2 | SiLU | 30x240x28x28 | 0.0174 | 0.023 | 22.58 | 1.0 | 1297 | 450% |
| encoder.features.3.1.block.1.0 | conv dw 5x5 | 30x240x28x28 | 0.3185 | 0.282 | 22.59 | 12.5 | 71 | 25% |
| encoder.features.3.1.block.1.1 | BatchNorm2d | 30x240x28x28 | 0.0581 | 0.023 | 22.58 | 1.0 | 389 | 135% |
| encoder.features.3.1.block.1.2 | SiLU | 30x240x28x28 | 0.0266 | 0.023 | 22.58 | 1.0 | 848 | 294% |
| encoder.features.3.1.block.2.avgpool | AdaptiveAvgPool2d | 30x240x1x1 | 0.0266 | 0.006 | 11.30 | 0.5 | 425 | 147% |
| encoder.features.3.1.block.2.fc1 | conv 1x1 (SE) | 30x10x1x1 | 0.0512 | 0.000 | 0.02 | 7.3 | 0 | 0% |
| encoder.features.3.1.block.2.activation | SiLU | 30x10x1x1 | 0.0287 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.3.1.block.2.fc2 | conv 1x1 (SE) | 30x240x1x1 | 0.0451 | 0.000 | 0.02 | 7.1 | 0 | 0% |
| encoder.features.3.1.block.2.scale_activation | Sigmoid | 30x240x1x1 | 0.0231 | 0.000 | 0.03 | 1.0 | 1 | 0% |
| encoder.features.3.1.block.3.0 | conv 1x1 | 30x40x28x28 | 0.0644 | 0.452 | 13.19 | 34.2 | 205 | 71% |
| encoder.features.3.1.block.3.1 | BatchNorm2d | 30x40x28x28 | 0.0543 | 0.004 | 3.76 | 1.0 | 69 | 24% |
| encoder.features.4.0.block.0.0 | conv 1x1 | 30x240x28x28 | 0.0686 | 0.452 | 13.19 | 34.2 | 192 | 67% |
| encoder.features.4.0.block.0.1 | BatchNorm2d | 30x240x28x28 | 0.0563 | 0.023 | 22.58 | 1.0 | 401 | 139% |
| encoder.features.4.0.block.0.2 | SiLU | 30x240x28x28 | 0.0182 | 0.023 | 22.58 | 1.0 | 1240 | 431% |
| encoder.features.4.0.block.1.0 | conv dw 3x3 | 30x240x14x14 | 0.0624 | 0.025 | 14.12 | 1.8 | 226 | 79% |
| encoder.features.4.0.block.1.1 | BatchNorm2d | 30x240x14x14 | 0.0520 | 0.006 | 5.65 | 1.0 | 109 | 38% |
| encoder.features.4.0.block.1.2 | SiLU | 30x240x14x14 | 0.0204 | 0.006 | 5.64 | 1.0 | 276 | 96% |
| encoder.features.4.0.block.2.avgpool | AdaptiveAvgPool2d | 30x240x1x1 | 0.0289 | 0.001 | 2.84 | 0.5 | 98 | 34% |
| encoder.features.4.0.block.2.fc1 | conv 1x1 (SE) | 30x10x1x1 | 0.0526 | 0.000 | 0.02 | 7.3 | 0 | 0% |
| encoder.features.4.0.block.2.activation | SiLU | 30x10x1x1 | 0.0214 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.4.0.block.2.fc2 | conv 1x1 (SE) | 30x240x1x1 | 0.0409 | 0.000 | 0.02 | 7.1 | 0 | 0% |
| encoder.features.4.0.block.2.scale_activation | Sigmoid | 30x240x1x1 | 0.0338 | 0.000 | 0.03 | 1.0 | 1 | 0% |
| encoder.features.4.0.block.3.0 | conv 1x1 | 30x80x14x14 | 0.0440 | 0.226 | 3.80 | 59.4 | 86 | 30% |
| encoder.features.4.0.block.3.1 | BatchNorm2d | 30x80x14x14 | 0.0659 | 0.002 | 1.88 | 1.0 | 29 | 10% |
| encoder.features.4.1.block.0.0 | conv 1x1 | 30x480x14x14 | 0.0532 | 0.452 | 6.66 | 67.8 | 125 | 43% |
| encoder.features.4.1.block.0.1 | BatchNorm2d | 30x480x14x14 | 0.0481 | 0.011 | 11.29 | 1.0 | 235 | 81% |
| encoder.features.4.1.block.0.2 | SiLU | 30x480x14x14 | 0.0174 | 0.011 | 11.29 | 1.0 | 649 | 225% |
| encoder.features.4.1.block.1.0 | conv dw 3x3 | 30x480x14x14 | 0.1065 | 0.051 | 11.30 | 4.5 | 106 | 37% |
| encoder.features.4.1.block.1.1 | BatchNorm2d | 30x480x14x14 | 0.0492 | 0.011 | 11.29 | 1.0 | 229 | 80% |
| encoder.features.4.1.block.1.2 | SiLU | 30x480x14x14 | 0.0166 | 0.011 | 11.29 | 1.0 | 678 | 236% |
| encoder.features.4.1.block.2.avgpool | AdaptiveAvgPool2d | 30x480x1x1 | 0.0277 | 0.003 | 5.67 | 0.5 | 204 | 71% |
| encoder.features.4.1.block.2.fc1 | conv 1x1 (SE) | 30x20x1x1 | 0.0501 | 0.001 | 0.05 | 11.7 | 1 | 0% |
| encoder.features.4.1.block.2.activation | SiLU | 30x20x1x1 | 0.0211 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.4.1.block.2.fc2 | conv 1x1 (SE) | 30x480x1x1 | 0.0499 | 0.001 | 0.05 | 11.5 | 1 | 0% |
| encoder.features.4.1.block.2.scale_activation | Sigmoid | 30x480x1x1 | 0.0225 | 0.000 | 0.06 | 1.0 | 3 | 1% |
| encoder.features.4.1.block.3.0 | conv 1x1 | 30x80x14x14 | 0.0512 | 0.452 | 6.66 | 67.8 | 130 | 45% |
| encoder.features.4.1.block.3.1 | BatchNorm2d | 30x80x14x14 | 0.0543 | 0.002 | 1.88 | 1.0 | 35 | 12% |
| encoder.features.4.2.block.0.0 | conv 1x1 | 30x480x14x14 | 0.0652 | 0.452 | 6.66 | 67.8 | 102 | 35% |
| encoder.features.4.2.block.0.1 | BatchNorm2d | 30x480x14x14 | 0.0545 | 0.011 | 11.29 | 1.0 | 207 | 72% |
| encoder.features.4.2.block.0.2 | SiLU | 30x480x14x14 | 0.0266 | 0.011 | 11.29 | 1.0 | 424 | 147% |
| encoder.features.4.2.block.1.0 | conv dw 3x3 | 30x480x14x14 | 0.1106 | 0.051 | 11.30 | 4.5 | 102 | 35% |
| encoder.features.4.2.block.1.1 | BatchNorm2d | 30x480x14x14 | 0.0522 | 0.011 | 11.29 | 1.0 | 216 | 75% |
| encoder.features.4.2.block.1.2 | SiLU | 30x480x14x14 | 0.0277 | 0.011 | 11.29 | 1.0 | 408 | 142% |
| encoder.features.4.2.block.2.avgpool | AdaptiveAvgPool2d | 30x480x1x1 | 0.0286 | 0.003 | 5.67 | 0.5 | 198 | 69% |
| encoder.features.4.2.block.2.fc1 | conv 1x1 (SE) | 30x20x1x1 | 0.0532 | 0.001 | 0.05 | 11.7 | 1 | 0% |
| encoder.features.4.2.block.2.activation | SiLU | 30x20x1x1 | 0.0207 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.4.2.block.2.fc2 | conv 1x1 (SE) | 30x480x1x1 | 0.0504 | 0.001 | 0.05 | 11.5 | 1 | 0% |
| encoder.features.4.2.block.2.scale_activation | Sigmoid | 30x480x1x1 | 0.0236 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| encoder.features.4.2.block.3.0 | conv 1x1 | 30x80x14x14 | 0.0512 | 0.452 | 6.66 | 67.8 | 130 | 45% |
| encoder.features.4.2.block.3.1 | BatchNorm2d | 30x80x14x14 | 0.0553 | 0.002 | 1.88 | 1.0 | 34 | 12% |
| encoder.features.5.0.block.0.0 | conv 1x1 | 30x480x14x14 | 0.0551 | 0.452 | 6.66 | 67.8 | 121 | 42% |
| encoder.features.5.0.block.0.1 | BatchNorm2d | 30x480x14x14 | 0.0584 | 0.011 | 11.29 | 1.0 | 193 | 67% |
| encoder.features.5.0.block.0.2 | SiLU | 30x480x14x14 | 0.0174 | 0.011 | 11.29 | 1.0 | 649 | 225% |
| encoder.features.5.0.block.1.0 | conv dw 5x5 | 30x480x14x14 | 0.1669 | 0.141 | 11.31 | 12.5 | 68 | 24% |
| encoder.features.5.0.block.1.1 | BatchNorm2d | 30x480x14x14 | 0.0481 | 0.011 | 11.29 | 1.0 | 235 | 81% |
| encoder.features.5.0.block.1.2 | SiLU | 30x480x14x14 | 0.0174 | 0.011 | 11.29 | 1.0 | 649 | 225% |
| encoder.features.5.0.block.2.avgpool | AdaptiveAvgPool2d | 30x480x1x1 | 0.0287 | 0.003 | 5.67 | 0.5 | 198 | 69% |
| encoder.features.5.0.block.2.fc1 | conv 1x1 (SE) | 30x20x1x1 | 0.0481 | 0.001 | 0.05 | 11.7 | 1 | 0% |
| encoder.features.5.0.block.2.activation | SiLU | 30x20x1x1 | 0.0205 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.5.0.block.2.fc2 | conv 1x1 (SE) | 30x480x1x1 | 0.0473 | 0.001 | 0.05 | 11.5 | 1 | 0% |
| encoder.features.5.0.block.2.scale_activation | Sigmoid | 30x480x1x1 | 0.0226 | 0.000 | 0.06 | 1.0 | 3 | 1% |
| encoder.features.5.0.block.3.0 | conv 1x1 | 30x112x14x14 | 0.0528 | 0.632 | 7.07 | 89.4 | 134 | 46% |
| encoder.features.5.0.block.3.1 | BatchNorm2d | 30x112x14x14 | 0.0532 | 0.003 | 2.64 | 1.0 | 50 | 17% |
| encoder.features.5.1.block.0.0 | conv 1x1 | 30x672x14x14 | 0.0737 | 0.885 | 9.37 | 94.5 | 127 | 44% |
| encoder.features.5.1.block.0.1 | BatchNorm2d | 30x672x14x14 | 0.0500 | 0.016 | 15.81 | 1.0 | 316 | 110% |
| encoder.features.5.1.block.0.2 | SiLU | 30x672x14x14 | 0.0142 | 0.016 | 15.81 | 1.0 | 1110 | 385% |
| encoder.features.5.1.block.1.0 | conv dw 5x5 | 30x672x14x14 | 0.2253 | 0.198 | 15.84 | 12.5 | 70 | 24% |
| encoder.features.5.1.block.1.1 | BatchNorm2d | 30x672x14x14 | 0.0604 | 0.016 | 15.81 | 1.0 | 262 | 91% |
| encoder.features.5.1.block.1.2 | SiLU | 30x672x14x14 | 0.0328 | 0.016 | 15.81 | 1.0 | 482 | 167% |
| encoder.features.5.1.block.2.avgpool | AdaptiveAvgPool2d | 30x672x1x1 | 0.0349 | 0.004 | 7.94 | 0.5 | 228 | 79% |
| encoder.features.5.1.block.2.fc1 | conv 1x1 (SE) | 30x28x1x1 | 0.0502 | 0.001 | 0.08 | 14.2 | 2 | 1% |
| encoder.features.5.1.block.2.activation | SiLU | 30x28x1x1 | 0.0215 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.5.1.block.2.fc2 | conv 1x1 (SE) | 30x672x1x1 | 0.0563 | 0.001 | 0.08 | 13.9 | 1 | 0% |
| encoder.features.5.1.block.2.scale_activation | Sigmoid | 30x672x1x1 | 0.0236 | 0.000 | 0.08 | 1.0 | 3 | 1% |
| encoder.features.5.1.block.3.0 | conv 1x1 | 30x112x14x14 | 0.0645 | 0.885 | 9.37 | 94.5 | 145 | 50% |
| encoder.features.5.1.block.3.1 | BatchNorm2d | 30x112x14x14 | 0.0543 | 0.003 | 2.64 | 1.0 | 49 | 17% |
| encoder.features.5.2.block.0.0 | conv 1x1 | 30x672x14x14 | 0.0788 | 0.885 | 9.37 | 94.5 | 119 | 41% |
| encoder.features.5.2.block.0.1 | BatchNorm2d | 30x672x14x14 | 0.0502 | 0.016 | 15.81 | 1.0 | 315 | 109% |
| encoder.features.5.2.block.0.2 | SiLU | 30x672x14x14 | 0.0203 | 0.016 | 15.81 | 1.0 | 780 | 271% |
| encoder.features.5.2.block.1.0 | conv dw 5x5 | 30x672x14x14 | 0.2273 | 0.198 | 15.84 | 12.5 | 70 | 24% |
| encoder.features.5.2.block.1.1 | BatchNorm2d | 30x672x14x14 | 0.0503 | 0.016 | 15.81 | 1.0 | 314 | 109% |
| encoder.features.5.2.block.1.2 | SiLU | 30x672x14x14 | 0.0195 | 0.016 | 15.81 | 1.0 | 812 | 282% |
| encoder.features.5.2.block.2.avgpool | AdaptiveAvgPool2d | 30x672x1x1 | 0.0266 | 0.004 | 7.94 | 0.5 | 298 | 104% |
| encoder.features.5.2.block.2.fc1 | conv 1x1 (SE) | 30x28x1x1 | 0.0483 | 0.001 | 0.08 | 14.2 | 2 | 1% |
| encoder.features.5.2.block.2.activation | SiLU | 30x28x1x1 | 0.0202 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.5.2.block.2.fc2 | conv 1x1 (SE) | 30x672x1x1 | 0.0535 | 0.001 | 0.08 | 13.9 | 2 | 1% |
| encoder.features.5.2.block.2.scale_activation | Sigmoid | 30x672x1x1 | 0.0225 | 0.000 | 0.08 | 1.0 | 4 | 1% |
| encoder.features.5.2.block.3.0 | conv 1x1 | 30x112x14x14 | 0.0643 | 0.885 | 9.37 | 94.5 | 146 | 51% |
| encoder.features.5.2.block.3.1 | BatchNorm2d | 30x112x14x14 | 0.0528 | 0.003 | 2.64 | 1.0 | 50 | 17% |
| encoder.features.6.0.block.0.0 | conv 1x1 | 30x672x14x14 | 0.0774 | 0.885 | 9.37 | 94.5 | 121 | 42% |
| encoder.features.6.0.block.0.1 | BatchNorm2d | 30x672x14x14 | 0.0480 | 0.016 | 15.81 | 1.0 | 329 | 114% |
| encoder.features.6.0.block.0.2 | SiLU | 30x672x14x14 | 0.0196 | 0.016 | 15.81 | 1.0 | 807 | 280% |
| encoder.features.6.0.block.1.0 | conv dw 5x5 | 30x672x7x7 | 0.0694 | 0.049 | 9.91 | 5.0 | 143 | 50% |
| encoder.features.6.0.block.1.1 | BatchNorm2d | 30x672x7x7 | 0.0518 | 0.004 | 3.96 | 1.0 | 76 | 27% |
| encoder.features.6.0.block.1.2 | SiLU | 30x672x7x7 | 0.0258 | 0.004 | 3.95 | 1.0 | 153 | 53% |
| encoder.features.6.0.block.2.avgpool | AdaptiveAvgPool2d | 30x672x1x1 | 0.0274 | 0.001 | 2.02 | 0.5 | 74 | 26% |
| encoder.features.6.0.block.2.fc1 | conv 1x1 (SE) | 30x28x1x1 | 0.0529 | 0.001 | 0.08 | 14.2 | 2 | 1% |
| encoder.features.6.0.block.2.activation | SiLU | 30x28x1x1 | 0.0209 | 0.000 | 0.00 | 1.0 | 0 | 0% |
| encoder.features.6.0.block.2.fc2 | conv 1x1 (SE) | 30x672x1x1 | 0.0615 | 0.001 | 0.08 | 13.9 | 1 | 0% |
| encoder.features.6.0.block.2.scale_activation | Sigmoid | 30x672x1x1 | 0.0225 | 0.000 | 0.08 | 1.0 | 4 | 1% |
| encoder.features.6.0.block.3.0 | conv 1x1 | 30x192x7x7 | 0.0410 | 0.379 | 2.80 | 135.6 | 68 | 24% |
| encoder.features.6.0.block.3.1 | BatchNorm2d | 30x192x7x7 | 0.0548 | 0.001 | 1.13 | 1.0 | 21 | 7% |
| encoder.features.6.1.block.0.0 | conv 1x1 | 30x1152x7x7 | 0.0604 | 0.650 | 4.39 | 148.0 | 73 | 25% |
| encoder.features.6.1.block.0.1 | BatchNorm2d | 30x1152x7x7 | 0.0707 | 0.007 | 6.78 | 1.0 | 96 | 33% |
| encoder.features.6.1.block.0.2 | SiLU | 30x1152x7x7 | 0.0213 | 0.007 | 6.77 | 1.0 | 317 | 110% |
| encoder.features.6.1.block.1.0 | conv dw 5x5 | 30x1152x7x7 | 0.1065 | 0.085 | 6.83 | 12.4 | 64 | 22% |
| encoder.features.6.1.block.1.1 | BatchNorm2d | 30x1152x7x7 | 0.0672 | 0.007 | 6.78 | 1.0 | 101 | 35% |
| encoder.features.6.1.block.1.2 | SiLU | 30x1152x7x7 | 0.0358 | 0.007 | 6.77 | 1.0 | 189 | 66% |
| encoder.features.6.1.block.2.avgpool | AdaptiveAvgPool2d | 30x1152x1x1 | 0.0478 | 0.002 | 3.46 | 0.5 | 72 | 25% |
| encoder.features.6.1.block.2.fc1 | conv 1x1 (SE) | 30x48x1x1 | 0.0533 | 0.003 | 0.18 | 18.2 | 3 | 1% |
| encoder.features.6.1.block.2.activation | SiLU | 30x48x1x1 | 0.0215 | 0.000 | 0.01 | 1.0 | 0 | 0% |
| encoder.features.6.1.block.2.fc2 | conv 1x1 (SE) | 30x1152x1x1 | 0.0502 | 0.003 | 0.18 | 17.9 | 4 | 1% |
| encoder.features.6.1.block.2.scale_activation | Sigmoid | 30x1152x1x1 | 0.0244 | 0.000 | 0.14 | 1.0 | 6 | 2% |
| encoder.features.6.1.block.3.0 | conv 1x1 | 30x192x7x7 | 0.0594 | 0.650 | 4.39 | 148.0 | 74 | 26% |
| encoder.features.6.1.block.3.1 | BatchNorm2d | 30x192x7x7 | 0.0634 | 0.001 | 1.13 | 1.0 | 18 | 6% |
| encoder.features.6.2.block.0.0 | conv 1x1 | 30x1152x7x7 | 0.0584 | 0.650 | 4.39 | 148.0 | 75 | 26% |
| encoder.features.6.2.block.0.1 | BatchNorm2d | 30x1152x7x7 | 0.0590 | 0.007 | 6.78 | 1.0 | 115 | 40% |
| encoder.features.6.2.block.0.2 | SiLU | 30x1152x7x7 | 0.0349 | 0.007 | 6.77 | 1.0 | 194 | 67% |
| encoder.features.6.2.block.1.0 | conv dw 5x5 | 30x1152x7x7 | 0.1115 | 0.085 | 6.83 | 12.4 | 61 | 21% |
| encoder.features.6.2.block.1.1 | BatchNorm2d | 30x1152x7x7 | 0.0573 | 0.007 | 6.78 | 1.0 | 118 | 41% |
| encoder.features.6.2.block.1.2 | SiLU | 30x1152x7x7 | 0.0214 | 0.007 | 6.77 | 1.0 | 316 | 110% |
| encoder.features.6.2.block.2.avgpool | AdaptiveAvgPool2d | 30x1152x1x1 | 0.0369 | 0.002 | 3.46 | 0.5 | 94 | 33% |
| encoder.features.6.2.block.2.fc1 | conv 1x1 (SE) | 30x48x1x1 | 0.0502 | 0.003 | 0.18 | 18.2 | 4 | 1% |
| encoder.features.6.2.block.2.activation | SiLU | 30x48x1x1 | 0.0212 | 0.000 | 0.01 | 1.0 | 0 | 0% |
| encoder.features.6.2.block.2.fc2 | conv 1x1 (SE) | 30x1152x1x1 | 0.0532 | 0.003 | 0.18 | 17.9 | 3 | 1% |
| encoder.features.6.2.block.2.scale_activation | Sigmoid | 30x1152x1x1 | 0.0225 | 0.000 | 0.14 | 1.0 | 6 | 2% |
| encoder.features.6.2.block.3.0 | conv 1x1 | 30x192x7x7 | 0.0584 | 0.650 | 4.39 | 148.0 | 75 | 26% |
| encoder.features.6.2.block.3.1 | BatchNorm2d | 30x192x7x7 | 0.0500 | 0.001 | 1.13 | 1.0 | 23 | 8% |
| encoder.features.6.3.block.0.0 | conv 1x1 | 30x1152x7x7 | 0.0584 | 0.650 | 4.39 | 148.0 | 75 | 26% |
| encoder.features.6.3.block.0.1 | BatchNorm2d | 30x1152x7x7 | 0.0584 | 0.007 | 6.78 | 1.0 | 116 | 40% |
| encoder.features.6.3.block.0.2 | SiLU | 30x1152x7x7 | 0.0214 | 0.007 | 6.77 | 1.0 | 316 | 110% |
| encoder.features.6.3.block.1.0 | conv dw 5x5 | 30x1152x7x7 | 0.1075 | 0.085 | 6.83 | 12.4 | 64 | 22% |
| encoder.features.6.3.block.1.1 | BatchNorm2d | 30x1152x7x7 | 0.0603 | 0.007 | 6.78 | 1.0 | 113 | 39% |
| encoder.features.6.3.block.1.2 | SiLU | 30x1152x7x7 | 0.0285 | 0.007 | 6.77 | 1.0 | 237 | 82% |
| encoder.features.6.3.block.2.avgpool | AdaptiveAvgPool2d | 30x1152x1x1 | 0.0245 | 0.002 | 3.46 | 0.5 | 141 | 49% |
| encoder.features.6.3.block.2.fc1 | conv 1x1 (SE) | 30x48x1x1 | 0.0515 | 0.003 | 0.18 | 18.2 | 4 | 1% |
| encoder.features.6.3.block.2.activation | SiLU | 30x48x1x1 | 0.0205 | 0.000 | 0.01 | 1.0 | 0 | 0% |
| encoder.features.6.3.block.2.fc2 | conv 1x1 (SE) | 30x1152x1x1 | 0.0563 | 0.003 | 0.18 | 17.9 | 3 | 1% |
| encoder.features.6.3.block.2.scale_activation | Sigmoid | 30x1152x1x1 | 0.0268 | 0.000 | 0.14 | 1.0 | 5 | 2% |
| encoder.features.6.3.block.3.0 | conv 1x1 | 30x192x7x7 | 0.0686 | 0.650 | 4.39 | 148.0 | 64 | 22% |
| encoder.features.6.3.block.3.1 | BatchNorm2d | 30x192x7x7 | 0.0594 | 0.001 | 1.13 | 1.0 | 19 | 7% |
| encoder.features.7.0.block.0.0 | conv 1x1 | 30x1152x7x7 | 0.0584 | 0.650 | 4.39 | 148.0 | 75 | 26% |
| encoder.features.7.0.block.0.1 | BatchNorm2d | 30x1152x7x7 | 0.0635 | 0.007 | 6.78 | 1.0 | 107 | 37% |
| encoder.features.7.0.block.0.2 | SiLU | 30x1152x7x7 | 0.0217 | 0.007 | 6.77 | 1.0 | 312 | 108% |
| encoder.features.7.0.block.1.0 | conv dw 3x3 | 30x1152x7x7 | 0.0714 | 0.030 | 6.79 | 4.5 | 95 | 33% |
| encoder.features.7.0.block.1.1 | BatchNorm2d | 30x1152x7x7 | 0.0860 | 0.007 | 6.78 | 1.0 | 79 | 27% |
| encoder.features.7.0.block.1.2 | SiLU | 30x1152x7x7 | 0.0215 | 0.007 | 6.77 | 1.0 | 315 | 109% |
| encoder.features.7.0.block.2.avgpool | AdaptiveAvgPool2d | 30x1152x1x1 | 0.0349 | 0.002 | 3.46 | 0.5 | 99 | 34% |
| encoder.features.7.0.block.2.fc1 | conv 1x1 (SE) | 30x48x1x1 | 0.0514 | 0.003 | 0.18 | 18.2 | 4 | 1% |
| encoder.features.7.0.block.2.activation | SiLU | 30x48x1x1 | 0.0227 | 0.000 | 0.01 | 1.0 | 0 | 0% |
| encoder.features.7.0.block.2.fc2 | conv 1x1 (SE) | 30x1152x1x1 | 0.0551 | 0.003 | 0.18 | 17.9 | 3 | 1% |
| encoder.features.7.0.block.2.scale_activation | Sigmoid | 30x1152x1x1 | 0.0232 | 0.000 | 0.14 | 1.0 | 6 | 2% |
| encoder.features.7.0.block.3.0 | conv 1x1 | 30x320x7x7 | 0.0696 | 1.084 | 5.06 | 214.0 | 73 | 25% |
| encoder.features.7.0.block.3.1 | BatchNorm2d | 30x320x7x7 | 0.0532 | 0.002 | 1.88 | 1.0 | 35 | 12% |
| encoder.features.8.0 | conv 1x1 | 30x1280x7x7 | 0.0737 | 1.204 | 5.52 | 218.0 | 75 | 26% |
| encoder.features.8.1 | BatchNorm2d | 30x1280x7x7 | 0.0601 | 0.008 | 7.54 | 1.0 | 125 | 44% |
| encoder.features.8.2 | SiLU | 30x1280x7x7 | 0.0214 | 0.008 | 7.53 | 1.0 | 352 | 122% |
| encoder.pool | AdaptiveAvgPool2d | 30x1280x1x1 | 0.0264 | 0.002 | 3.84 | 0.5 | 146 | 51% |
| temporal.proj | Linear | 1x30x512 | 0.0421 | 0.039 | 1.42 | 27.7 | 34 | 12% |
| temporal.layers.0.norm | LayerNorm | 1x30x512 | 0.0341 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| temporal.layers.0.mixer.in_proj | Linear | 1x30x2048 | 0.0420 | 0.063 | 2.25 | 28.0 | 54 | 19% |
| temporal.layers.0.mixer.conv1d | Conv1d | 1x1024x33 | 0.0502 | 0.000 | 0.14 | 1.9 | 3 | 1% |
| temporal.layers.0.mixer.dt_proj | Linear | 1x30x1024 | 0.0541 | 0.063 | 2.22 | 28.3 | 41 | 14% |
| temporal.layers.0.mixer.x_proj | Linear | 1x30x32 | 0.0453 | 0.002 | 0.13 | 15.3 | 3 | 1% |
| temporal.layers.0.mixer.out_proj | Linear | 1x30x512 | 0.0416 | 0.031 | 1.14 | 27.6 | 27 | 10% |
| temporal.layers.1.norm | LayerNorm | 1x30x512 | 0.0338 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| temporal.layers.1.mixer.in_proj | Linear | 1x30x2048 | 0.0411 | 0.063 | 2.25 | 28.0 | 55 | 19% |
| temporal.layers.1.mixer.conv1d | Conv1d | 1x1024x33 | 0.0484 | 0.000 | 0.14 | 1.9 | 3 | 1% |
| temporal.layers.1.mixer.dt_proj | Linear | 1x30x1024 | 0.0502 | 0.063 | 2.22 | 28.3 | 44 | 15% |
| temporal.layers.1.mixer.x_proj | Linear | 1x30x32 | 0.0419 | 0.002 | 0.13 | 15.3 | 3 | 1% |
| temporal.layers.1.mixer.out_proj | Linear | 1x30x512 | 0.0420 | 0.031 | 1.14 | 27.6 | 27 | 9% |
| temporal.layers.2.norm | LayerNorm | 1x30x512 | 0.0338 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| temporal.layers.2.mixer.in_proj | Linear | 1x30x2048 | 0.0402 | 0.063 | 2.25 | 28.0 | 56 | 19% |
| temporal.layers.2.mixer.conv1d | Conv1d | 1x1024x33 | 0.0500 | 0.000 | 0.14 | 1.9 | 3 | 1% |
| temporal.layers.2.mixer.dt_proj | Linear | 1x30x1024 | 0.0561 | 0.063 | 2.22 | 28.3 | 40 | 14% |
| temporal.layers.2.mixer.x_proj | Linear | 1x30x32 | 0.0430 | 0.002 | 0.13 | 15.3 | 3 | 1% |
| temporal.layers.2.mixer.out_proj | Linear | 1x30x512 | 0.0388 | 0.031 | 1.14 | 27.6 | 29 | 10% |
| temporal.layers.3.norm | LayerNorm | 1x30x512 | 0.0385 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| temporal.layers.3.mixer.in_proj | Linear | 1x30x2048 | 0.0399 | 0.063 | 2.25 | 28.0 | 56 | 20% |
| temporal.layers.3.mixer.conv1d | Conv1d | 1x1024x33 | 0.0604 | 0.000 | 0.14 | 1.9 | 2 | 1% |
| temporal.layers.3.mixer.dt_proj | Linear | 1x30x1024 | 0.0572 | 0.063 | 2.22 | 28.3 | 39 | 13% |
| temporal.layers.3.mixer.x_proj | Linear | 1x30x32 | 0.0461 | 0.002 | 0.13 | 15.3 | 3 | 1% |
| temporal.layers.3.mixer.out_proj | Linear | 1x30x512 | 0.0411 | 0.031 | 1.14 | 27.6 | 28 | 10% |
| temporal.norm_f | LayerNorm | 1x30x512 | 0.0362 | 0.000 | 0.06 | 1.0 | 2 | 1% |
| temporal.head.0 | Linear | 1x256 | 0.0359 | 0.000 | 0.26 | 1.0 | 7 | 3% |
| temporal.head.1 | GELU | 1x256 | 0.0276 | 0.000 | 0.00 | 0.2 | 0 | 0% |
| temporal.head.3 | Linear | 1x2 | 0.0386 | 0.000 | 0.00 | 0.7 | 0 | 0% |
| temporal.layers.0.mixer.scan | selective scan (python loop) | 1x30x1024 | 4.8003 | 0.068 | 8.05 | 8.5 | 2 | 1% |
| temporal.layers.1.mixer.scan | selective scan (python loop) | 1x30x1024 | 5.4030 | 0.068 | 8.05 | 8.5 | 1 | 1% |
| temporal.layers.2.mixer.scan | selective scan (python loop) | 1x30x1024 | 5.0606 | 0.068 | 8.05 | 8.5 | 2 | 1% |
| temporal.layers.3.mixer.scan | selective scan (python loop) | 1x30x1024 | 4.8300 | 0.068 | 8.05 | 8.5 | 2 | 1% |
