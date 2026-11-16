# Per-layer real speed (v2)

All times are GPU time per call in milliseconds, measured by replaying a CUDA graph (no launch overhead). 'bf16 best' is the faster of cuBLAS (+ separate SiLU) and a Triton bf16 GEMM with the same fused epilogue as the int8 kernel, so the speedup is not inflated by fusion alone.
'Quant best' is the fastest complete quantized path that takes bf16 in and gives bf16 out, quantization of the input and dequantization included: Triton int8 GEMM with fused quantize/dequantize, torch._int_mm + quantize + dequantize kernels, or torch._scaled_mm FP8 + quantize kernel.

Provenance: {"idle_samples": [1, 5, 5], "gpu_before": "5 %, 210 MHz, 405 MHz, 43, 8.47 W, 1687 MiB", "gpu_after": "98 %, 2805 MHz, 8751 MHz, 59, 74.54 W, 2778 MiB", "torch": "2.14.0+cu130", "timing": "median of 15 CUDA-graph replays of 20 back-to-back calls (graph_ms) and median of 60 eager calls with CUDA events (eager_ms)"}

## Batch 1: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 1x32x8 | 0.0013 | bf16_triton | 0.0014 | int8_triton_bf16in | 0.93x | 0.0039 | 0.0022 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 1x8x32 | 0.0014 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.05x | 0.0040 | 0.0021 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 16384x32x16 | 0.0037 | bf16_cublas | 0.0031 | int8_triton_bf16in | 1.22x | 0.0056 | 0.0094 | 0.0055 |
| trunk.2.0.block.0.0 | pw1x1 | 16384x16x96 | 0.0083 | bf16_triton | 0.0089 | int8_triton_bf16in | 0.94x | 0.0114 | 0.0242 | 0.0104 |
| trunk.2.0.block.2.fc1 | se_fc | 1x96x4 | 0.0024 | bf16_triton | 0.0020 | int8_triton_bf16in | 1.19x | 0.0040 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 1x4x96 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.08x | 0.0040 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 4096x96x24 | 0.0029 | bf16_cublas | 0.0027 | int8_triton_bf16in | 1.08x | 0.0052 | 0.0034 | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 4096x24x144 | 0.0061 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.31x | 0.0070 | 0.0110 | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 1x144x6 | 0.0017 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.89x | 0.0041 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 1x6x144 | 0.0015 | bf16_triton | 0.0018 | int8_triton_bf16in | 0.83x | 0.0042 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 4096x144x24 | 0.0035 | bf16_cublas | 0.0048 | int8_triton_bf16in | 0.72x | 0.0060 | 0.0045 | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 4096x24x144 | 0.0061 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.31x | 0.0070 | 0.0110 | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 1x144x6 | 0.0017 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.89x | 0.0041 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 1x6x144 | 0.0015 | bf16_triton | 0.0018 | int8_triton_bf16in | 0.83x | 0.0042 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 1024x144x40 | 0.0024 | bf16_triton | 0.0030 | int8_triton_bf16in | 0.81x | 0.0052 | 0.0034 | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 1024x40x240 | 0.0036 | bf16_triton | 0.0034 | int8_triton_bf16in | 1.06x | 0.0058 | 0.0033 | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 1x240x10 | 0.0019 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.88x | 0.0045 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 1x10x240 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.99x | 0.0044 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 1024x240x40 | 0.0034 | bf16_cublas | 0.0035 | int8_triton_bf16in | 0.98x | 0.0052 | 0.0034 | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 1024x40x240 | 0.0036 | bf16_triton | 0.0034 | int8_triton_bf16in | 1.06x | 0.0058 | 0.0033 | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 1x240x10 | 0.0019 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.88x | 0.0045 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 1x10x240 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.99x | 0.0044 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 256x240x80 | 0.0024 | bf16_triton | 0.0026 | int8_triton_bf16in | 0.91x | 0.0048 | 0.0027 | 0.0024 |
| trunk.4.1.block.0.0 | pw1x1 | 256x80x480 | 0.0026 | bf16_triton | 0.0025 | int8_triton_bf16in | 1.02x | 0.0054 | 0.0033 | 0.0026 |
| trunk.4.1.block.2.fc1 | se_fc | 1x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.79x | 0.0046 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 1x20x480 | 0.0017 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.08x | 0.0043 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 256x480x80 | 0.0030 | bf16_triton | 0.0031 | int8_triton_bf16in | 0.98x | 0.0048 | 0.0035 | 0.0027 |
| trunk.4.2.block.0.0 | pw1x1 | 256x80x480 | 0.0026 | bf16_triton | 0.0025 | int8_triton_bf16in | 1.02x | 0.0054 | 0.0033 | 0.0026 |
| trunk.4.2.block.2.fc1 | se_fc | 1x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.79x | 0.0046 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 1x20x480 | 0.0017 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.08x | 0.0043 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 256x480x80 | 0.0030 | bf16_triton | 0.0031 | int8_triton_bf16in | 0.98x | 0.0048 | 0.0035 | 0.0027 |
| trunk.5.0.block.0.0 | pw1x1 | 256x80x480 | 0.0026 | bf16_triton | 0.0025 | int8_triton_bf16in | 1.02x | 0.0054 | 0.0033 | 0.0026 |
| trunk.5.0.block.2.fc1 | se_fc | 1x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.79x | 0.0046 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 1x20x480 | 0.0017 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.08x | 0.0043 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 256x480x112 | 0.0034 | bf16_triton | 0.0032 | int8_triton_bf16in | 1.06x | 0.0052 | 0.0036 | 0.0027 |
| trunk.5.1.block.0.0 | pw1x1 | 256x112x672 | 0.0036 | bf16_triton | 0.0034 | int8_triton_bf16in | 1.06x | 0.0062 | 0.0047 | 0.0031 |
| trunk.5.1.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0039 | int8_triton_bf16in | 0.71x | 0.0050 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0016 | int8_triton_bf16in | 0.94x | 0.0044 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 256x672x112 | 0.0042 | bf16_triton | 0.0039 | int8_triton_bf16in | 1.08x | 0.0055 | 0.0042 | 0.0030 |
| trunk.5.2.block.0.0 | pw1x1 | 256x112x672 | 0.0036 | bf16_triton | 0.0034 | int8_triton_bf16in | 1.06x | 0.0062 | 0.0047 | 0.0031 |
| trunk.5.2.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0039 | int8_triton_bf16in | 0.71x | 0.0050 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0016 | int8_triton_bf16in | 0.94x | 0.0044 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 256x672x112 | 0.0042 | bf16_triton | 0.0039 | int8_triton_bf16in | 1.08x | 0.0055 | 0.0042 | 0.0030 |
| trunk.6.0.block.0.0 | pw1x1 | 256x112x672 | 0.0036 | bf16_triton | 0.0034 | int8_triton_bf16in | 1.06x | 0.0062 | 0.0047 | 0.0031 |
| trunk.6.0.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0039 | int8_triton_bf16in | 0.71x | 0.0050 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0016 | int8_triton_bf16in | 0.94x | 0.0044 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 64x672x192 | 0.0040 | bf16_triton | 0.0038 | int8_triton_bf16in | 1.05x | 0.0050 | 0.0039 | 0.0029 |
| trunk.6.1.block.0.0 | pw1x1 | 64x192x1152 | 0.0031 | bf16_triton | 0.0033 | int8_triton_bf16in | 0.93x | 0.0053 | 0.0039 | 0.0027 |
| trunk.6.1.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0048 | int8_triton_bf16in | 0.59x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.1.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.1.block.3.0 | pw1x1 | 64x1152x192 | 0.0041 | bf16_triton | 0.0046 | fp8_pipeline | 0.89x | 0.0066 | 0.0049 | 0.0036 |
| trunk.6.2.block.0.0 | pw1x1 | 64x192x1152 | 0.0031 | bf16_triton | 0.0033 | int8_triton_bf16in | 0.93x | 0.0053 | 0.0039 | 0.0027 |
| trunk.6.2.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0048 | int8_triton_bf16in | 0.59x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.2.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.2.block.3.0 | pw1x1 | 64x1152x192 | 0.0041 | bf16_triton | 0.0046 | fp8_pipeline | 0.89x | 0.0066 | 0.0049 | 0.0036 |
| trunk.6.3.block.0.0 | pw1x1 | 64x192x1152 | 0.0031 | bf16_triton | 0.0033 | int8_triton_bf16in | 0.93x | 0.0053 | 0.0039 | 0.0027 |
| trunk.6.3.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0048 | int8_triton_bf16in | 0.59x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.3.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.3.block.3.0 | pw1x1 | 64x1152x192 | 0.0041 | bf16_triton | 0.0046 | fp8_pipeline | 0.89x | 0.0066 | 0.0049 | 0.0036 |
| trunk.7.0.block.0.0 | pw1x1 | 64x192x1152 | 0.0031 | bf16_triton | 0.0033 | int8_triton_bf16in | 0.93x | 0.0053 | 0.0039 | 0.0027 |
| trunk.7.0.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0048 | int8_triton_bf16in | 0.59x | 0.0052 | 0.0049 | 0.0036 |
| trunk.7.0.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.7.0.block.3.0 | pw1x1 | 64x1152x320 | 0.0047 | bf16_triton | 0.0047 | fp8_pipeline | 0.99x | 0.0080 | 0.0052 | 0.0037 |
| trunk.8.0 | pw1x1 | 64x320x1280 | 0.0042 | bf16_triton | 0.0044 | int8_triton_bf16in | 0.96x | 0.0062 | 0.0049 | 0.0031 |
| head.1 | head_fc | 1x1280x2 | 0.0019 | bf16_cublas | 0.0054 | int8_triton_bf16in | 0.35x | 0.0057 | n/a | n/a |

## Batch 1: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 1x128x128x32, 3, s1 | 0.0125 | 0.0057 | 0.0068 | 0.0090 | 0.0112 | 369 | 175 | 0.63x | 0.51x |
| trunk.2.0.block.1.0 | 1x128x128x96, 3, s2 | 0.0104 | 0.0835 | 0.0075 | 0.0087 | 0.0139 | 47 | 271 | 0.86x | 0.54x |
| trunk.2.1.block.1.0 | 1x64x64x144, 3, s1 | 0.0123 | 0.0086 | 0.0075 | 0.0106 | 0.0137 | 275 | 166 | 0.70x | 0.55x |
| trunk.3.0.block.1.0 | 1x64x64x144, 5, s2 | 0.0785 | 0.0069 | 0.0076 | 0.0079 | 0.0110 | 214 | 112 | 0.87x | 0.63x |
| trunk.3.1.block.1.0 | 1x32x32x240, 5, s1 | 0.0120 | 0.0084 | 0.0070 | 0.0095 | 0.0115 | 119 | 78 | 0.73x | 0.61x |
| trunk.4.0.block.1.0 | 1x32x32x240, 3, s2 | 0.0055 | 0.0027 | 0.0021 | 0.0022 | 0.0042 | 233 | 165 | 0.94x | 0.50x |
| trunk.4.1.block.1.0 | 1x16x16x480, 3, s1 | 0.0055 | 0.0032 | 0.0026 | 0.0033 | 0.0048 | 158 | 112 | 0.78x | 0.55x |
| trunk.4.2.block.1.0 | 1x16x16x480, 3, s1 | 0.0055 | 0.0032 | 0.0026 | 0.0033 | 0.0048 | 158 | 112 | 0.78x | 0.55x |
| trunk.5.0.block.1.0 | 1x16x16x480, 5, s1 | 0.0094 | 0.0039 | 0.0041 | 0.0064 | 0.0079 | 131 | 59 | 0.62x | 0.50x |
| trunk.5.1.block.1.0 | 1x16x16x672, 5, s1 | 0.0103 | 0.0057 | 0.0052 | 0.0078 | 0.0097 | 126 | 68 | 0.66x | 0.53x |
| trunk.5.2.block.1.0 | 1x16x16x672, 5, s1 | 0.0103 | 0.0057 | 0.0052 | 0.0078 | 0.0097 | 126 | 68 | 0.66x | 0.53x |
| trunk.6.0.block.1.0 | 1x16x16x672, 5, s2 | 0.0410 | 0.0327 | 0.0133 | 0.0046 | 0.0067 | 14 | 60 | 2.88x | 1.98x |
| trunk.6.1.block.1.0 | 1x8x8x1152, 5, s1 | 0.0077 | 0.0040 | 0.0036 | 0.0052 | 0.0067 | 87 | 48 | 0.70x | 0.55x |
| trunk.6.2.block.1.0 | 1x8x8x1152, 5, s1 | 0.0077 | 0.0040 | 0.0036 | 0.0052 | 0.0067 | 87 | 48 | 0.70x | 0.55x |
| trunk.6.3.block.1.0 | 1x8x8x1152, 5, s1 | 0.0077 | 0.0040 | 0.0036 | 0.0052 | 0.0067 | 87 | 48 | 0.70x | 0.55x |
| trunk.7.0.block.1.0 | 1x8x8x1152, 3, s1 | 0.0050 | 0.0029 | 0.0022 | 0.0024 | 0.0038 | 108 | 96 | 0.89x | 0.56x |

## Batch 8: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 8x32x8 | 0.0013 | bf16_triton | 0.0014 | int8_triton_bf16in | 0.93x | 0.0061 | 0.0022 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 8x8x32 | 0.0018 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.29x | 0.0042 | 0.0021 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 131072x32x16 | 0.0205 | bf16_triton | 0.0138 | int8_triton_bf16in | 1.49x | 0.0231 | n/a | 0.0309 |
| trunk.2.0.block.0.0 | pw1x1 | 131072x16x96 | 0.0547 | bf16_triton | 0.0520 | int8_triton_bf16in | 1.05x | 0.0762 | n/a | 0.0811 |
| trunk.2.0.block.2.fc1 | se_fc | 8x96x4 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.88x | 0.0063 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 8x4x96 | 0.0015 | bf16_triton | 0.0013 | int8_triton_bf16in | 1.11x | 0.0040 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 32768x96x24 | 0.0088 | bf16_triton | 0.0117 | int8_triton_bf16in | 0.75x | 0.0113 | 0.0171 | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 32768x24x144 | 0.0352 | bf16_triton | 0.0362 | int8_triton_bf16in | 0.97x | 0.0282 | 0.0774 | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 8x144x6 | 0.0018 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.92x | 0.0042 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 8x6x144 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.97x | 0.0047 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 32768x144x24 | 0.0121 | bf16_triton | 0.0157 | int8_triton_bf16in | 0.77x | 0.0147 | 0.0232 | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 32768x24x144 | 0.0352 | bf16_triton | 0.0362 | int8_triton_bf16in | 0.97x | 0.0282 | 0.0774 | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 8x144x6 | 0.0018 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.92x | 0.0042 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 8x6x144 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.97x | 0.0047 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 8192x144x40 | 0.0061 | bf16_triton | 0.0065 | int8_triton_bf16in | 0.94x | 0.0081 | 0.0068 | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 8192x40x240 | 0.0150 | bf16_triton | 0.0154 | int8_triton_bf16in | 0.97x | 0.0181 | 0.0113 | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 8x240x10 | 0.0019 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.84x | 0.0042 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 8x10x240 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.98x | 0.0042 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 8192x240x40 | 0.0090 | bf16_triton | 0.0081 | int8_triton_bf16in | 1.11x | 0.0096 | 0.0094 | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 8192x40x240 | 0.0150 | bf16_triton | 0.0154 | int8_triton_bf16in | 0.97x | 0.0181 | 0.0113 | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 8x240x10 | 0.0019 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.84x | 0.0042 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 8x10x240 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.98x | 0.0042 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 2048x240x80 | 0.0053 | bf16_triton | 0.0045 | int8_triton_bf16in | 1.18x | 0.0069 | 0.0061 | 0.0045 |
| trunk.4.1.block.0.0 | pw1x1 | 2048x80x480 | 0.0090 | bf16_triton | 0.0079 | int8_triton_bf16in | 1.14x | 0.0123 | 0.0084 | 0.0075 |
| trunk.4.1.block.2.fc1 | se_fc | 8x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.78x | 0.0046 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_cublas | 0.0018 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 2048x480x80 | 0.0083 | bf16_cublas | 0.0077 | fp8_pipeline | 1.08x | 0.0082 | 0.0090 | 0.0054 |
| trunk.4.2.block.0.0 | pw1x1 | 2048x80x480 | 0.0090 | bf16_triton | 0.0079 | int8_triton_bf16in | 1.14x | 0.0123 | 0.0084 | 0.0075 |
| trunk.4.2.block.2.fc1 | se_fc | 8x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.78x | 0.0046 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_cublas | 0.0018 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 2048x480x80 | 0.0083 | bf16_cublas | 0.0077 | fp8_pipeline | 1.08x | 0.0082 | 0.0090 | 0.0054 |
| trunk.5.0.block.0.0 | pw1x1 | 2048x80x480 | 0.0090 | bf16_triton | 0.0079 | int8_triton_bf16in | 1.14x | 0.0123 | 0.0084 | 0.0075 |
| trunk.5.0.block.2.fc1 | se_fc | 8x480x20 | 0.0025 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.78x | 0.0046 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_cublas | 0.0018 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 2048x480x112 | 0.0085 | bf16_cublas | 0.0079 | int8_triton_bf16in | 1.08x | 0.0089 | 0.0093 | 0.0056 |
| trunk.5.1.block.0.0 | pw1x1 | 2048x112x672 | 0.0143 | bf16_triton | 0.0119 | int8_triton_bf16in | 1.20x | 0.0153 | 0.0221 | 0.0106 |
| trunk.5.1.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.84x | 0.0050 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 8x28x672 | 0.0016 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.03x | 0.0046 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 2048x672x112 | 0.0107 | bf16_cublas | 0.0079 | int8_triton_bf16in | 1.35x | 0.0105 | 0.0121 | 0.0067 |
| trunk.5.2.block.0.0 | pw1x1 | 2048x112x672 | 0.0143 | bf16_triton | 0.0119 | int8_triton_bf16in | 1.20x | 0.0153 | 0.0221 | 0.0106 |
| trunk.5.2.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.84x | 0.0050 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 8x28x672 | 0.0016 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.03x | 0.0046 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 2048x672x112 | 0.0107 | bf16_cublas | 0.0079 | int8_triton_bf16in | 1.35x | 0.0105 | 0.0121 | 0.0067 |
| trunk.6.0.block.0.0 | pw1x1 | 2048x112x672 | 0.0143 | bf16_triton | 0.0119 | int8_triton_bf16in | 1.20x | 0.0153 | 0.0221 | 0.0106 |
| trunk.6.0.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.84x | 0.0050 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 8x28x672 | 0.0016 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.03x | 0.0046 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 512x672x192 | 0.0065 | bf16_cublas | 0.0058 | fp8_pipeline | 1.11x | 0.0075 | 0.0075 | 0.0044 |
| trunk.6.1.block.0.0 | pw1x1 | 512x192x1152 | 0.0101 | bf16_triton | 0.0082 | int8_triton_bf16in | 1.22x | 0.0104 | 0.0139 | 0.0060 |
| trunk.6.1.block.2.fc1 | se_fc | 8x1152x48 | 0.0040 | bf16_triton | 0.0047 | int8_triton_bf16in | 0.86x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.1.block.2.fc2 | se_fc | 8x48x1152 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.85x | 0.0044 | 0.0025 | 0.0020 |
| trunk.6.1.block.3.0 | pw1x1 | 512x1152x192 | 0.0092 | bf16_cublas | 0.0075 | fp8_pipeline | 1.22x | 0.0084 | 0.0097 | 0.0057 |
| trunk.6.2.block.0.0 | pw1x1 | 512x192x1152 | 0.0101 | bf16_triton | 0.0082 | int8_triton_bf16in | 1.22x | 0.0104 | 0.0139 | 0.0060 |
| trunk.6.2.block.2.fc1 | se_fc | 8x1152x48 | 0.0040 | bf16_triton | 0.0047 | int8_triton_bf16in | 0.86x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.2.block.2.fc2 | se_fc | 8x48x1152 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.85x | 0.0044 | 0.0025 | 0.0020 |
| trunk.6.2.block.3.0 | pw1x1 | 512x1152x192 | 0.0092 | bf16_cublas | 0.0075 | fp8_pipeline | 1.22x | 0.0084 | 0.0097 | 0.0057 |
| trunk.6.3.block.0.0 | pw1x1 | 512x192x1152 | 0.0101 | bf16_triton | 0.0082 | int8_triton_bf16in | 1.22x | 0.0104 | 0.0139 | 0.0060 |
| trunk.6.3.block.2.fc1 | se_fc | 8x1152x48 | 0.0040 | bf16_triton | 0.0047 | int8_triton_bf16in | 0.86x | 0.0052 | 0.0049 | 0.0036 |
| trunk.6.3.block.2.fc2 | se_fc | 8x48x1152 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.85x | 0.0044 | 0.0025 | 0.0020 |
| trunk.6.3.block.3.0 | pw1x1 | 512x1152x192 | 0.0092 | bf16_cublas | 0.0075 | fp8_pipeline | 1.22x | 0.0084 | 0.0097 | 0.0057 |
| trunk.7.0.block.0.0 | pw1x1 | 512x192x1152 | 0.0101 | bf16_triton | 0.0082 | int8_triton_bf16in | 1.22x | 0.0104 | 0.0139 | 0.0060 |
| trunk.7.0.block.2.fc1 | se_fc | 8x1152x48 | 0.0040 | bf16_triton | 0.0047 | int8_triton_bf16in | 0.86x | 0.0052 | 0.0049 | 0.0036 |
| trunk.7.0.block.2.fc2 | se_fc | 8x48x1152 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.85x | 0.0044 | 0.0025 | 0.0020 |
| trunk.7.0.block.3.0 | pw1x1 | 512x1152x320 | 0.0139 | bf16_triton | 0.0111 | fp8_pipeline | 1.25x | 0.0108 | 0.0147 | 0.0093 |
| trunk.8.0 | pw1x1 | 512x320x1280 | 0.0125 | bf16_triton | 0.0111 | int8_triton_bf16in | 1.13x | 0.0120 | 0.0200 | 0.0089 |
| head.1 | head_fc | 8x1280x2 | 0.0042 | bf16_triton | 0.0050 | int8_triton_bf16in | 0.84x | 0.0057 | n/a | n/a |

## Batch 8: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 8x128x128x32, 3, s1 | 0.0507 | 0.0309 | 0.0463 | 0.0604 | 0.0690 | 544 | 208 | 0.51x | 0.45x |
| trunk.2.0.block.1.0 | 8x128x128x96, 3, s2 | 0.1305 | 0.0377 | 0.0434 | 0.0430 | 0.1748 | 834 | 439 | 0.88x | 0.22x |
| trunk.2.1.block.1.0 | 8x64x64x144, 3, s1 | 0.0571 | 0.0460 | 0.0525 | 0.0623 | 0.0741 | 410 | 227 | 0.74x | 0.62x |
| trunk.3.0.block.1.0 | 8x64x64x144, 5, s2 | 0.8902 | 0.0295 | 0.0432 | 0.0501 | 0.0619 | 400 | 141 | 0.59x | 0.48x |
| trunk.3.1.block.1.0 | 8x32x32x240, 5, s1 | 0.0337 | 0.0385 | 0.0399 | 0.0493 | 0.0555 | 205 | 120 | 0.68x | 0.61x |
| trunk.4.0.block.1.0 | 8x32x32x240, 3, s2 | 0.0116 | 0.0082 | 0.0082 | 0.0094 | 0.0156 | 601 | 315 | 0.88x | 0.53x |
| trunk.4.1.block.1.0 | 8x16x16x480, 3, s1 | 0.0158 | 0.0102 | 0.0135 | 0.0137 | 0.0176 | 385 | 215 | 0.75x | 0.58x |
| trunk.4.2.block.1.0 | 8x16x16x480, 3, s1 | 0.0158 | 0.0102 | 0.0135 | 0.0137 | 0.0176 | 385 | 215 | 0.75x | 0.58x |
| trunk.5.0.block.1.0 | 8x16x16x480, 5, s1 | 0.0208 | 0.0207 | 0.0231 | 0.0374 | 0.0415 | 191 | 79 | 0.55x | 0.50x |
| trunk.5.1.block.1.0 | 8x16x16x672, 5, s1 | 0.0269 | 0.0271 | 0.0320 | 0.0513 | 0.0561 | 204 | 81 | 0.52x | 0.48x |
| trunk.5.2.block.1.0 | 8x16x16x672, 5, s1 | 0.0269 | 0.0271 | 0.0320 | 0.0513 | 0.0561 | 204 | 81 | 0.52x | 0.48x |
| trunk.6.0.block.1.0 | 8x16x16x672, 5, s2 | 0.1200 | 0.0095 | 0.0106 | 0.0140 | 0.0188 | 365 | 148 | 0.68x | 0.51x |
| trunk.6.1.block.1.0 | 8x8x8x1152, 5, s1 | 0.0148 | 0.0126 | 0.0173 | 0.0230 | 0.0261 | 191 | 78 | 0.55x | 0.49x |
| trunk.6.2.block.1.0 | 8x8x8x1152, 5, s1 | 0.0148 | 0.0126 | 0.0173 | 0.0230 | 0.0261 | 191 | 78 | 0.55x | 0.49x |
| trunk.6.3.block.1.0 | 8x8x8x1152, 5, s1 | 0.0148 | 0.0126 | 0.0173 | 0.0230 | 0.0261 | 191 | 78 | 0.55x | 0.49x |
| trunk.7.0.block.1.0 | 8x8x8x1152, 3, s1 | 0.0113 | 0.0065 | 0.0097 | 0.0101 | 0.0131 | 367 | 176 | 0.64x | 0.50x |

## Batch 64: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 64x32x8 | 0.0013 | bf16_triton | 0.0029 | int8_triton_bf16in | 0.46x | 0.0041 | 0.0022 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 64x8x32 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.09x | 0.0040 | 0.0021 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 1048576x32x16 | 0.4014 | bf16_triton | 0.4102 | int8_triton_bf16in | 0.98x | 0.1881 | n/a | 0.2850 |
| trunk.2.0.block.0.0 | pw1x1 | 1048576x16x96 | 0.9240 | bf16_triton | 0.9250 | int8_triton_bf16in | 1.00x | 0.6418 | n/a | 0.8718 |
| trunk.2.0.block.2.fc1 | se_fc | 64x96x4 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.90x | 0.0043 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 64x4x96 | 0.0016 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.15x | 0.0040 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 262144x96x24 | 0.2357 | bf16_cublas | 0.2359 | int8_triton_bf16in | 1.00x | 0.0641 | n/a | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 262144x24x144 | 0.4618 | bf16_triton | 0.4601 | int8_triton_bf16in | 1.00x | 0.3674 | n/a | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 64x144x6 | 0.0017 | bf16_triton | 0.0024 | int8_triton_bf16in | 0.72x | 0.0063 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 64x6x144 | 0.0016 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.17x | 0.0042 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 262144x144x24 | 0.3326 | bf16_cublas | 0.3303 | int8_triton_bf16in | 1.01x | 0.1689 | n/a | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 262144x24x144 | 0.4618 | bf16_triton | 0.4601 | int8_triton_bf16in | 1.00x | 0.3674 | n/a | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 64x144x6 | 0.0017 | bf16_triton | 0.0024 | int8_triton_bf16in | 0.72x | 0.0063 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 64x6x144 | 0.0016 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.17x | 0.0042 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 65536x144x40 | 0.0395 | bf16_triton | 0.0360 | int8_triton_bf16in | 1.10x | 0.0368 | n/a | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 65536x40x240 | 0.1117 | bf16_triton | 0.1119 | int8_triton_bf16in | 1.00x | 0.1039 | n/a | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 64x240x10 | 0.0020 | bf16_triton | 0.0025 | int8_triton_bf16in | 0.80x | 0.0042 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 64x10x240 | 0.0016 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.11x | 0.0043 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 65536x240x40 | 0.1379 | bf16_cublas | 0.1395 | int8_triton_bf16in | 0.99x | 0.0418 | n/a | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 65536x40x240 | 0.1117 | bf16_triton | 0.1119 | int8_triton_bf16in | 1.00x | 0.1039 | n/a | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 64x240x10 | 0.0020 | bf16_triton | 0.0025 | int8_triton_bf16in | 0.80x | 0.0042 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 64x10x240 | 0.0016 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.11x | 0.0043 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 16384x240x80 | 0.0285 | bf16_triton | 0.0243 | int8_triton_bf16in | 1.18x | 0.0216 | 0.0347 | 0.0188 |
| trunk.4.1.block.0.0 | pw1x1 | 16384x80x480 | 0.0460 | bf16_triton | 0.0459 | int8_triton_bf16in | 1.00x | 0.0580 | 0.1151 | 0.0431 |
| trunk.4.1.block.2.fc1 | se_fc | 64x480x20 | 0.0040 | bf16_triton | 0.0032 | int8_triton_bf16in | 1.25x | 0.0052 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 64x20x480 | 0.0016 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.93x | 0.0043 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 16384x480x80 | 0.0477 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.80x | 0.0295 | 0.0565 | 0.0295 |
| trunk.4.2.block.0.0 | pw1x1 | 16384x80x480 | 0.0460 | bf16_triton | 0.0459 | int8_triton_bf16in | 1.00x | 0.0580 | 0.1151 | 0.0431 |
| trunk.4.2.block.2.fc1 | se_fc | 64x480x20 | 0.0040 | bf16_triton | 0.0032 | int8_triton_bf16in | 1.25x | 0.0052 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 64x20x480 | 0.0016 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.93x | 0.0043 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 16384x480x80 | 0.0477 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.80x | 0.0295 | 0.0565 | 0.0295 |
| trunk.5.0.block.0.0 | pw1x1 | 16384x80x480 | 0.0460 | bf16_triton | 0.0459 | int8_triton_bf16in | 1.00x | 0.0580 | 0.1151 | 0.0431 |
| trunk.5.0.block.2.fc1 | se_fc | 64x480x20 | 0.0040 | bf16_triton | 0.0032 | int8_triton_bf16in | 1.25x | 0.0052 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 64x20x480 | 0.0016 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.93x | 0.0043 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 16384x480x112 | 0.0502 | bf16_triton | 0.0352 | int8_triton_bf16in | 1.43x | 0.0301 | 0.0538 | 0.0285 |
| trunk.5.1.block.0.0 | pw1x1 | 16384x112x672 | 0.0838 | bf16_triton | 0.0792 | int8_triton_bf16in | 1.06x | 0.0779 | 0.1776 | 0.0679 |
| trunk.5.1.block.2.fc1 | se_fc | 64x672x28 | 0.0033 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.84x | 0.0049 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0045 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 16384x672x112 | 0.0662 | bf16_triton | 0.0453 | int8_triton_bf16in | 1.46x | 0.0377 | 0.0738 | 0.0376 |
| trunk.5.2.block.0.0 | pw1x1 | 16384x112x672 | 0.0838 | bf16_triton | 0.0792 | int8_triton_bf16in | 1.06x | 0.0779 | 0.1776 | 0.0679 |
| trunk.5.2.block.2.fc1 | se_fc | 64x672x28 | 0.0033 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.84x | 0.0049 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0045 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 16384x672x112 | 0.0662 | bf16_triton | 0.0453 | int8_triton_bf16in | 1.46x | 0.0377 | 0.0738 | 0.0376 |
| trunk.6.0.block.0.0 | pw1x1 | 16384x112x672 | 0.0838 | bf16_triton | 0.0792 | int8_triton_bf16in | 1.06x | 0.0779 | 0.1776 | 0.0679 |
| trunk.6.0.block.2.fc1 | se_fc | 64x672x28 | 0.0033 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.84x | 0.0049 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0045 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 4096x672x192 | 0.0268 | bf16_triton | 0.0200 | int8_triton_bf16in | 1.34x | 0.0184 | 0.0298 | 0.0158 |
| trunk.6.1.block.0.0 | pw1x1 | 4096x192x1152 | 0.0450 | bf16_triton | 0.0345 | int8_triton_bf16in | 1.30x | 0.0382 | 0.0540 | 0.0290 |
| trunk.6.1.block.2.fc1 | se_fc | 64x1152x48 | 0.0052 | bf16_cublas | 0.0049 | int8_triton_bf16in | 1.06x | 0.0061 | 0.0049 | 0.0036 |
| trunk.6.1.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.96x | 0.0047 | 0.0030 | 0.0023 |
| trunk.6.1.block.3.0 | pw1x1 | 4096x1152x192 | 0.0439 | bf16_triton | 0.0301 | int8_triton_bf16in | 1.46x | 0.0242 | 0.0486 | 0.0233 |
| trunk.6.2.block.0.0 | pw1x1 | 4096x192x1152 | 0.0450 | bf16_triton | 0.0345 | int8_triton_bf16in | 1.30x | 0.0382 | 0.0540 | 0.0290 |
| trunk.6.2.block.2.fc1 | se_fc | 64x1152x48 | 0.0052 | bf16_cublas | 0.0049 | int8_triton_bf16in | 1.06x | 0.0061 | 0.0049 | 0.0036 |
| trunk.6.2.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.96x | 0.0047 | 0.0030 | 0.0023 |
| trunk.6.2.block.3.0 | pw1x1 | 4096x1152x192 | 0.0439 | bf16_triton | 0.0301 | int8_triton_bf16in | 1.46x | 0.0242 | 0.0486 | 0.0233 |
| trunk.6.3.block.0.0 | pw1x1 | 4096x192x1152 | 0.0450 | bf16_triton | 0.0345 | int8_triton_bf16in | 1.30x | 0.0382 | 0.0540 | 0.0290 |
| trunk.6.3.block.2.fc1 | se_fc | 64x1152x48 | 0.0052 | bf16_cublas | 0.0049 | int8_triton_bf16in | 1.06x | 0.0061 | 0.0049 | 0.0036 |
| trunk.6.3.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.96x | 0.0047 | 0.0030 | 0.0023 |
| trunk.6.3.block.3.0 | pw1x1 | 4096x1152x192 | 0.0439 | bf16_triton | 0.0301 | int8_triton_bf16in | 1.46x | 0.0242 | 0.0486 | 0.0233 |
| trunk.7.0.block.0.0 | pw1x1 | 4096x192x1152 | 0.0450 | bf16_triton | 0.0345 | int8_triton_bf16in | 1.30x | 0.0382 | 0.0540 | 0.0290 |
| trunk.7.0.block.2.fc1 | se_fc | 64x1152x48 | 0.0052 | bf16_cublas | 0.0049 | int8_triton_bf16in | 1.06x | 0.0061 | 0.0049 | 0.0036 |
| trunk.7.0.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.96x | 0.0047 | 0.0030 | 0.0023 |
| trunk.7.0.block.3.0 | pw1x1 | 4096x1152x320 | 0.0835 | bf16_triton | 0.0445 | int8_triton_bf16in | 1.87x | 0.0397 | 0.0788 | 0.0377 |
| trunk.8.0 | pw1x1 | 4096x320x1280 | 0.0786 | bf16_triton | 0.0570 | int8_triton_bf16in | 1.38x | 0.0532 | 0.0917 | 0.0438 |
| head.1 | head_fc | 64x1280x2 | 0.0040 | bf16_cublas | 0.0060 | int8_triton_bf16in | 0.67x | 0.0055 | n/a | n/a |

## Batch 64: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 64x128x128x32, 3, s1 | 1.6040 | 0.5402 | 0.5479 | 0.4834 | 0.8792 | 248 | 208 | 1.12x | 0.61x |
| trunk.2.0.block.1.0 | 64x128x128x96, 3, s2 | 1.7837 | 0.9518 | 0.9729 | 0.5769 | 1.7684 | 264 | 262 | 1.65x | 0.54x |
| trunk.2.1.block.1.0 | 64x64x64x144, 3, s1 | 1.8092 | 0.6878 | 0.6323 | 0.6026 | 1.0475 | 220 | 188 | 1.05x | 0.60x |
| trunk.3.0.block.1.0 | 64x64x64x144, 5, s2 | 7.4287 | 0.3965 | 0.3747 | 0.3246 | 0.7705 | 238 | 174 | 1.15x | 0.49x |
| trunk.3.1.block.1.0 | 64x32x32x240, 5, s1 | 0.7258 | 0.3446 | 0.3497 | 0.4066 | 0.5708 | 183 | 116 | 0.85x | 0.60x |
| trunk.4.0.block.1.0 | 64x32x32x240, 3, s2 | 0.1968 | 0.1064 | 0.1475 | 0.0437 | 0.2307 | 369 | 540 | 2.43x | 0.46x |
| trunk.4.1.block.1.0 | 64x16x16x480, 3, s1 | 0.2035 | 0.0584 | 0.1020 | 0.0955 | 0.1753 | 539 | 247 | 0.61x | 0.33x |
| trunk.4.2.block.1.0 | 64x16x16x480, 3, s1 | 0.2035 | 0.0584 | 0.1020 | 0.0955 | 0.1753 | 539 | 247 | 0.61x | 0.33x |
| trunk.5.0.block.1.0 | 64x16x16x480, 5, s1 | 0.2082 | 0.1232 | 0.1918 | 0.1949 | 0.2747 | 256 | 121 | 0.63x | 0.45x |
| trunk.5.1.block.1.0 | 64x16x16x672, 5, s1 | 0.3771 | 0.2342 | 0.2630 | 0.2613 | 0.4015 | 188 | 126 | 0.90x | 0.58x |
| trunk.5.2.block.1.0 | 64x16x16x672, 5, s1 | 0.3771 | 0.2342 | 0.2630 | 0.2613 | 0.4015 | 188 | 126 | 0.90x | 0.58x |
| trunk.6.0.block.1.0 | 64x16x16x672, 5, s2 | 0.8415 | 0.0562 | 0.0781 | 0.0871 | 0.1910 | 490 | 190 | 0.65x | 0.29x |
| trunk.6.1.block.1.0 | 64x8x8x1152, 5, s1 | 0.0765 | 0.0771 | 0.1321 | 0.1055 | 0.1174 | 246 | 134 | 0.72x | 0.65x |
| trunk.6.2.block.1.0 | 64x8x8x1152, 5, s1 | 0.0765 | 0.0771 | 0.1321 | 0.1055 | 0.1174 | 246 | 134 | 0.72x | 0.65x |
| trunk.6.3.block.1.0 | 64x8x8x1152, 5, s1 | 0.0765 | 0.0771 | 0.1321 | 0.1055 | 0.1174 | 246 | 134 | 0.72x | 0.65x |
| trunk.7.0.block.1.0 | 64x8x8x1152, 3, s1 | 0.0577 | 0.0356 | 0.0694 | 0.0459 | 0.0572 | 530 | 309 | 0.78x | 0.62x |

