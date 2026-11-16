# Per-layer real speed (diloco)

All times are GPU time per call in milliseconds, measured by replaying a CUDA graph (no launch overhead). 'bf16 best' is the faster of cuBLAS (+ separate SiLU) and a Triton bf16 GEMM with the same fused epilogue as the int8 kernel, so the speedup is not inflated by fusion alone.
'Quant best' is the fastest complete quantized path that takes bf16 in and gives bf16 out, quantization of the input and dequantization included: Triton int8 GEMM with fused quantize/dequantize, torch._int_mm + quantize + dequantize kernels, or torch._scaled_mm FP8 + quantize kernel.

Provenance: {"idle_samples": [2, 1, 0], "gpu_before": "0 %, 510 MHz, 810 MHz, 37, 14.34 W, 1435 MiB", "gpu_after": "98 %, 2805 MHz, 8751 MHz, 58, 73.94 W, 2391 MiB", "torch": "2.14.0+cu130", "timing": "median of 15 CUDA-graph replays of 20 back-to-back calls (graph_ms) and median of 60 eager calls with CUDA events (eager_ms)"}

## Batch 1: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 1x32x8 | 0.0026 | bf16_cublas | 0.0021 | int8_triton_bf16in | 1.22x | 0.0102 | 0.0066 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 1x8x32 | 0.0121 | bf16_triton | 0.0017 | int8_triton_bf16in | 7.04x | 0.0195 | 0.0130 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 12544x32x16 | 0.0250 | bf16_triton | 0.0024 | int8_triton_bf16in | 10.25x | 0.0539 | 0.0913 | 0.0094 |
| trunk.2.0.block.0.0 | pw1x1 | 12544x16x96 | 0.0112 | bf16_cublas | 0.0080 | int8_triton_bf16in | 1.39x | 0.1294 | 0.2109 | 0.0455 |
| trunk.2.0.block.2.fc1 | se_fc | 1x96x4 | 0.0016 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.95x | 0.0040 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 1x4x96 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.97x | 0.0040 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 3136x96x24 | 0.0026 | bf16_cublas | 0.0029 | int8_triton_bf16in | 0.89x | 0.0168 | 0.0121 | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 3136x24x144 | 0.0063 | bf16_cublas | 0.0188 | int8_triton_bf16in | 0.33x | 0.0921 | 0.0787 | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 1x144x6 | 0.0017 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.81x | 0.0041 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 1x6x144 | 0.0015 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.11x | 0.0044 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 3136x144x24 | 0.0030 | bf16_triton | 0.0034 | int8_triton_bf16in | 0.88x | 0.0056 | 0.0039 | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 3136x24x144 | 0.0063 | bf16_cublas | 0.0188 | int8_triton_bf16in | 0.33x | 0.0921 | 0.0787 | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 1x144x6 | 0.0017 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.81x | 0.0041 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 1x6x144 | 0.0015 | bf16_cublas | 0.0014 | int8_triton_bf16in | 1.11x | 0.0044 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 784x144x40 | 0.0024 | bf16_triton | 0.0028 | int8_triton_bf16in | 0.85x | 0.0049 | 0.0033 | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 784x40x240 | 0.0029 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.91x | 0.0063 | 0.0046 | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 1x240x10 | 0.0019 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.86x | 0.0043 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 1x10x240 | 0.0014 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.93x | 0.0044 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 784x240x40 | 0.0032 | bf16_triton | 0.0033 | int8_triton_bf16in | 0.95x | 0.0048 | 0.0032 | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 784x40x240 | 0.0029 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.91x | 0.0063 | 0.0046 | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 1x240x10 | 0.0019 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.86x | 0.0043 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 1x10x240 | 0.0014 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.93x | 0.0044 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 196x240x80 | 0.0025 | bf16_cublas | 0.0024 | int8_triton_bf16in | 1.04x | 0.0386 | 0.0282 | 0.0254 |
| trunk.4.1.block.0.0 | pw1x1 | 196x80x480 | 0.0027 | bf16_triton | 0.0028 | int8_triton_bf16in | 0.96x | 0.0052 | 0.0034 | 0.0026 |
| trunk.4.1.block.2.fc1 | se_fc | 1x480x20 | 0.0027 | bf16_triton | 0.0031 | int8_triton_bf16in | 0.87x | 0.0045 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 1x20x480 | 0.0016 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.72x | 0.0044 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 196x480x80 | 0.0037 | bf16_triton | 0.0030 | int8_triton_bf16in | 1.22x | 0.0047 | 0.0034 | 0.0026 |
| trunk.4.2.block.0.0 | pw1x1 | 196x80x480 | 0.0027 | bf16_triton | 0.0028 | int8_triton_bf16in | 0.96x | 0.0052 | 0.0034 | 0.0026 |
| trunk.4.2.block.2.fc1 | se_fc | 1x480x20 | 0.0027 | bf16_triton | 0.0031 | int8_triton_bf16in | 0.87x | 0.0045 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 1x20x480 | 0.0016 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.72x | 0.0044 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 196x480x80 | 0.0037 | bf16_triton | 0.0030 | int8_triton_bf16in | 1.22x | 0.0047 | 0.0034 | 0.0026 |
| trunk.5.0.block.0.0 | pw1x1 | 196x80x480 | 0.0027 | bf16_triton | 0.0028 | int8_triton_bf16in | 0.96x | 0.0052 | 0.0034 | 0.0026 |
| trunk.5.0.block.2.fc1 | se_fc | 1x480x20 | 0.0027 | bf16_triton | 0.0031 | int8_triton_bf16in | 0.87x | 0.0045 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 1x20x480 | 0.0016 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.72x | 0.0044 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 196x480x112 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.84x | 0.0049 | 0.0034 | 0.0027 |
| trunk.5.1.block.0.0 | pw1x1 | 196x112x672 | 0.0034 | bf16_triton | 0.0038 | int8_triton_bf16in | 0.88x | 0.0062 | 0.0045 | 0.0032 |
| trunk.5.1.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0041 | int8_triton_bf16in | 0.69x | 0.0050 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.11x | 0.0045 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 196x672x112 | 0.0046 | bf16_triton | 0.0045 | int8_triton_bf16in | 1.02x | 0.0052 | 0.0041 | 0.0030 |
| trunk.5.2.block.0.0 | pw1x1 | 196x112x672 | 0.0034 | bf16_triton | 0.0038 | int8_triton_bf16in | 0.88x | 0.0062 | 0.0045 | 0.0032 |
| trunk.5.2.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0041 | int8_triton_bf16in | 0.69x | 0.0050 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.11x | 0.0045 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 196x672x112 | 0.0046 | bf16_triton | 0.0045 | int8_triton_bf16in | 1.02x | 0.0052 | 0.0041 | 0.0030 |
| trunk.6.0.block.0.0 | pw1x1 | 196x112x672 | 0.0034 | bf16_triton | 0.0038 | int8_triton_bf16in | 0.88x | 0.0062 | 0.0045 | 0.0032 |
| trunk.6.0.block.2.fc1 | se_fc | 1x672x28 | 0.0028 | bf16_cublas | 0.0041 | int8_triton_bf16in | 0.69x | 0.0050 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 1x28x672 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.11x | 0.0045 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 49x672x192 | 0.0033 | bf16_triton | 0.0048 | int8_triton_bf16in | 0.68x | 0.0057 | 0.0039 | 0.0029 |
| trunk.6.1.block.0.0 | pw1x1 | 49x192x1152 | 0.0031 | bf16_triton | 0.0034 | int8_triton_bf16in | 0.92x | 0.0052 | 0.0040 | 0.0027 |
| trunk.6.1.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0049 | int8_triton_bf16in | 0.57x | 0.0249 | 0.0286 | 0.0201 |
| trunk.6.1.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.1.block.3.0 | pw1x1 | 49x1152x192 | 0.0051 | bf16_cublas | 0.0063 | int8_triton_bf16in | 0.81x | 0.0066 | 0.0049 | 0.0036 |
| trunk.6.2.block.0.0 | pw1x1 | 49x192x1152 | 0.0031 | bf16_triton | 0.0034 | int8_triton_bf16in | 0.92x | 0.0052 | 0.0040 | 0.0027 |
| trunk.6.2.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0049 | int8_triton_bf16in | 0.57x | 0.0249 | 0.0286 | 0.0201 |
| trunk.6.2.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.2.block.3.0 | pw1x1 | 49x1152x192 | 0.0051 | bf16_cublas | 0.0063 | int8_triton_bf16in | 0.81x | 0.0066 | 0.0049 | 0.0036 |
| trunk.6.3.block.0.0 | pw1x1 | 49x192x1152 | 0.0031 | bf16_triton | 0.0034 | int8_triton_bf16in | 0.92x | 0.0052 | 0.0040 | 0.0027 |
| trunk.6.3.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0049 | int8_triton_bf16in | 0.57x | 0.0249 | 0.0286 | 0.0201 |
| trunk.6.3.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.6.3.block.3.0 | pw1x1 | 49x1152x192 | 0.0051 | bf16_cublas | 0.0063 | int8_triton_bf16in | 0.81x | 0.0066 | 0.0049 | 0.0036 |
| trunk.7.0.block.0.0 | pw1x1 | 49x192x1152 | 0.0031 | bf16_triton | 0.0034 | int8_triton_bf16in | 0.92x | 0.0052 | 0.0040 | 0.0027 |
| trunk.7.0.block.2.fc1 | se_fc | 1x1152x48 | 0.0028 | bf16_cublas | 0.0049 | int8_triton_bf16in | 0.57x | 0.0249 | 0.0286 | 0.0201 |
| trunk.7.0.block.2.fc2 | se_fc | 1x48x1152 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.03x | 0.0040 | 0.0025 | 0.0020 |
| trunk.7.0.block.3.0 | pw1x1 | 49x1152x320 | 0.0044 | bf16_triton | 0.0063 | int8_triton_bf16in | 0.69x | 0.0067 | 0.0052 | 0.0037 |
| trunk.8.0 | pw1x1 | 49x320x1280 | 0.0046 | bf16_triton | 0.0050 | int8_triton_bf16in | 0.92x | 0.0063 | 0.0049 | 0.0031 |
| head.proj | head_proj | 1x1280x512 | 0.0038 | bf16_cublas | 0.0052 | int8_triton_bf16in | 0.73x | 0.0056 | 0.0052 | 0.0038 |
| head.layers.0.mixer.in_proj | mamba_in_proj | 1x512x2048 | 0.0032 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.81x | 0.0049 | 0.0044 | 0.0028 |
| head.layers.0.mixer.x_proj | mamba_x_proj | 1x1024x32 | 0.0017 | bf16_cublas | 0.0042 | int8_triton_bf16in | 0.40x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.0.mixer.dt_proj | mamba_dt_proj | 1x1024x1024 | 0.0038 | bf16_triton | 0.0061 | fp8_pipeline | 0.63x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.0.mixer.out_proj | mamba_out_proj | 1x1024x512 | 0.0031 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.70x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.1.mixer.in_proj | mamba_in_proj | 1x512x2048 | 0.0032 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.81x | 0.0049 | 0.0044 | 0.0028 |
| head.layers.1.mixer.x_proj | mamba_x_proj | 1x1024x32 | 0.0017 | bf16_cublas | 0.0042 | int8_triton_bf16in | 0.40x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.1.mixer.dt_proj | mamba_dt_proj | 1x1024x1024 | 0.0038 | bf16_triton | 0.0061 | fp8_pipeline | 0.63x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.1.mixer.out_proj | mamba_out_proj | 1x1024x512 | 0.0031 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.70x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.2.mixer.in_proj | mamba_in_proj | 1x512x2048 | 0.0032 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.81x | 0.0049 | 0.0044 | 0.0028 |
| head.layers.2.mixer.x_proj | mamba_x_proj | 1x1024x32 | 0.0017 | bf16_cublas | 0.0042 | int8_triton_bf16in | 0.40x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.2.mixer.dt_proj | mamba_dt_proj | 1x1024x1024 | 0.0038 | bf16_triton | 0.0061 | fp8_pipeline | 0.63x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.2.mixer.out_proj | mamba_out_proj | 1x1024x512 | 0.0031 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.70x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.3.mixer.in_proj | mamba_in_proj | 1x512x2048 | 0.0032 | bf16_triton | 0.0039 | int8_triton_bf16in | 0.81x | 0.0049 | 0.0044 | 0.0028 |
| head.layers.3.mixer.x_proj | mamba_x_proj | 1x1024x32 | 0.0017 | bf16_cublas | 0.0042 | int8_triton_bf16in | 0.40x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.3.mixer.dt_proj | mamba_dt_proj | 1x1024x1024 | 0.0038 | bf16_triton | 0.0061 | fp8_pipeline | 0.63x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.3.mixer.out_proj | mamba_out_proj | 1x1024x512 | 0.0031 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.70x | 0.0053 | 0.0046 | 0.0034 |
| head.head.0 | head_mlp | 1x512x256 | 0.0024 | bf16_cublas | 0.0029 | int8_triton_bf16in | 0.81x | 0.0045 | 0.0032 | 0.0026 |
| head.head.3 | head_mlp | 1x256x2 | 0.0017 | bf16_cublas | 0.0021 | int8_triton_bf16in | 0.79x | 0.0041 | n/a | n/a |

## Batch 1: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 1x112x112x32, 3, s1 | 0.0106 | 0.0066 | 0.0061 | 0.0408 | 0.0502 | 242 | 30 | 0.15x | 0.12x |
| trunk.2.0.block.1.0 | 1x112x112x96, 3, s2 | 0.0090 | 0.0553 | 0.0556 | 0.0191 | 0.0219 | 54 | 95 | 0.47x | 0.41x |
| trunk.2.1.block.1.0 | 1x56x56x144, 3, s1 | 0.0104 | 0.0393 | 0.0468 | 0.1111 | 0.1421 | 46 | 12 | 0.09x | 0.07x |
| trunk.3.0.block.1.0 | 1x56x56x144, 5, s2 | 0.0667 | 0.0044 | 0.0052 | 0.0067 | 0.0093 | 256 | 102 | 0.67x | 0.48x |
| trunk.3.1.block.1.0 | 1x28x28x240, 5, s1 | 0.0104 | 0.0064 | 0.0076 | 0.0090 | 0.0110 | 119 | 63 | 0.71x | 0.58x |
| trunk.4.0.block.1.0 | 1x28x28x240, 3, s2 | 0.0054 | 0.0143 | 0.0134 | 0.0232 | 0.0418 | 33 | 12 | 0.23x | 0.13x |
| trunk.4.1.block.1.0 | 1x14x14x480, 3, s1 | 0.0051 | 0.0155 | 0.0172 | 0.0222 | 0.0336 | 25 | 13 | 0.23x | 0.15x |
| trunk.4.2.block.1.0 | 1x14x14x480, 3, s1 | 0.0051 | 0.0155 | 0.0172 | 0.0222 | 0.0336 | 25 | 13 | 0.23x | 0.15x |
| trunk.5.0.block.1.0 | 1x14x14x480, 5, s1 | 0.0081 | 0.0093 | 0.0105 | 0.0274 | 0.0337 | 43 | 11 | 0.30x | 0.24x |
| trunk.5.1.block.1.0 | 1x14x14x672, 5, s1 | 0.0090 | 0.0052 | 0.0059 | 0.0065 | 0.0079 | 109 | 64 | 0.80x | 0.65x |
| trunk.5.2.block.1.0 | 1x14x14x672, 5, s1 | 0.0090 | 0.0052 | 0.0059 | 0.0065 | 0.0079 | 109 | 64 | 0.80x | 0.65x |
| trunk.6.0.block.1.0 | 1x14x14x672, 5, s2 | 0.0388 | 0.0034 | 0.0027 | 0.0191 | 0.0239 | 106 | 11 | 0.14x | 0.11x |
| trunk.6.1.block.1.0 | 1x7x7x1152, 5, s1 | 0.0076 | 0.0039 | 0.0033 | 0.0039 | 0.0053 | 72 | 50 | 0.83x | 0.62x |
| trunk.6.2.block.1.0 | 1x7x7x1152, 5, s1 | 0.0076 | 0.0039 | 0.0033 | 0.0039 | 0.0053 | 72 | 50 | 0.83x | 0.62x |
| trunk.6.3.block.1.0 | 1x7x7x1152, 5, s1 | 0.0076 | 0.0039 | 0.0033 | 0.0039 | 0.0053 | 72 | 50 | 0.83x | 0.62x |
| trunk.7.0.block.1.0 | 1x7x7x1152, 3, s1 | 0.0048 | 0.0025 | 0.0022 | 0.0022 | 0.0035 | 100 | 84 | 1.01x | 0.62x |

## Batch 8: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 8x32x8 | 0.0055 | bf16_triton | 0.0101 | int8_triton_bf16in | 0.55x | 0.0312 | 0.0226 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 8x8x32 | 0.0101 | bf16_cublas | 0.0113 | int8_triton_bf16in | 0.90x | 0.0323 | 0.0220 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 100352x32x16 | 0.0106 | bf16_triton | 0.0141 | int8_triton_bf16in | 0.75x | 0.0185 | n/a | 0.0241 |
| trunk.2.0.block.0.0 | pw1x1 | 100352x16x96 | 0.0591 | bf16_triton | 0.0399 | int8_triton_bf16in | 1.48x | 0.0597 | n/a | 0.0484 |
| trunk.2.0.block.2.fc1 | se_fc | 8x96x4 | 0.0016 | bf16_triton | 0.0036 | int8_triton_bf16in | 0.44x | 0.0040 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 8x4x96 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.99x | 0.0041 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 25088x96x24 | 0.0070 | bf16_triton | 0.0080 | int8_triton_bf16in | 0.88x | 0.0094 | 0.0135 | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 25088x24x144 | 0.0274 | bf16_triton | 0.0280 | int8_triton_bf16in | 0.98x | 0.0223 | 0.0592 | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 8x144x6 | 0.0019 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.97x | 0.0041 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 8x6x144 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.07x | 0.0042 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 25088x144x24 | 0.0114 | bf16_triton | 0.0112 | int8_triton_bf16in | 1.01x | 0.0125 | 0.0182 | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 25088x24x144 | 0.0274 | bf16_triton | 0.0280 | int8_triton_bf16in | 0.98x | 0.0223 | 0.0592 | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 8x144x6 | 0.0019 | bf16_triton | 0.0019 | int8_triton_bf16in | 0.97x | 0.0041 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 8x6x144 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.07x | 0.0042 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 6272x144x40 | 0.0052 | bf16_triton | 0.0079 | int8_triton_bf16in | 0.65x | 0.0079 | 0.0102 | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 6272x40x240 | 0.0118 | bf16_triton | 0.0121 | int8_triton_bf16in | 0.97x | 0.0141 | 0.0091 | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 8x240x10 | 0.0019 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.86x | 0.0055 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 8x10x240 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.11x | 0.0045 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 6272x240x40 | 0.0073 | bf16_triton | 0.0078 | int8_triton_bf16in | 0.94x | 0.0087 | 0.0102 | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 6272x40x240 | 0.0118 | bf16_triton | 0.0121 | int8_triton_bf16in | 0.97x | 0.0141 | 0.0091 | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 8x240x10 | 0.0019 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.86x | 0.0055 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 8x10x240 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.11x | 0.0045 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 1568x240x80 | 0.0051 | bf16_triton | 0.0051 | int8_triton_bf16in | 1.01x | 0.0063 | 0.0055 | 0.0048 |
| trunk.4.1.block.0.0 | pw1x1 | 1568x80x480 | 0.0086 | bf16_triton | 0.0067 | int8_triton_bf16in | 1.27x | 0.0102 | 0.0068 | 0.0067 |
| trunk.4.1.block.2.fc1 | se_fc | 8x480x20 | 0.0027 | bf16_triton | 0.0030 | int8_triton_bf16in | 0.91x | 0.0046 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.08x | 0.0044 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 1568x480x80 | 0.0072 | bf16_triton | 0.0072 | fp8_pipeline | 1.01x | 0.0073 | 0.0078 | 0.0051 |
| trunk.4.2.block.0.0 | pw1x1 | 1568x80x480 | 0.0086 | bf16_triton | 0.0067 | int8_triton_bf16in | 1.27x | 0.0102 | 0.0068 | 0.0067 |
| trunk.4.2.block.2.fc1 | se_fc | 8x480x20 | 0.0027 | bf16_triton | 0.0030 | int8_triton_bf16in | 0.91x | 0.0046 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.08x | 0.0044 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 1568x480x80 | 0.0072 | bf16_triton | 0.0072 | fp8_pipeline | 1.01x | 0.0073 | 0.0078 | 0.0051 |
| trunk.5.0.block.0.0 | pw1x1 | 1568x80x480 | 0.0086 | bf16_triton | 0.0067 | int8_triton_bf16in | 1.27x | 0.0102 | 0.0068 | 0.0067 |
| trunk.5.0.block.2.fc1 | se_fc | 8x480x20 | 0.0027 | bf16_triton | 0.0030 | int8_triton_bf16in | 0.91x | 0.0046 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 8x20x480 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.08x | 0.0044 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 1568x480x112 | 0.0078 | bf16_triton | 0.0055 | int8_triton_bf16in | 1.42x | 0.0075 | 0.0090 | 0.0054 |
| trunk.5.1.block.0.0 | pw1x1 | 1568x112x672 | 0.0113 | bf16_triton | 0.0118 | int8_triton_bf16in | 0.96x | 0.0135 | 0.0173 | 0.0090 |
| trunk.5.1.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.83x | 0.0050 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 8x28x672 | 0.0015 | bf16_cublas | 0.0016 | int8_triton_bf16in | 0.97x | 0.0046 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 1568x672x112 | 0.0105 | bf16_cublas | 0.0079 | int8_triton_bf16in | 1.34x | 0.0090 | 0.0120 | 0.0065 |
| trunk.5.2.block.0.0 | pw1x1 | 1568x112x672 | 0.0113 | bf16_triton | 0.0118 | int8_triton_bf16in | 0.96x | 0.0135 | 0.0173 | 0.0090 |
| trunk.5.2.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.83x | 0.0050 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 8x28x672 | 0.0015 | bf16_cublas | 0.0016 | int8_triton_bf16in | 0.97x | 0.0046 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 1568x672x112 | 0.0105 | bf16_cublas | 0.0079 | int8_triton_bf16in | 1.34x | 0.0090 | 0.0120 | 0.0065 |
| trunk.6.0.block.0.0 | pw1x1 | 1568x112x672 | 0.0113 | bf16_triton | 0.0118 | int8_triton_bf16in | 0.96x | 0.0135 | 0.0173 | 0.0090 |
| trunk.6.0.block.2.fc1 | se_fc | 8x672x28 | 0.0031 | bf16_triton | 0.0037 | int8_triton_bf16in | 0.83x | 0.0050 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 8x28x672 | 0.0015 | bf16_cublas | 0.0016 | int8_triton_bf16in | 0.97x | 0.0046 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 392x672x192 | 0.0065 | bf16_cublas | 0.0064 | int8_triton_bf16in | 1.02x | 0.0071 | 0.0073 | 0.0043 |
| trunk.6.1.block.0.0 | pw1x1 | 392x192x1152 | 0.0077 | bf16_triton | 0.0056 | int8_triton_bf16in | 1.39x | 0.0081 | 0.0079 | 0.0051 |
| trunk.6.1.block.2.fc1 | se_fc | 8x1152x48 | 0.0072 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.55x | 0.0053 | 0.0049 | 0.0036 |
| trunk.6.1.block.2.fc2 | se_fc | 8x48x1152 | 0.0019 | bf16_cublas | 0.0016 | int8_triton_bf16in | 1.17x | 0.0042 | 0.0025 | 0.0020 |
| trunk.6.1.block.3.0 | pw1x1 | 392x1152x192 | 0.0091 | bf16_cublas | 0.0072 | int8_triton_bf16in | 1.27x | 0.0081 | 0.0096 | 0.0056 |
| trunk.6.2.block.0.0 | pw1x1 | 392x192x1152 | 0.0077 | bf16_triton | 0.0056 | int8_triton_bf16in | 1.39x | 0.0081 | 0.0079 | 0.0051 |
| trunk.6.2.block.2.fc1 | se_fc | 8x1152x48 | 0.0072 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.55x | 0.0053 | 0.0049 | 0.0036 |
| trunk.6.2.block.2.fc2 | se_fc | 8x48x1152 | 0.0019 | bf16_cublas | 0.0016 | int8_triton_bf16in | 1.17x | 0.0042 | 0.0025 | 0.0020 |
| trunk.6.2.block.3.0 | pw1x1 | 392x1152x192 | 0.0091 | bf16_cublas | 0.0072 | int8_triton_bf16in | 1.27x | 0.0081 | 0.0096 | 0.0056 |
| trunk.6.3.block.0.0 | pw1x1 | 392x192x1152 | 0.0077 | bf16_triton | 0.0056 | int8_triton_bf16in | 1.39x | 0.0081 | 0.0079 | 0.0051 |
| trunk.6.3.block.2.fc1 | se_fc | 8x1152x48 | 0.0072 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.55x | 0.0053 | 0.0049 | 0.0036 |
| trunk.6.3.block.2.fc2 | se_fc | 8x48x1152 | 0.0019 | bf16_cublas | 0.0016 | int8_triton_bf16in | 1.17x | 0.0042 | 0.0025 | 0.0020 |
| trunk.6.3.block.3.0 | pw1x1 | 392x1152x192 | 0.0091 | bf16_cublas | 0.0072 | int8_triton_bf16in | 1.27x | 0.0081 | 0.0096 | 0.0056 |
| trunk.7.0.block.0.0 | pw1x1 | 392x192x1152 | 0.0077 | bf16_triton | 0.0056 | int8_triton_bf16in | 1.39x | 0.0081 | 0.0079 | 0.0051 |
| trunk.7.0.block.2.fc1 | se_fc | 8x1152x48 | 0.0072 | bf16_triton | 0.0047 | int8_triton_bf16in | 1.55x | 0.0053 | 0.0049 | 0.0036 |
| trunk.7.0.block.2.fc2 | se_fc | 8x48x1152 | 0.0019 | bf16_cublas | 0.0016 | int8_triton_bf16in | 1.17x | 0.0042 | 0.0025 | 0.0020 |
| trunk.7.0.block.3.0 | pw1x1 | 392x1152x320 | 0.0112 | bf16_triton | 0.0082 | int8_triton_bf16in | 1.37x | 0.0101 | 0.0122 | 0.0065 |
| trunk.8.0 | pw1x1 | 392x320x1280 | 0.0113 | bf16_triton | 0.0104 | int8_triton_bf16in | 1.08x | 0.0121 | 0.0168 | 0.0076 |
| head.proj | head_proj | 8x1280x512 | 0.0044 | bf16_triton | 0.0051 | int8_triton_bf16in | 0.85x | 0.0056 | 0.0052 | 0.0038 |
| head.layers.0.mixer.in_proj | mamba_in_proj | 8x512x2048 | 0.0038 | bf16_cublas | 0.0035 | int8_triton_bf16in | 1.09x | 0.0052 | 0.0044 | 0.0029 |
| head.layers.0.mixer.x_proj | mamba_x_proj | 8x1024x32 | 0.0036 | bf16_triton | 0.0042 | int8_triton_bf16in | 0.86x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.0.mixer.dt_proj | mamba_dt_proj | 8x1024x1024 | 0.0038 | bf16_triton | 0.0044 | int8_triton_bf16in | 0.87x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.0.mixer.out_proj | mamba_out_proj | 8x1024x512 | 0.0038 | bf16_triton | 0.0058 | int8_triton_bf16in | 0.65x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.1.mixer.in_proj | mamba_in_proj | 8x512x2048 | 0.0038 | bf16_cublas | 0.0035 | int8_triton_bf16in | 1.09x | 0.0052 | 0.0044 | 0.0029 |
| head.layers.1.mixer.x_proj | mamba_x_proj | 8x1024x32 | 0.0036 | bf16_triton | 0.0042 | int8_triton_bf16in | 0.86x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.1.mixer.dt_proj | mamba_dt_proj | 8x1024x1024 | 0.0038 | bf16_triton | 0.0044 | int8_triton_bf16in | 0.87x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.1.mixer.out_proj | mamba_out_proj | 8x1024x512 | 0.0038 | bf16_triton | 0.0058 | int8_triton_bf16in | 0.65x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.2.mixer.in_proj | mamba_in_proj | 8x512x2048 | 0.0038 | bf16_cublas | 0.0035 | int8_triton_bf16in | 1.09x | 0.0052 | 0.0044 | 0.0029 |
| head.layers.2.mixer.x_proj | mamba_x_proj | 8x1024x32 | 0.0036 | bf16_triton | 0.0042 | int8_triton_bf16in | 0.86x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.2.mixer.dt_proj | mamba_dt_proj | 8x1024x1024 | 0.0038 | bf16_triton | 0.0044 | int8_triton_bf16in | 0.87x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.2.mixer.out_proj | mamba_out_proj | 8x1024x512 | 0.0038 | bf16_triton | 0.0058 | int8_triton_bf16in | 0.65x | 0.0053 | 0.0046 | 0.0034 |
| head.layers.3.mixer.in_proj | mamba_in_proj | 8x512x2048 | 0.0038 | bf16_cublas | 0.0035 | int8_triton_bf16in | 1.09x | 0.0052 | 0.0044 | 0.0029 |
| head.layers.3.mixer.x_proj | mamba_x_proj | 8x1024x32 | 0.0036 | bf16_triton | 0.0042 | int8_triton_bf16in | 0.86x | 0.0050 | 0.0045 | 0.0034 |
| head.layers.3.mixer.dt_proj | mamba_dt_proj | 8x1024x1024 | 0.0038 | bf16_triton | 0.0044 | int8_triton_bf16in | 0.87x | 0.0052 | 0.0048 | 0.0035 |
| head.layers.3.mixer.out_proj | mamba_out_proj | 8x1024x512 | 0.0038 | bf16_triton | 0.0058 | int8_triton_bf16in | 0.65x | 0.0053 | 0.0046 | 0.0034 |
| head.head.0 | head_mlp | 8x512x256 | 0.0033 | bf16_cublas | 0.0030 | int8_triton_bf16in | 1.10x | 0.0045 | 0.0032 | 0.0026 |
| head.head.3 | head_mlp | 8x256x2 | 0.0024 | bf16_cublas | 0.0105 | int8_triton_bf16in | 0.23x | 0.0347 | n/a | n/a |

## Batch 8: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 8x112x112x32, 3, s1 | 0.1845 | 0.0239 | 0.0364 | 0.0467 | 0.0537 | 537 | 206 | 0.51x | 0.45x |
| trunk.2.0.block.1.0 | 8x112x112x96, 3, s2 | 0.0509 | 0.0291 | 0.0371 | 0.0331 | 0.0577 | 827 | 436 | 0.88x | 0.51x |
| trunk.2.1.block.1.0 | 8x56x56x144, 3, s1 | 0.0525 | 0.0382 | 0.0541 | 0.0479 | 0.0575 | 378 | 226 | 0.80x | 0.66x |
| trunk.3.0.block.1.0 | 8x56x56x144, 5, s2 | 0.6259 | 0.0232 | 0.0313 | 0.0377 | 0.0473 | 389 | 144 | 0.62x | 0.49x |
| trunk.3.1.block.1.0 | 8x28x28x240, 5, s1 | 0.0288 | 0.0304 | 0.0495 | 0.0452 | 0.0504 | 198 | 100 | 0.64x | 0.57x |
| trunk.4.0.block.1.0 | 8x28x28x240, 3, s2 | 0.0698 | 0.0502 | 0.0718 | 0.0848 | 0.0149 | 75 | 27 | 0.59x | 3.37x |
| trunk.4.1.block.1.0 | 8x14x14x480, 3, s1 | 0.0723 | 0.0083 | 0.0148 | 0.0112 | 0.0145 | 362 | 202 | 0.74x | 0.58x |
| trunk.4.2.block.1.0 | 8x14x14x480, 3, s1 | 0.0723 | 0.0083 | 0.0148 | 0.0112 | 0.0145 | 362 | 202 | 0.74x | 0.58x |
| trunk.5.0.block.1.0 | 8x14x14x480, 5, s1 | 0.0691 | 0.0165 | 0.0276 | 0.0293 | 0.0326 | 184 | 77 | 0.56x | 0.51x |
| trunk.5.1.block.1.0 | 8x14x14x672, 5, s1 | 0.0220 | 0.0219 | 0.0383 | 0.0402 | 0.0442 | 194 | 79 | 0.54x | 0.49x |
| trunk.5.2.block.1.0 | 8x14x14x672, 5, s1 | 0.0220 | 0.0219 | 0.0383 | 0.0402 | 0.0442 | 194 | 79 | 0.54x | 0.49x |
| trunk.6.0.block.1.0 | 8x14x14x672, 5, s2 | 0.2125 | 0.0079 | 0.0106 | 0.0115 | 0.0155 | 339 | 139 | 0.69x | 0.51x |
| trunk.6.1.block.1.0 | 8x7x7x1152, 5, s1 | 0.0132 | 0.0109 | 0.0173 | 0.0186 | 0.0213 | 171 | 74 | 0.59x | 0.51x |
| trunk.6.2.block.1.0 | 8x7x7x1152, 5, s1 | 0.0132 | 0.0109 | 0.0173 | 0.0186 | 0.0213 | 171 | 74 | 0.59x | 0.51x |
| trunk.6.3.block.1.0 | 8x7x7x1152, 5, s1 | 0.0132 | 0.0109 | 0.0173 | 0.0186 | 0.0213 | 171 | 74 | 0.59x | 0.51x |
| trunk.7.0.block.1.0 | 8x7x7x1152, 3, s1 | 0.0100 | 0.0059 | 0.0097 | 0.0082 | 0.0109 | 310 | 167 | 0.72x | 0.54x |

## Batch 64: GEMM-type layers (1x1 convolutions, SE, Linear)

| Layer | Kind | M x K x N | bf16 best ms | (impl) | Quant best ms | (impl) | Speedup | int8 in/int8 out ms | _int_mm only ms | _scaled_mm only ms |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.1.fc1 | se_fc | 64x32x8 | 0.0107 | bf16_triton | 0.0016 | int8_triton_bf16in | 6.74x | 0.0314 | 0.0226 | n/a |
| trunk.1.0.block.1.fc2 | se_fc | 64x8x32 | 0.0117 | bf16_triton | 0.0014 | int8_triton_bf16in | 8.25x | 0.0328 | 0.0220 | n/a |
| trunk.1.0.block.2.0 | pw1x1 | 802816x32x16 | 0.3266 | bf16_cublas | 0.3239 | int8_triton_bf16in | 1.01x | 0.1448 | n/a | 0.3905 |
| trunk.2.0.block.0.0 | pw1x1 | 802816x16x96 | 0.7075 | bf16_triton | 0.7151 | int8_triton_bf16in | 0.99x | 0.5131 | n/a | 0.6968 |
| trunk.2.0.block.2.fc1 | se_fc | 64x96x4 | 0.0015 | bf16_triton | 0.0017 | int8_triton_bf16in | 0.88x | 0.0043 | n/a | n/a |
| trunk.2.0.block.2.fc2 | se_fc | 64x4x96 | 0.0015 | bf16_triton | 0.0014 | int8_triton_bf16in | 1.12x | 0.0040 | n/a | n/a |
| trunk.2.0.block.3.0 | pw1x1 | 200704x96x24 | 0.1809 | bf16_cublas | 0.1820 | int8_triton_bf16in | 0.99x | 0.0499 | n/a | n/a |
| trunk.2.1.block.0.0 | pw1x1 | 200704x24x144 | 0.3388 | bf16_triton | 0.3370 | int8_triton_bf16in | 1.01x | 0.2523 | n/a | n/a |
| trunk.2.1.block.2.fc1 | se_fc | 64x144x6 | 0.0017 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.75x | 0.0042 | n/a | n/a |
| trunk.2.1.block.2.fc2 | se_fc | 64x6x144 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.02x | 0.0067 | n/a | n/a |
| trunk.2.1.block.3.0 | pw1x1 | 200704x144x24 | 0.2535 | bf16_cublas | 0.2582 | int8_triton_bf16in | 0.98x | 0.1262 | n/a | n/a |
| trunk.3.0.block.0.0 | pw1x1 | 200704x24x144 | 0.3388 | bf16_triton | 0.3370 | int8_triton_bf16in | 1.01x | 0.2523 | n/a | n/a |
| trunk.3.0.block.2.fc1 | se_fc | 64x144x6 | 0.0017 | bf16_triton | 0.0023 | int8_triton_bf16in | 0.75x | 0.0042 | n/a | n/a |
| trunk.3.0.block.2.fc2 | se_fc | 64x6x144 | 0.0016 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.02x | 0.0067 | n/a | n/a |
| trunk.3.0.block.3.0 | pw1x1 | 50176x144x40 | 0.0311 | bf16_triton | 0.0286 | int8_triton_bf16in | 1.09x | 0.0295 | 0.0306 | n/a |
| trunk.3.1.block.0.0 | pw1x1 | 50176x40x240 | 0.0750 | bf16_triton | 0.0746 | int8_triton_bf16in | 1.00x | 0.0800 | 0.1992 | n/a |
| trunk.3.1.block.2.fc1 | se_fc | 64x240x10 | 0.0020 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.93x | 0.0042 | n/a | n/a |
| trunk.3.1.block.2.fc2 | se_fc | 64x10x240 | 0.0016 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.03x | 0.0042 | n/a | n/a |
| trunk.3.1.block.3.0 | pw1x1 | 50176x240x40 | 0.0641 | bf16_cublas | 0.0658 | int8_triton_bf16in | 0.97x | 0.0322 | 0.0450 | n/a |
| trunk.4.0.block.0.0 | pw1x1 | 50176x40x240 | 0.0750 | bf16_triton | 0.0746 | int8_triton_bf16in | 1.00x | 0.0800 | 0.1992 | n/a |
| trunk.4.0.block.2.fc1 | se_fc | 64x240x10 | 0.0020 | bf16_triton | 0.0022 | int8_triton_bf16in | 0.93x | 0.0042 | n/a | n/a |
| trunk.4.0.block.2.fc2 | se_fc | 64x10x240 | 0.0016 | bf16_cublas | 0.0015 | int8_triton_bf16in | 1.03x | 0.0042 | n/a | n/a |
| trunk.4.0.block.3.0 | pw1x1 | 12544x240x80 | 0.0207 | bf16_triton | 0.0148 | int8_triton_bf16in | 1.40x | 0.0170 | 0.0270 | 0.0165 |
| trunk.4.1.block.0.0 | pw1x1 | 12544x80x480 | 0.0355 | bf16_triton | 0.0367 | int8_triton_bf16in | 0.97x | 0.0461 | 0.0419 | 0.0332 |
| trunk.4.1.block.2.fc1 | se_fc | 64x480x20 | 0.0027 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.4.1.block.2.fc2 | se_fc | 64x20x480 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.96x | 0.0043 | n/a | n/a |
| trunk.4.1.block.3.0 | pw1x1 | 12544x480x80 | 0.0360 | bf16_triton | 0.0214 | int8_triton_bf16in | 1.68x | 0.0232 | 0.0434 | 0.0211 |
| trunk.4.2.block.0.0 | pw1x1 | 12544x80x480 | 0.0355 | bf16_triton | 0.0367 | int8_triton_bf16in | 0.97x | 0.0461 | 0.0419 | 0.0332 |
| trunk.4.2.block.2.fc1 | se_fc | 64x480x20 | 0.0027 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.4.2.block.2.fc2 | se_fc | 64x20x480 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.96x | 0.0043 | n/a | n/a |
| trunk.4.2.block.3.0 | pw1x1 | 12544x480x80 | 0.0360 | bf16_triton | 0.0214 | int8_triton_bf16in | 1.68x | 0.0232 | 0.0434 | 0.0211 |
| trunk.5.0.block.0.0 | pw1x1 | 12544x80x480 | 0.0355 | bf16_triton | 0.0367 | int8_triton_bf16in | 0.97x | 0.0461 | 0.0419 | 0.0332 |
| trunk.5.0.block.2.fc1 | se_fc | 64x480x20 | 0.0027 | bf16_triton | 0.0032 | int8_triton_bf16in | 0.84x | 0.0045 | n/a | n/a |
| trunk.5.0.block.2.fc2 | se_fc | 64x20x480 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 0.96x | 0.0043 | n/a | n/a |
| trunk.5.0.block.3.0 | pw1x1 | 12544x480x112 | 0.0366 | bf16_triton | 0.0232 | int8_triton_bf16in | 1.57x | 0.0279 | 0.0409 | 0.0220 |
| trunk.5.1.block.0.0 | pw1x1 | 12544x112x672 | 0.0664 | bf16_triton | 0.0600 | int8_triton_bf16in | 1.11x | 0.0609 | 0.1341 | 0.0496 |
| trunk.5.1.block.2.fc1 | se_fc | 64x672x28 | 0.0031 | bf16_triton | 0.0040 | int8_triton_bf16in | 0.76x | 0.0049 | n/a | n/a |
| trunk.5.1.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0052 | n/a | n/a |
| trunk.5.1.block.3.0 | pw1x1 | 12544x672x112 | 0.0502 | bf16_triton | 0.0300 | int8_triton_bf16in | 1.67x | 0.0294 | 0.0561 | 0.0285 |
| trunk.5.2.block.0.0 | pw1x1 | 12544x112x672 | 0.0664 | bf16_triton | 0.0600 | int8_triton_bf16in | 1.11x | 0.0609 | 0.1341 | 0.0496 |
| trunk.5.2.block.2.fc1 | se_fc | 64x672x28 | 0.0031 | bf16_triton | 0.0040 | int8_triton_bf16in | 0.76x | 0.0049 | n/a | n/a |
| trunk.5.2.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0052 | n/a | n/a |
| trunk.5.2.block.3.0 | pw1x1 | 12544x672x112 | 0.0502 | bf16_triton | 0.0300 | int8_triton_bf16in | 1.67x | 0.0294 | 0.0561 | 0.0285 |
| trunk.6.0.block.0.0 | pw1x1 | 12544x112x672 | 0.0664 | bf16_triton | 0.0600 | int8_triton_bf16in | 1.11x | 0.0609 | 0.1341 | 0.0496 |
| trunk.6.0.block.2.fc1 | se_fc | 64x672x28 | 0.0031 | bf16_triton | 0.0040 | int8_triton_bf16in | 0.76x | 0.0049 | n/a | n/a |
| trunk.6.0.block.2.fc2 | se_fc | 64x28x672 | 0.0015 | bf16_triton | 0.0015 | int8_triton_bf16in | 1.00x | 0.0052 | n/a | n/a |
| trunk.6.0.block.3.0 | pw1x1 | 3136x672x192 | 0.0223 | bf16_triton | 0.0173 | int8_triton_bf16in | 1.29x | 0.0174 | 0.0318 | 0.0157 |
| trunk.6.1.block.0.0 | pw1x1 | 3136x192x1152 | 0.0386 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.45x | 0.0310 | 0.0402 | 0.0224 |
| trunk.6.1.block.2.fc1 | se_fc | 64x1152x48 | 0.0040 | bf16_triton | 0.0049 | int8_triton_bf16in | 0.83x | 0.0054 | 0.0049 | 0.0036 |
| trunk.6.1.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.97x | 0.0049 | 0.0029 | 0.0023 |
| trunk.6.1.block.3.0 | pw1x1 | 3136x1152x192 | 0.0367 | bf16_triton | 0.0275 | int8_triton_bf16in | 1.33x | 0.0227 | 0.0456 | 0.0206 |
| trunk.6.2.block.0.0 | pw1x1 | 3136x192x1152 | 0.0386 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.45x | 0.0310 | 0.0402 | 0.0224 |
| trunk.6.2.block.2.fc1 | se_fc | 64x1152x48 | 0.0040 | bf16_triton | 0.0049 | int8_triton_bf16in | 0.83x | 0.0054 | 0.0049 | 0.0036 |
| trunk.6.2.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.97x | 0.0049 | 0.0029 | 0.0023 |
| trunk.6.2.block.3.0 | pw1x1 | 3136x1152x192 | 0.0367 | bf16_triton | 0.0275 | int8_triton_bf16in | 1.33x | 0.0227 | 0.0456 | 0.0206 |
| trunk.6.3.block.0.0 | pw1x1 | 3136x192x1152 | 0.0386 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.45x | 0.0310 | 0.0402 | 0.0224 |
| trunk.6.3.block.2.fc1 | se_fc | 64x1152x48 | 0.0040 | bf16_triton | 0.0049 | int8_triton_bf16in | 0.83x | 0.0054 | 0.0049 | 0.0036 |
| trunk.6.3.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.97x | 0.0049 | 0.0029 | 0.0023 |
| trunk.6.3.block.3.0 | pw1x1 | 3136x1152x192 | 0.0367 | bf16_triton | 0.0275 | int8_triton_bf16in | 1.33x | 0.0227 | 0.0456 | 0.0206 |
| trunk.7.0.block.0.0 | pw1x1 | 3136x192x1152 | 0.0386 | bf16_triton | 0.0265 | int8_triton_bf16in | 1.45x | 0.0310 | 0.0402 | 0.0224 |
| trunk.7.0.block.2.fc1 | se_fc | 64x1152x48 | 0.0040 | bf16_triton | 0.0049 | int8_triton_bf16in | 0.83x | 0.0054 | 0.0049 | 0.0036 |
| trunk.7.0.block.2.fc2 | se_fc | 64x48x1152 | 0.0020 | bf16_cublas | 0.0020 | int8_triton_bf16in | 0.97x | 0.0049 | 0.0029 | 0.0023 |
| trunk.7.0.block.3.0 | pw1x1 | 3136x1152x320 | 0.0568 | bf16_triton | 0.0369 | fp8_pipeline | 1.54x | 0.0323 | 0.0636 | 0.0305 |
| trunk.8.0 | pw1x1 | 3136x320x1280 | 0.0623 | bf16_triton | 0.0473 | int8_triton_bf16in | 1.32x | 0.0481 | 0.0700 | 0.0357 |
| head.proj | head_proj | 64x1280x512 | 0.0057 | bf16_cublas | 0.0050 | fp8_pipeline | 1.15x | 0.0070 | 0.0055 | 0.0039 |
| head.layers.0.mixer.in_proj | mamba_in_proj | 64x512x2048 | 0.0071 | bf16_cublas | 0.0050 | int8_triton_bf16in | 1.42x | 0.0069 | 0.0067 | 0.0040 |
| head.layers.0.mixer.x_proj | mamba_x_proj | 64x1024x32 | 0.0032 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.74x | 0.0051 | 0.0045 | 0.0034 |
| head.layers.0.mixer.dt_proj | mamba_dt_proj | 64x1024x1024 | 0.0072 | bf16_triton | 0.0049 | fp8_pipeline | 1.46x | 0.0082 | 0.0068 | 0.0039 |
| head.layers.0.mixer.out_proj | mamba_out_proj | 64x1024x512 | 0.0057 | bf16_triton | 0.0045 | fp8_pipeline | 1.28x | 0.0063 | 0.0048 | 0.0035 |
| head.layers.1.mixer.in_proj | mamba_in_proj | 64x512x2048 | 0.0071 | bf16_cublas | 0.0050 | int8_triton_bf16in | 1.42x | 0.0069 | 0.0067 | 0.0040 |
| head.layers.1.mixer.x_proj | mamba_x_proj | 64x1024x32 | 0.0032 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.74x | 0.0051 | 0.0045 | 0.0034 |
| head.layers.1.mixer.dt_proj | mamba_dt_proj | 64x1024x1024 | 0.0072 | bf16_triton | 0.0049 | fp8_pipeline | 1.46x | 0.0082 | 0.0068 | 0.0039 |
| head.layers.1.mixer.out_proj | mamba_out_proj | 64x1024x512 | 0.0057 | bf16_triton | 0.0045 | fp8_pipeline | 1.28x | 0.0063 | 0.0048 | 0.0035 |
| head.layers.2.mixer.in_proj | mamba_in_proj | 64x512x2048 | 0.0071 | bf16_cublas | 0.0050 | int8_triton_bf16in | 1.42x | 0.0069 | 0.0067 | 0.0040 |
| head.layers.2.mixer.x_proj | mamba_x_proj | 64x1024x32 | 0.0032 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.74x | 0.0051 | 0.0045 | 0.0034 |
| head.layers.2.mixer.dt_proj | mamba_dt_proj | 64x1024x1024 | 0.0072 | bf16_triton | 0.0049 | fp8_pipeline | 1.46x | 0.0082 | 0.0068 | 0.0039 |
| head.layers.2.mixer.out_proj | mamba_out_proj | 64x1024x512 | 0.0057 | bf16_triton | 0.0045 | fp8_pipeline | 1.28x | 0.0063 | 0.0048 | 0.0035 |
| head.layers.3.mixer.in_proj | mamba_in_proj | 64x512x2048 | 0.0071 | bf16_cublas | 0.0050 | int8_triton_bf16in | 1.42x | 0.0069 | 0.0067 | 0.0040 |
| head.layers.3.mixer.x_proj | mamba_x_proj | 64x1024x32 | 0.0032 | bf16_cublas | 0.0044 | int8_triton_bf16in | 0.74x | 0.0051 | 0.0045 | 0.0034 |
| head.layers.3.mixer.dt_proj | mamba_dt_proj | 64x1024x1024 | 0.0072 | bf16_triton | 0.0049 | fp8_pipeline | 1.46x | 0.0082 | 0.0068 | 0.0039 |
| head.layers.3.mixer.out_proj | mamba_out_proj | 64x1024x512 | 0.0057 | bf16_triton | 0.0045 | fp8_pipeline | 1.28x | 0.0063 | 0.0048 | 0.0035 |
| head.head.0 | head_mlp | 64x512x256 | 0.0028 | bf16_triton | 0.0030 | int8_triton_bf16in | 0.91x | 0.0047 | 0.0032 | 0.0026 |
| head.head.3 | head_mlp | 64x256x2 | 0.0024 | bf16_triton | 0.0022 | int8_triton_bf16in | 1.12x | 0.0041 | n/a | n/a |

## Batch 64: depthwise convolutions (bias + SiLU fused where marked)

| Layer | N x H x W x C, k, stride | cuDNN bf16 + SiLU | Triton bf16 NHWC+SiLU | Triton bf16 NCHW existing+SiLU | Triton int8 NHWC+SiLU (int8 input ready) | int8 incl. quantize pass | bf16 GB/s | int8 GB/s | int8 vs best bf16 | int8 incl. quantize vs best bf16 |
|---|---|---|---|---|---|---|---|---|---|---|
| trunk.1.0.block.0.0 | 64x112x112x32, 3, s1 | 1.2329 | 0.4076 | 0.4094 | 0.3620 | 0.6697 | 252 | 213 | 1.13x | 0.61x |
| trunk.2.0.block.1.0 | 64x112x112x96, 3, s2 | 1.3869 | 0.7263 | 0.7820 | 0.4386 | 1.3617 | 265 | 264 | 1.66x | 0.53x |
| trunk.2.1.block.1.0 | 64x56x56x144, 3, s1 | 1.3735 | 0.5278 | 0.5668 | 0.4567 | 0.7972 | 219 | 190 | 1.16x | 0.66x |
| trunk.3.0.block.1.0 | 64x56x56x144, 5, s2 | 5.6339 | 0.3081 | 0.3227 | 0.3170 | 0.6387 | 235 | 137 | 0.97x | 0.48x |
| trunk.3.1.block.1.0 | 64x28x28x240, 5, s1 | 0.4804 | 0.2467 | 0.4335 | 0.3508 | 0.4914 | 195 | 103 | 0.70x | 0.50x |
| trunk.4.0.block.1.0 | 64x28x28x240, 3, s2 | 0.1063 | 0.0344 | 0.0587 | 0.0353 | 0.1568 | 875 | 512 | 0.98x | 0.22x |
| trunk.4.1.block.1.0 | 64x14x14x480, 3, s1 | 0.1123 | 0.0634 | 0.1120 | 0.0777 | 0.0913 | 380 | 232 | 0.82x | 0.69x |
| trunk.4.2.block.1.0 | 64x14x14x480, 3, s1 | 0.1123 | 0.0634 | 0.1120 | 0.0777 | 0.0913 | 380 | 232 | 0.82x | 0.69x |
| trunk.5.0.block.1.0 | 64x14x14x480, 5, s1 | 0.1222 | 0.0935 | 0.2138 | 0.1663 | 0.1819 | 258 | 109 | 0.56x | 0.51x |
| trunk.5.1.block.1.0 | 64x14x14x672, 5, s1 | 0.2278 | 0.1592 | 0.3197 | 0.2469 | 0.3487 | 212 | 102 | 0.64x | 0.46x |
| trunk.5.2.block.1.0 | 64x14x14x672, 5, s1 | 0.2278 | 0.1592 | 0.3197 | 0.2469 | 0.3487 | 212 | 102 | 0.64x | 0.46x |
| trunk.6.0.block.1.0 | 64x14x14x672, 5, s2 | 0.6120 | 0.0443 | 0.0777 | 0.0805 | 0.0992 | 477 | 157 | 0.55x | 0.45x |
| trunk.6.1.block.1.0 | 64x7x7x1152, 5, s1 | 0.0647 | 0.0608 | 0.1316 | 0.0944 | 0.1046 | 239 | 115 | 0.64x | 0.58x |
| trunk.6.2.block.1.0 | 64x7x7x1152, 5, s1 | 0.0647 | 0.0608 | 0.1316 | 0.0944 | 0.1046 | 239 | 115 | 0.64x | 0.58x |
| trunk.6.3.block.1.0 | 64x7x7x1152, 5, s1 | 0.0647 | 0.0608 | 0.1316 | 0.0944 | 0.1046 | 239 | 115 | 0.64x | 0.58x |
| trunk.7.0.block.1.0 | 64x7x7x1152, 3, s1 | 0.0472 | 0.0286 | 0.0697 | 0.0357 | 0.0452 | 507 | 304 | 0.80x | 0.63x |

