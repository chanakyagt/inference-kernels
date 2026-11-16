# Real kernels versus simulated quantization (diloco, 512 Test images)

The bf16 path with no quantization already differs from FP32 by KL 4.53e-03; that is the noise floor for the real kernels.

| Layer | Kind | Scheme | KL simulated (fp32) | KL real kernels (bf16) | Mean abs logit diff |
|---|---|---|---|---|---|
| trunk.4.0.block.1.0 | dw3x3 | W8A8pc | 3.15e-04 | 4.93e-03 | 0.140 |
| trunk.5.2.block.1.0 | dw5x5 | W8A8pc | 5.02e-05 | 4.68e-03 | 0.136 |
| head.head.3 | head_mlp | W8A8 | 2.38e-05 | 4.53e-03 | 0.140 |
| head.proj | head_proj | W8A8 | 6.91e-04 | 5.31e-03 | 0.157 |
| head.proj | head_proj | FP8 | 5.78e-04 | 5.74e-03 | 0.157 |
| head.layers.2.mixer.dt_proj | mamba_dt_proj | W8A8 | 5.65e-08 | 4.54e-03 | 0.139 |
| head.layers.2.mixer.dt_proj | mamba_dt_proj | FP8 | 2.97e-08 | 4.54e-03 | 0.139 |
| head.layers.2.mixer.in_proj | mamba_in_proj | W8A8 | 1.73e-06 | 4.54e-03 | 0.139 |
| head.layers.2.mixer.in_proj | mamba_in_proj | FP8 | 3.21e-06 | 4.53e-03 | 0.138 |
| head.layers.2.mixer.out_proj | mamba_out_proj | W8A8 | 1.68e-03 | 5.43e-03 | 0.144 |
| head.layers.2.mixer.out_proj | mamba_out_proj | FP8 | 3.38e-06 | 4.54e-03 | 0.140 |
| head.layers.2.mixer.x_proj | mamba_x_proj | W8A8 | 2.33e-07 | 4.54e-03 | 0.139 |
| head.layers.2.mixer.x_proj | mamba_x_proj | FP8 | 1.61e-07 | 4.53e-03 | 0.139 |
| trunk.5.0.block.3.0 | pw1x1 | W8A8 | 5.59e-04 | 4.51e-03 | 0.147 |
| trunk.5.0.block.3.0 | pw1x1 | FP8 | 1.99e-03 | 6.65e-03 | 0.149 |
| trunk.5.0.block.2.fc1 | se_fc | W8A8 | 6.35e-05 | 4.51e-03 | 0.136 |

All layers int8 (W8A8): simulated KL 1.70e-01, accuracy 0.6406; real kernels KL 1.76e-01, accuracy 0.6641. FP32 accuracy on these images 0.6934.
