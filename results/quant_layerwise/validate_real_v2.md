# Real kernels versus simulated quantization (v2, 512 Test images)

The bf16 path with no quantization already differs from FP32 by KL 3.90e-04; that is the noise floor for the real kernels.

| Layer | Kind | Scheme | KL simulated (fp32) | KL real kernels (bf16) | Mean abs logit diff |
|---|---|---|---|---|---|
| trunk.4.0.block.1.0 | dw3x3 | W8A8pc | 2.65e-05 | 3.57e-04 | 0.015 |
| trunk.5.2.block.1.0 | dw5x5 | W8A8pc | 3.08e-06 | 3.91e-04 | 0.014 |
| head.1 | head_fc | W8A8 | 7.05e-06 | 3.98e-04 | 0.014 |
| trunk.5.0.block.3.0 | pw1x1 | W8A8 | 1.21e-04 | 6.08e-04 | 0.016 |
| trunk.5.0.block.3.0 | pw1x1 | FP8 | 1.00e-04 | 4.15e-04 | 0.015 |
| trunk.5.0.block.2.fc1 | se_fc | W8A8 | 2.27e-06 | 3.93e-04 | 0.014 |

All layers int8 (W8A8): simulated KL 1.58e-02, accuracy 0.8926; real kernels KL 1.92e-02, accuracy 0.8848. FP32 accuracy on these images 0.9004.
