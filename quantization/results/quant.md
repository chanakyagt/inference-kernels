# Precision vs speed (4 clips x 30 frames, untrained weights, RTX 4060 Ti)

Error is measured against the same model in FP32. The selective-scan recurrence stays in FP32 in every row.
Detection accuracy needs trained weights and a labeled test set, which this repository does not have yet,
so the error columns measure numerical drift only, not accuracy loss.

| Variant | Linears swapped | Linear weight MB | Mamba output rel. error vs FP32 | Logit rel. error | Max change in P(fake) | Full forward ms | Mamba + head ms |
|---|---|---|---|---|---|---|---|
| BF16 (fused kernels) | 0 | 22.8 | 9.99e-03 | 7.92e-03 | 0.0012 | 38.86 | 2.080 |
| FP8 Mamba + projection linears | 17 | 11.5 | 8.43e-02 | 6.62e-02 | 0.0051 | 40.04 | 5.201 |
| FP8 all linears | 18 | 11.4 | 8.43e-02 | 7.00e-02 | 0.0077 | 40.38 | 7.242 |
| INT8 weight-only Mamba + projection | 17 | 11.5 | 1.56e-02 | 1.04e-02 | 0.0012 | 39.19 | 2.853 |
