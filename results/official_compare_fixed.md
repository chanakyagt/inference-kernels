# Official-kernel comparison

GPU NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130, vLLM 0.33.0, official mamba-ssm commit e9594ce. Timing: CUDA events, median of 200. idle gate met: True. SM clock at start/end: 210/2760 MHz, temp 35/70 C.

GPU-time column = CUDA graph of 5 back-to-back calls divided by 5 (no CPU launch cost). Eager column = events around one Python call (includes launch cost). Ratio = official time / ours; above 1 means ours is faster. Ours = best of a config sweep (CUDA kernel threads/channels and split; Triton block size) vs the official kernels at their fixed settings.

## Full-sequence scan (GPU time, ms)

| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | official scan | official/ours(CUDA) | vLLM/ours(CUDA) | official/ours(Triton) | CUDA GB/s | vLLM GB/s | err ours CUDA | err vLLM | err official |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | float32 | - | - | 0.0193 | 0.0190 | - | - | - | - | 23.0 | nan | 9.54e-07 | 9.54e-07 |
| 1 | 1024 | 300 | float32 | - | - | 0.0774 | 0.0535 | - | - | - | - | 49.0 | nan | 1.43e-06 | 1.43e-06 |
| 1 | 1024 | 3000 | float32 | - | - | 0.4731 | 0.4280 | - | - | - | - | 78.9 | nan | 1.91e-06 | 1.91e-06 |
| 1 | 2048 | 30 | float32 | - | - | 0.0360 | 0.0342 | - | - | - | - | 24.4 | nan | 9.54e-07 | 9.54e-07 |
| 1 | 2048 | 300 | float32 | - | - | 0.1526 | 0.1014 | - | - | - | - | 49.5 | nan | 1.91e-06 | 1.91e-06 |
| 1 | 2048 | 3000 | float32 | - | - | 0.9375 | 0.8527 | - | - | - | - | 79.2 | nan | 2.38e-06 | 2.38e-06 |
| 4 | 1024 | 30 | float32 | - | - | 0.0674 | 0.0651 | - | - | - | - | 23.1 | nan | 1.43e-06 | 1.43e-06 |
| 4 | 1024 | 300 | float32 | - | - | 0.3035 | 0.1982 | - | - | - | - | 49.3 | nan | 1.91e-06 | 1.91e-06 |
| 4 | 1024 | 3000 | float32 | - | - | 1.8940 | 1.7134 | - | - | - | - | 78.7 | nan | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 30 | float32 | - | - | 0.1288 | 0.1268 | - | - | - | - | 24.1 | nan | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 300 | float32 | - | - | 0.5877 | 0.3887 | - | - | - | - | 50.7 | nan | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 3000 | float32 | - | - | 3.7706 | 3.4421 | - | - | - | - | 78.7 | nan | 2.38e-06 | 2.38e-06 |
| 1 | 1024 | 30 | bfloat16 | - | - | 0.0203 | 0.0184 | - | - | - | - | 12.6 | nan | 1.36e-02 | 1.36e-02 |
| 1 | 1024 | 300 | bfloat16 | - | - | 0.0782 | 0.0516 | - | - | - | - | 24.7 | nan | 1.56e-02 | 1.56e-02 |
| 1 | 1024 | 3000 | bfloat16 | - | - | 0.5050 | 0.3973 | - | - | - | - | 37.0 | nan | 2.76e-02 | 2.76e-02 |
| 1 | 2048 | 30 | bfloat16 | - | - | 0.0370 | 0.0344 | - | - | - | - | 13.8 | nan | 1.51e-02 | 1.51e-02 |
| 1 | 2048 | 300 | bfloat16 | - | - | 0.1528 | 0.0983 | - | - | - | - | 25.2 | nan | 1.56e-02 | 1.56e-02 |
| 1 | 2048 | 3000 | bfloat16 | - | - | 0.9806 | 0.7895 | - | - | - | - | 37.9 | nan | 2.54e-02 | 2.54e-02 |
| 4 | 1024 | 30 | bfloat16 | - | - | 0.0668 | 0.0649 | - | - | - | - | 12.2 | nan | 1.55e-02 | 1.55e-02 |
| 4 | 1024 | 300 | bfloat16 | - | - | 0.3154 | 0.1904 | - | - | - | - | 23.8 | nan | 1.56e-02 | 1.56e-02 |
| 4 | 1024 | 3000 | bfloat16 | - | - | 1.9505 | 1.5801 | - | - | - | - | 38.2 | nan | 2.82e-02 | 2.82e-02 |
| 4 | 2048 | 30 | bfloat16 | - | - | 0.1282 | 0.1266 | - | - | - | - | 12.6 | nan | 1.56e-02 | 1.56e-02 |
| 4 | 2048 | 300 | bfloat16 | - | - | 0.6105 | 0.3739 | - | - | - | - | 24.5 | nan | 1.56e-02 | 1.56e-02 |
| 4 | 2048 | 3000 | bfloat16 | - | - | 3.8754 | 3.1579 | - | - | - | - | 38.3 | nan | 3.05e-02 | 3.05e-02 |

## Full-sequence scan (eager, ms, includes Python launch cost)

| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | vLLM scan + delta restore | official scan | official scan + layout transposes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | float32 | - | - | 0.0467 | 0.0573 | 0.0246 | 0.1219 |
| 1 | 1024 | 300 | float32 | - | - | 0.1126 | 0.1248 | 0.0588 | 0.1348 |
| 1 | 1024 | 3000 | float32 | - | - | 0.5080 | 0.5304 | 0.4260 | 0.7690 |
| 1 | 2048 | 30 | float32 | - | - | 0.0593 | 0.0604 | 0.0368 | 0.0911 |
| 1 | 2048 | 300 | float32 | - | - | 0.1740 | 0.1792 | 0.1044 | 0.1710 |
| 1 | 2048 | 3000 | float32 | - | - | 0.9626 | 1.1397 | 0.8396 | 1.9282 |
| 4 | 1024 | 30 | float32 | - | - | 0.1113 | 0.0942 | 0.0706 | 0.1331 |
| 4 | 1024 | 300 | float32 | - | - | 0.3215 | 0.3308 | 0.2007 | 0.3174 |
| 4 | 1024 | 3000 | float32 | - | - | 1.8452 | 2.2158 | 1.6527 | 3.0996 |
| 4 | 2048 | 30 | float32 | - | - | 0.1680 | 0.1823 | 0.1290 | 0.1928 |
| 4 | 2048 | 300 | float32 | - | - | 0.6093 | 0.6164 | 0.3919 | 0.6349 |
| 4 | 2048 | 3000 | float32 | - | - | 3.6977 | 4.4657 | 3.3985 | 7.9658 |
| 1 | 1024 | 30 | bfloat16 | - | - | 0.0377 | 0.0440 | 0.0205 | 0.0729 |
| 1 | 1024 | 300 | bfloat16 | - | - | 0.1196 | 0.1302 | 0.0543 | 0.1843 |
| 1 | 1024 | 3000 | bfloat16 | - | - | 0.5263 | 0.5366 | 0.3994 | 0.6640 |
| 1 | 2048 | 30 | bfloat16 | - | - | 0.0521 | 0.0614 | 0.0369 | 0.0922 |
| 1 | 2048 | 300 | bfloat16 | - | - | 0.1946 | 0.2058 | 0.1012 | 0.2403 |
| 1 | 2048 | 3000 | bfloat16 | - | - | 0.9830 | 1.0449 | 0.7939 | 1.3066 |
| 4 | 1024 | 30 | bfloat16 | - | - | 0.1104 | 0.0939 | 0.0676 | 0.1219 |
| 4 | 1024 | 300 | bfloat16 | - | - | 0.3487 | 0.3451 | 0.1929 | 0.2959 |
| 4 | 1024 | 3000 | bfloat16 | - | - | 1.9763 | 2.1226 | 1.5698 | 2.6120 |
| 4 | 2048 | 30 | bfloat16 | - | - | 0.1709 | 0.1772 | 0.1321 | 0.1932 |
| 4 | 2048 | 300 | bfloat16 | - | - | 0.6431 | 0.6676 | 0.3840 | 0.5878 |
| 4 | 2048 | 3000 | bfloat16 | - | - | 3.8789 | 4.2680 | 3.1130 | 5.3780 |

## Single-token state update, like-for-like core (fp32, GPU time ms)

| B | E | ours Triton | official Triton | vLLM Triton | official/ours | vLLM/ours | ours GB/s | official GB/s | vLLM GB/s | err y ours | err y official | err y vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Single-token state update, eager (ms)

| B | E | ours Triton | official Triton | vLLM Triton |
|---|---|---|---|---|

## Whole Mamba layer step: our fused unit vs a composed official pipeline (fp32, ms)

Our unit is LayerNorm + in_proj + conv step + x_proj + dt_proj + scan step + gate + out_proj + residual in two Triton launches. The composed baseline is PyTorch ops for everything except the state update, which uses the official or vLLM Triton kernel. This compares granularity, not just the scan.

| B | E | ours GPU | composed+official GPU | composed+vLLM GPU | ours eager | composed+official eager | composed+vLLM eager | composed-official/ours GPU | ours GB/s | max abs diff |
|---|---|---|---|---|---|---|---|---|---|---|

