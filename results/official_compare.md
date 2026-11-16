# Official-kernel comparison

GPU NVIDIA GeForce RTX 4060 Ti, torch 2.14.0+cu130, vLLM 0.33.0, official mamba-ssm commit e9594ce. Timing: CUDA events, median of 200. idle gate met: True. SM clock at start/end: 210/2790 MHz, temp 41/72 C.

GPU-time column = CUDA graph of 5 back-to-back calls divided by 5 (no CPU launch cost). Eager column = events around one Python call (includes launch cost). Ratio = official time / ours; above 1 means ours is faster. Ours = best of a config sweep (CUDA kernel threads/channels and split; Triton block size) vs the official kernels at their fixed settings.

Notes: quote the fp32 rows only. The bf16 rows were disturbed (the idle check ran once, at the start), so they are not quoted. "Ours" in the tables is the best of a per-shape sweep; see `../compare_official/README.md` for the fixed-default column. The GB/s columns are analytic bytes divided by time; values above 288 GB/s come from repeated timing keeping the data in the 32 MB L2 cache, so they are not DRAM bandwidth. vLLM's full-scan output is bit-identical to the official kernel's in all 24 rows, so it is the same kernel and not an independent baseline.

## Full-sequence scan (GPU time, ms)

| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | official scan | official/ours(CUDA) | vLLM/ours(CUDA) | official/ours(Triton) | CUDA GB/s | vLLM GB/s | err ours CUDA | err vLLM | err official |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | float32 | 0.0053 | 0.0078 | 0.0214 | 0.0206 | 3.88 | 4.03 | 2.65 | 83.0 | 20.6 | 5.96e-07 | 9.54e-07 | 9.54e-07 |
| 1 | 1024 | 300 | float32 | 0.0307 | 0.0584 | 0.0776 | 0.0530 | 1.73 | 2.53 | 0.91 | 123.5 | 48.9 | 1.07e-06 | 1.43e-06 | 1.43e-06 |
| 1 | 1024 | 3000 | float32 | 0.4588 | 1.0772 | 0.4735 | 0.4292 | 0.94 | 1.03 | 0.40 | 81.3 | 78.8 | 1.91e-06 | 1.91e-06 | 1.91e-06 |
| 1 | 2048 | 30 | float32 | 0.0049 | 0.0076 | 0.0362 | 0.0344 | 7.00 | 7.37 | 4.54 | 179.1 | 24.3 | 7.15e-07 | 9.54e-07 | 9.54e-07 |
| 1 | 2048 | 300 | float32 | 0.0319 | 0.0629 | 0.1532 | 0.1019 | 3.20 | 4.80 | 1.62 | 236.8 | 49.3 | 1.91e-06 | 1.91e-06 | 1.91e-06 |
| 1 | 2048 | 3000 | float32 | 0.5218 | 1.1384 | 0.9482 | 0.8710 | 1.67 | 1.82 | 0.77 | 142.3 | 78.3 | 1.67e-06 | 2.38e-06 | 2.38e-06 |
| 4 | 1024 | 30 | float32 | 0.0061 | 0.0087 | 0.0680 | 0.0657 | 10.72 | 11.10 | 7.51 | 254.6 | 22.9 | 9.54e-07 | 1.43e-06 | 1.43e-06 |
| 4 | 1024 | 300 | float32 | 0.0387 | 0.0733 | 0.3047 | 0.1993 | 5.15 | 7.87 | 2.72 | 386.7 | 49.1 | 1.67e-06 | 1.91e-06 | 1.91e-06 |
| 4 | 1024 | 3000 | float32 | 0.6703 | 1.2919 | 1.9017 | 1.7395 | 2.60 | 2.84 | 1.35 | 222.4 | 78.4 | 1.91e-06 | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 30 | float32 | 0.0086 | 0.0111 | 0.1294 | 0.1274 | 14.85 | 15.09 | 11.52 | 361.9 | 24.0 | 9.54e-07 | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 300 | float32 | 0.0627 | 0.0981 | 0.5937 | 0.3938 | 6.28 | 9.47 | 4.01 | 475.3 | 50.2 | 1.43e-06 | 1.91e-06 | 1.91e-06 |
| 4 | 2048 | 3000 | float32 | 1.1536 | 1.5606 | 3.8222 | 3.4823 | 3.02 | 3.31 | 2.23 | 257.1 | 77.6 | 1.91e-06 | 2.38e-06 | 2.38e-06 |
| 1 | 1024 | 30 | bfloat16 | 0.0049 | 0.0071 | 0.0194 | 0.0186 | 3.79 | 3.94 | 2.62 | 52.1 | 13.2 | 1.36e-02 | 1.36e-02 | 1.36e-02 |
| 1 | 1024 | 300 | bfloat16 | 0.0319 | 0.0596 | 0.0786 | 0.0518 | 1.62 | 2.46 | 0.87 | 60.5 | 24.6 | 1.56e-02 | 1.56e-02 | 1.56e-02 |
| 1 | 1024 | 3000 | bfloat16 | 0.3004 | 0.5847 | 0.5087 | 0.4010 | 1.33 | 1.69 | 0.69 | 62.2 | 36.7 | 2.76e-02 | 2.76e-02 | 2.76e-02 |
| 1 | 2048 | 30 | bfloat16 | 0.0051 | 0.0075 | 0.0366 | 0.0346 | 6.76 | 7.15 | 4.61 | 99.6 | 13.9 | 1.51e-02 | 1.51e-02 | 1.51e-02 |
| 1 | 2048 | 300 | bfloat16 | 0.0328 | 0.0614 | 0.1537 | 0.0989 | 3.02 | 4.69 | 1.61 | 117.4 | 25.0 | 1.56e-02 | 1.56e-02 | 1.56e-02 |
| 1 | 2048 | 3000 | bfloat16 | 0.4880 | 1.0926 | 0.9861 | 0.8014 | 1.64 | 2.02 | 0.73 | 76.2 | 37.7 | 2.54e-02 | 2.54e-02 | 2.54e-02 |
| 4 | 1024 | 30 | bfloat16 | 0.0059 | 0.0082 | 0.0672 | 0.0653 | 11.00 | 11.31 | 7.99 | 137.2 | 12.1 | 1.55e-02 | 1.55e-02 | 1.55e-02 |
| 4 | 1024 | 300 | bfloat16 | 0.0403 | 0.0680 | 0.3166 | 0.1914 | 4.74 | 7.85 | 2.82 | 186.4 | 23.7 | 1.56e-02 | 1.56e-02 | 1.56e-02 |
| 4 | 1024 | 3000 | bfloat16 | 0.5573 | 1.1734 | 1.9687 | 3.8878 | 6.98 | 3.53 | 3.31 | 133.8 | 37.9 | 2.82e-02 | 2.82e-02 | 2.82e-02 |
| 4 | 2048 | 30 | bfloat16 | 0.0111 | 0.0139 | 0.2705 | 0.2687 | 24.30 | 24.46 | 19.29 | 146.6 | 6.0 | 1.56e-02 | 1.56e-02 | 1.56e-02 |
| 4 | 2048 | 300 | bfloat16 | 0.0638 | 0.1073 | 0.6146 | 0.3793 | 5.94 | 9.63 | 3.54 | 234.4 | 24.3 | 1.56e-02 | 1.56e-02 | 1.56e-02 |
| 4 | 2048 | 3000 | bfloat16 | 0.8219 | 1.4299 | 3.9096 | 3.2250 | 3.92 | 4.76 | 2.26 | 180.5 | 37.9 | 3.05e-02 | 3.05e-02 | 3.05e-02 |

## Full-sequence scan (eager, ms, includes Python launch cost)

| B | E | T | dtype | ours CUDA | ours Triton | vLLM scan | vLLM scan + delta restore | official scan | official scan + layout transposes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | float32 | 0.0154 | 0.0338 | 0.0460 | 0.0519 | 0.0235 | 0.0861 |
| 1 | 1024 | 300 | float32 | 0.0378 | 0.0870 | 0.0952 | 0.1055 | 0.0605 | 0.1209 |
| 1 | 1024 | 3000 | float32 | 0.4700 | 1.1058 | 0.4945 | 0.5304 | 0.4280 | 0.7680 |
| 1 | 2048 | 30 | float32 | 0.0195 | 0.0379 | 0.0543 | 0.0665 | 0.0369 | 0.0993 |
| 1 | 2048 | 300 | float32 | 0.0338 | 0.0938 | 0.1730 | 0.1833 | 0.1044 | 0.1823 |
| 1 | 2048 | 3000 | float32 | 0.5229 | 1.1612 | 0.9554 | 1.1458 | 0.8520 | 1.9547 |
| 4 | 1024 | 30 | float32 | 0.0276 | 0.0367 | 0.0868 | 0.1003 | 0.0685 | 0.1331 |
| 4 | 1024 | 300 | float32 | 0.0411 | 0.0996 | 0.3300 | 0.3410 | 0.2017 | 0.3172 |
| 4 | 1024 | 3000 | float32 | 0.6758 | 1.3209 | 1.8959 | 2.2467 | 1.7265 | 3.1650 |
| 4 | 2048 | 30 | float32 | 0.0174 | 0.0389 | 0.1495 | 0.1596 | 0.1300 | 0.1976 |
| 4 | 2048 | 300 | float32 | 0.0653 | 0.1249 | 0.6144 | 0.6277 | 0.3942 | 0.6488 |
| 4 | 2048 | 3000 | float32 | 1.1621 | 1.5903 | 3.8011 | 4.5618 | 3.4923 | 8.0379 |
| 1 | 1024 | 30 | bfloat16 | 0.0205 | 0.0399 | 0.0379 | 0.0510 | 0.0206 | 0.0942 |
| 1 | 1024 | 300 | bfloat16 | 0.0340 | 0.0850 | 0.0942 | 0.1075 | 0.0543 | 0.1208 |
| 1 | 1024 | 3000 | bfloat16 | 0.3029 | 0.6134 | 0.5274 | 0.5434 | 0.4004 | 0.6687 |
| 1 | 2048 | 30 | bfloat16 | 0.0141 | 0.0338 | 0.0563 | 0.0655 | 0.0369 | 0.1014 |
| 1 | 2048 | 300 | bfloat16 | 0.0348 | 0.0881 | 0.1731 | 0.1843 | 0.1014 | 0.1772 |
| 1 | 2048 | 3000 | bfloat16 | 0.4913 | 1.1151 | 1.0060 | 1.0537 | 0.7997 | 1.3301 |
| 4 | 1024 | 30 | bfloat16 | 0.0149 | 0.0338 | 0.0993 | 0.0993 | 0.0676 | 0.1362 |
| 4 | 1024 | 300 | bfloat16 | 0.0429 | 0.0952 | 0.3368 | 0.3488 | 0.1939 | 0.3000 |
| 4 | 1024 | 3000 | bfloat16 | 0.5631 | 1.2104 | 1.9948 | 6.1158 | 4.2388 | 7.2934 |
| 4 | 2048 | 30 | bfloat16 | 0.0205 | 0.0236 | 0.1393 | 0.1434 | 0.1321 | 0.1597 |
| 4 | 2048 | 300 | bfloat16 | 0.0664 | 0.1434 | 0.6356 | 0.6563 | 0.3850 | 0.5939 |
| 4 | 2048 | 3000 | bfloat16 | 2.1043 | 1.4612 | 3.9168 | 4.2947 | 3.1887 | 5.4976 |

## Single-token state update, like-for-like core (fp32, GPU time ms)

| B | E | ours Triton | official Triton | vLLM Triton | official/ours | vLLM/ours | ours GB/s | official GB/s | vLLM GB/s | err y ours | err y official | err y vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 0.0012 | - | 0.0011 | - | 0.96 | 181.0 | - | 189.2 | 1.9e-06 | nan | 1.9e-06 |
| 8 | 1024 | 0.0020 | - | 0.0019 | - | 0.98 | 613.9 | - | 625.8 | 1.9e-06 | nan | 2.4e-06 |
| 64 | 1024 | 0.0077 | - | 0.0069 | - | 0.89 | 1204.8 | - | 1348.7 | 1.9e-06 | nan | 2.4e-06 |
| 1 | 2048 | 0.0013 | - | 0.0013 | - | 1.03 | 332.9 | - | 323.2 | 1.9e-06 | nan | 1.9e-06 |
| 8 | 2048 | 0.0029 | - | 0.0028 | - | 0.95 | 834.0 | - | 880.4 | 1.9e-06 | nan | 1.9e-06 |
| 64 | 2048 | 0.0137 | - | 0.0131 | - | 0.96 | 1353.1 | - | 1413.1 | 1.9e-06 | nan | 2.4e-06 |

## Single-token state update, eager (ms)

| B | E | ours Triton | official Triton | vLLM Triton |
|---|---|---|---|---|
| 1 | 1024 | 0.0334 | - | 0.0709 |
| 8 | 1024 | 0.0287 | - | 0.0816 |
| 64 | 1024 | 0.0287 | - | 0.0749 |
| 1 | 2048 | 0.0346 | - | 0.0727 |
| 8 | 2048 | 0.0295 | - | 0.0721 |
| 64 | 2048 | 0.0334 | - | 0.0696 |

## Whole Mamba layer step: our fused unit vs a composed official pipeline (fp32, ms)

Our unit is LayerNorm + in_proj + conv step + x_proj + dt_proj + scan step + gate + out_proj + residual in two Triton launches. The composed baseline is PyTorch ops for everything except the state update, which uses the official or vLLM Triton kernel. This compares granularity, not just the scan.

| B | E | ours GPU | composed+official GPU | composed+vLLM GPU | ours eager | composed+official eager | composed+vLLM eager | composed-official/ours GPU | ours GB/s | max abs diff |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 0.0186 | 0.0244 | 0.0247 | 0.0451 | 0.3022 | 0.2964 | 1.31 | 584.4 | 0.000002 |
| 8 | 1024 | 0.0648 | 0.0493 | 0.0493 | 0.0952 | 0.2836 | 0.3083 | 0.76 | 185.0 | 0.000001 |
| 64 | 1024 | 0.4727 | 0.0748 | 0.0745 | 0.5007 | 0.3111 | 0.3115 | 0.16 | 44.3 | 0.000001 |
| 1 | 2048 | 0.1656 | 0.0984 | 0.0990 | 0.4344 | 0.2918 | 0.2801 | 0.59 | 258.0 | 0.000002 |
| 8 | 2048 | 0.3083 | 0.2168 | 0.2178 | 0.3615 | 0.2808 | 0.3075 | 0.70 | 145.8 | 0.000001 |
| 64 | 2048 | 2.2715 | 0.3081 | 0.3063 | 2.2619 | 0.3575 | 0.3738 | 0.14 | 27.7 | 0.000002 |

## Errors recorded during the run

The official Triton single-token kernel raised `TypeError: Value after * must be an iterable, not int` on all 6 shapes, so its columns above are "-". The JSON lists 558 errors in total; the other 552 are `AttributeError: module 'selective_scan_cuda' has no attribute 'scan'` from a first run (a name clash with the official extension). The scan section was rerun correctly and merged, so those stale entries are left out here and kept in `official_compare.json`.

## Fixed-config rerun (one CUDA setting for every shape, idle check before every row, cold-L2 variant)

### float32: one fixed CUDA setting (16 channels per block, split 2) for every shape, idle check before every row

Warm = CUDA graph of 5 calls divided by 5 (inputs stay in L2). Cold L2 = a 64 MB buffer is written before every call and the call is timed with CUDA events behind that write (so launch latency is hidden); the warm-event column is the same event method without the flush. Speedup = official / ours.

| B | E | T | ours fixed (ms) | ours tuned (ms, earlier run) | official (ms) | vLLM (ms) | speedup fixed | speedup tuned | ours cold | official cold | speedup cold | idle met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | 0.0053 | 0.0053 | 0.0190 | 0.0193 | 3.58 | 3.88 | 0.0135 | 0.0256 | 1.89 | True |
| 1 | 1024 | 300 | 0.0321 | 0.0307 | 0.0535 | 0.0774 | 1.67 | 1.73 | 0.0686 | 0.0707 | 1.03 | True |
| 1 | 1024 | 3000 | 0.4559 | 0.4588 | 0.4280 | 0.4731 | 0.94 | 0.94 | 0.4905 | 0.4444 | 0.91 | True |
| 1 | 2048 | 30 | 0.0059 | 0.0049 | 0.0342 | 0.0360 | 5.76 | 7.00 | 0.0164 | 0.0420 | 2.56 | True |
| 1 | 2048 | 300 | 0.0340 | 0.0319 | 0.1014 | 0.1526 | 2.98 | 3.20 | 0.0809 | 0.1219 | 1.51 | True |
| 1 | 2048 | 3000 | 0.5222 | 0.5218 | 0.8527 | 0.9375 | 1.63 | 1.67 | 0.5622 | 0.8376 | 1.49 | True |
| 4 | 1024 | 30 | 0.0063 | 0.0061 | 0.0651 | 0.0674 | 10.32 | 10.72 | 0.0225 | 0.0717 | 3.18 | True |
| 4 | 1024 | 300 | 0.0414 | 0.0387 | 0.1982 | 0.3035 | 4.79 | 5.15 | 0.1004 | 0.2191 | 2.18 | True |
| 4 | 1024 | 3000 | 0.6638 | 0.6703 | 1.7134 | 1.8940 | 2.58 | 2.60 | 0.7075 | 1.6558 | 2.34 | True |
| 4 | 2048 | 30 | 0.0111 | 0.0086 | 0.1268 | 0.1288 | 11.46 | 14.85 | 0.0297 | 0.1332 | 4.49 | True |
| 4 | 2048 | 300 | 0.0674 | 0.0627 | 0.3887 | 0.5877 | 5.77 | 6.28 | 0.1587 | 0.4137 | 2.61 | True |
| 4 | 2048 | 3000 | 1.1555 | 1.1536 | 3.4421 | 3.7706 | 2.98 | 3.02 | 1.1935 | 3.4017 | 2.85 | True |

### bfloat16: one fixed CUDA setting (16 channels per block, split 2) for every shape, idle check before every row

Warm = CUDA graph of 5 calls divided by 5 (inputs stay in L2). Cold L2 = a 64 MB buffer is written before every call and the call is timed with CUDA events behind that write (so launch latency is hidden); the warm-event column is the same event method without the flush. Speedup = official / ours.

| B | E | T | ours fixed (ms) | ours tuned (ms, earlier run) | official (ms) | vLLM (ms) | speedup fixed | speedup tuned | ours cold | official cold | speedup cold | idle met |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | 0.0051 | - | 0.0184 | 0.0203 | 3.60 | - | 0.0133 | 0.0246 | 1.85 | True |
| 1 | 1024 | 300 | 0.0327 | - | 0.0516 | 0.0782 | 1.58 | - | 0.0676 | 0.0604 | 0.89 | True |
| 1 | 1024 | 3000 | 0.2992 | - | 0.3973 | 0.5050 | 1.33 | - | 0.4813 | 0.4178 | 0.87 | True |
| 1 | 2048 | 30 | 0.0053 | - | 0.0344 | 0.0370 | 6.53 | - | 0.0154 | 0.0399 | 2.60 | True |
| 1 | 2048 | 300 | 0.0352 | - | 0.0983 | 0.1528 | 2.79 | - | 0.0737 | 0.1096 | 1.49 | True |
| 1 | 2048 | 3000 | 0.4868 | - | 0.7895 | 0.9806 | 1.62 | - | 0.5100 | 0.8008 | 1.57 | True |
| 4 | 1024 | 30 | 0.0063 | - | 0.0649 | 0.0668 | 10.34 | - | 0.0215 | 0.0707 | 3.29 | True |
| 4 | 1024 | 300 | 0.0404 | - | 0.1904 | 0.3154 | 4.71 | - | 0.0860 | 0.1997 | 2.32 | True |
| 4 | 1024 | 3000 | 0.5658 | - | 1.5801 | 1.9505 | 2.79 | - | 0.6001 | 1.5636 | 2.61 | True |
| 4 | 2048 | 30 | 0.0110 | - | 0.1266 | 0.1282 | 11.47 | - | 0.0266 | 0.1321 | 4.96 | True |
| 4 | 2048 | 300 | 0.0676 | - | 0.3739 | 0.6105 | 5.53 | - | 0.1096 | 0.3840 | 3.50 | True |
| 4 | 2048 | 3000 | 0.8028 | - | 3.1579 | 3.8754 | 3.93 | - | 0.8274 | 3.1263 | 3.78 | True |

