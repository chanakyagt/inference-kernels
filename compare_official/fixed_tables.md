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

