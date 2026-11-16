# Trained-detector optimization ladder

Test set: 10,905 face images (5,413 real / 5,492 fake), threshold 0.5. Latency = median of CUDA-event timings of the model forward on a normalized fp32 batch, including the channels_last conversion. J/img = mean GPU board power during a sustained 20 s run divided by img/s; raw and idle-subtracted (net). Peak MB = torch.cuda.max_memory_allocated at batch 64 including weights. An asterisk after img/s marks a sustained run that was not clean (throughput dipped inside the window or idle power read high because another process shared the GPU); latency is the best of 3 trials.

## frozen

| Rung | Description | lat b1 / b8 / b64 (ms) | img/s b1 / b64 | peak MB b64 | J/img b64 raw / net | J/img b1 raw / net | Acc | AUC | flips vs R0s |
|---|---|---|---|---|---|---|---|---|---|
| R0s | fp32 eager NCHW, TF32 off | 6.85 / 6.89 / 48.15 | 138 / 1324 | 742 | 0.0957 / 0.0867 | 0.2452 / 0.1790 | 0.7144 | 0.7817 | - |
| R0 | fp32 eager NCHW | 5.64 / 6.71 / 47.17 | 157* / 1359 | 742 | 0.0852 / 0.0789 | 0.1963 / 0.1507 | 0.7143 | 0.7817 | 4 |
| R1 | + bf16 autocast | 10.79 / 11.40 / 26.39 | 88 / 2402 | 426 | 0.0535 / 0.0483 | 0.3235 / 0.2003 | 0.7144 | 0.7816 | 131 |
| R2 | + channels_last | 14.48 / 16.20 / 26.67 | 56 / 2370 | 465 | 0.0485 / 0.0439 | 0.5009 / 0.3598 | 0.7150 | 0.7816 | 132 |
| R3 | + BN folded into conv | 12.30 / 10.37 / 25.37 | 98 / 2452 | 351 | 0.0493 / 0.0427 | 0.2967 / 0.1912 | 0.7152 | 0.7785 | 190 |
| R4 | + fused dwconv+BN+SiLU (repo, NCHW) | 8.17 / 9.87 / 29.56 | 102 / 2126 | 499 | 0.0649 / 0.0600 | 0.2727 / 0.1679 | 0.7147 | 0.7808 | 133 |
| R4b | + fused dwconv+BN+SiLU (NHWC variant) | 7.69 / 10.16 / 19.12 | 121* / 3345 | 351 | 0.0377 / 0.0335 | 0.2288 / 0.1575 | 0.7142 | 0.7808 | 127 |
| R5 | + fused Triton scan | 8.22 / 9.20 / 18.68 | 112 / 3184* | 434 | 0.0389 / 0.0354 | 0.2495 / 0.1746 | 0.7139 | 0.7809 | 128 |
| R6 | + CUDA graph | 1.17 / 2.19 / 17.83 | 835 / 3612 | 247 | 0.0380 / 0.0353 | 0.0919 / 0.0707 | 0.7140 | 0.7808 | 123 |
| R6b | R6 with bf16 weights pre-cast (no autocast) | 0.78 / 1.80 / 17.31 | 1307 / 3706 | 178 | 0.0371 / 0.0332 | 0.0613 / 0.0531 | 0.7147 | 0.7784 | 195 |
| R7 | torch.compile max-autotune on R2 | 2.41 / 3.13 / 23.12 | 517* / 2945 | 97 | 0.0436 / 0.0384 | 0.1205 / 0.0902 | 0.7148 | 0.7816 | 118 |
| R8 | R6c + INT8 weight-only head Linears | - / - / - | - | - | - | - | 0.7148 | 0.7786 | 190 |
| R9 | R6c + FP8 head Linears | - / - / - | - | - | - | - | 0.7143 | 0.7778 | 222 |

## ft_effnet_b0_v2

| Rung | Description | lat b1 / b8 / b64 (ms) | img/s b1 / b64 | peak MB b64 | J/img b64 raw / net | J/img b1 raw / net | Acc | AUC | flips vs R0s |
|---|---|---|---|---|---|---|---|---|---|
| R0s | fp32 eager NCHW, TF32 off | 3.73 / 5.84 / 62.50 | 235 / 1015 | 904 | 0.1190 / 0.0997 | 0.2374 / 0.1623 | 0.8962 | 0.9587 | - |
| R0 | fp32 eager NCHW | 4.47 / 5.30 / 61.42 | 207 / 1040 | 904 | 0.1118 / 0.0999 | 0.1711 / 0.1172 | 0.8960 | 0.9587 | 2 |
| R1 | + bf16 autocast | 6.30 / 7.42 / 33.45 | 125 / 1917 | 488 | 0.0669 / 0.0574 | 0.2875 / 0.1913 | 0.8962 | 0.9582 | 18 |
| R4b | + fused dwconv+BN+SiLU (NHWC variant) | 5.20 / 6.05 / 22.89 | 85 / 2794 | 391 | 0.0475 / 0.0415 | 0.2086 / 0.1319 | 0.8959 | 0.9590 | 523 |
| R6 | + CUDA graph | 0.73 / 2.13 / 22.53 | 1390 / 2840 | 152 | 0.0488 / 0.0440 | 0.0568 / 0.0508 | 0.8959 | 0.9591 | 523 |
| R6b | R6 with bf16 weights pre-cast (no autocast) | 0.60 / 1.96 / 22.24 | 1680 / 2877 | 144 | 0.0478 / 0.0456 | 0.0514 / 0.0465 | 0.8958 | 0.9595 | 526 |
| R8 | R6c + INT8 weight-only head Linears | 0.95 / 2.86 / 32.34 | 1088 / 1978 | 152 | 0.0652 / 0.0620 | 0.0780 / 0.0717 | 0.8964 | 0.9596 | 524 |
| R9 | R6c + FP8 head Linears | 0.91 / 2.84 / 32.35 | 1101 / 1978 | 152 | 0.0654 / 0.0621 | 0.0778 / 0.0712 | 0.8963 | 0.9596 | 525 |


**Correction (13 Nov 2026):** for `ft_effnet_b0_v2`, the "flips vs R0s" values 523-526 (R4b, R6, R6b, R8, R9) were computed against the R0s predictions of the checkpoint lost in the WSL crash. Against the current R0s reference, R6b has 28 flips (same accuracy 0.8958). See `results/trt/README.md`.
