# Mixed-precision plan (v2)

Provenance: {"idle_samples": [1, 4, 4], "gpu_before": "4 %, 210 MHz, 405 MHz, 51, 8.50 W, 1728 MiB", "torch": "2.14.0+cu130", "gpu_mid": "97 %, 2775 MHz, 8751 MHz, 74, 142.79 W, 2767 MiB", "gpu_after": "100 %, 2775 MHz, 8751 MHz, 77, 155.89 W, 2829 MiB"}

## Pareto curve, batch 1

Layers are added in order of (measured latency gain) / (single-layer KL cost). Every row is a real run with the real kernels: latency is a CUDA-graph replay, accuracy/AUC/KL are on the 2,000-image Test subset (KL against the FP32 model). Knee marked with ***.

| Step | Layer added | Latency ms | Accuracy | AUC | KL vs fp32 | Prediction flips |
|---|---|---|---|---|---|---|
| 0 | none (all bf16) | 0.433 | 0.8875 | 0.9585 | 1.55e-04 | 0.0035 |
| 1 | trunk.6.0.block.1.0:int8 | 0.435 | 0.8885 | 0.9584 | 1.55e-04 | 0.0035 |
| 2 | trunk.6.2.block.2.fc2:int8 | 0.435 | 0.8875 | 0.9585 | 1.54e-04 | 0.0035 |
| 3 | trunk.6.3.block.2.fc2:int8 | 0.428 | 0.8880 | 0.9583 | 1.54e-04 | 0.0030 |
| 4 | trunk.6.1.block.2.fc2:int8 | 0.440 | 0.8880 | 0.9583 | 1.56e-04 | 0.0030 |
| 5 | trunk.5.2.block.0.0:int8 | 0.440 | 0.8885 | 0.9585 | 1.61e-04 | 0.0035 |
| 6 | trunk.5.1.block.0.0:int8 | 0.429 | 0.8875 | 0.9583 | 1.61e-04 | 0.0035 |
| 7 | trunk.4.2.block.2.fc2:int8 | 0.429 | 0.8880 | 0.9584 | 1.69e-04 | 0.0030 |
| 8 | trunk.7.0.block.2.fc2:int8 | 0.429 | 0.8880 | 0.9587 | 1.72e-04 | 0.0030 |
| 9 | trunk.2.1.block.0.0:int8 | 0.428 | 0.8870 | 0.9584 | 2.04e-04 | 0.0030 |
| 10 | trunk.6.0.block.0.0:int8 | 0.428 | 0.8880 | 0.9577 | 2.14e-04 | 0.0030 |
| 11 | trunk.7.0.block.1.0:int8 | 0.433 | 0.8885 | 0.9581 | 2.39e-04 | 0.0035 |
| 12 | trunk.5.2.block.3.0:fp8 | 0.429 | 0.8870 | 0.9585 | 2.41e-04 | 0.0030 |
| 13 | trunk.1.0.block.1.fc2:int8 | 0.429 | 0.8870 | 0.9586 | 2.43e-04 | 0.0030 |
| 14 | trunk.3.1.block.0.0:int8 | 0.427 | 0.8875 | 0.9583 | 2.68e-04 | 0.0025 |
| 15 | trunk.5.1.block.3.0:fp8 | 0.428 | 0.8875 | 0.9584 | 2.62e-04 | 0.0035 |
| 16 | trunk.5.0.block.2.fc2:int8 | 0.426 | 0.8870 | 0.9584 | 2.57e-04 | 0.0040 |
| 17 | trunk.3.0.block.0.0:int8 | 0.426 | 0.8885 | 0.9577 | 4.65e-04 | 0.0045 |
| 18 | trunk.6.0.block.3.0:int8 | 0.425 | 0.8890 | 0.9581 | 5.21e-04 | 0.0040 |
| 19 | trunk.2.0.block.2.fc1:int8 | 0.425 | 0.8875 | 0.9580 | 7.39e-04 | 0.0045 |
| 20 *** | trunk.2.0.block.2.fc2:int8 | 0.424 | 0.8885 | 0.9579 | 8.01e-04 | 0.0035 |
| 21 | trunk.2.0.block.1.0:int8 | 0.426 | 0.8890 | 0.9578 | 1.23e-03 | 0.0040 |
| 22 | trunk.4.2.block.0.0:int8 | 0.430 | 0.8895 | 0.9571 | 1.25e-03 | 0.0045 |
| 23 | trunk.5.0.block.3.0:int8 | 0.435 | 0.8885 | 0.9573 | 1.38e-03 | 0.0055 |
| 24 | trunk.4.1.block.2.fc2:int8 | 0.435 | 0.8890 | 0.9567 | 1.38e-03 | 0.0060 |
| 25 | trunk.4.0.block.1.0:int8 | 0.438 | 0.8900 | 0.9567 | 1.39e-03 | 0.0060 |
| 26 | trunk.5.0.block.0.0:int8 | 0.437 | 0.8895 | 0.9571 | 1.53e-03 | 0.0065 |
| 27 | trunk.4.0.block.0.0:int8 | 0.437 | 0.8900 | 0.9568 | 1.50e-03 | 0.0050 |
| 28 | trunk.4.1.block.0.0:int8 | 0.432 | 0.8895 | 0.9564 | 1.57e-03 | 0.0065 |
| 29 | trunk.2.0.block.3.0:int8 | 0.431 | 0.8865 | 0.9548 | 2.74e-03 | 0.0085 |
| 30 | trunk.1.0.block.2.0:int8 | 0.430 | 0.8910 | 0.9503 | 1.75e-02 | 0.0240 |

## Pareto curve, batch 64

Layers are added in order of (measured latency gain) / (single-layer KL cost). Every row is a real run with the real kernels: latency is a CUDA-graph replay, accuracy/AUC/KL are on the 2,000-image Test subset (KL against the FP32 model). Knee marked with ***.

| Step | Layer added | Latency ms | Accuracy | AUC | KL vs fp32 | Prediction flips |
|---|---|---|---|---|---|---|
| 0 | none (all bf16) | 14.235 | 0.8880 | 0.9583 | 1.09e-04 | 0.0020 |
| 1 | trunk.8.0:int8 | 14.228 | 0.8880 | 0.9582 | 1.08e-04 | 0.0020 |
| 2 | trunk.6.3.block.0.0:int8 | 14.222 | 0.8885 | 0.9581 | 1.08e-04 | 0.0025 |
| 3 | trunk.6.1.block.0.0:int8 | 14.205 | 0.8880 | 0.9582 | 1.07e-04 | 0.0020 |
| 4 | trunk.7.0.block.3.0:fp8 | 14.145 | 0.8880 | 0.9581 | 1.12e-04 | 0.0020 |
| 5 | trunk.6.2.block.0.0:int8 | 14.148 | 0.8880 | 0.9586 | 1.15e-04 | 0.0020 |
| 6 | trunk.6.2.block.3.0:fp8 | 14.112 | 0.8885 | 0.9581 | 1.15e-04 | 0.0025 |
| 7 | trunk.6.3.block.3.0:fp8 | 14.136 | 0.8885 | 0.9581 | 1.14e-04 | 0.0025 |
| 8 | trunk.7.0.block.0.0:int8 | 14.107 | 0.8885 | 0.9585 | 1.19e-04 | 0.0025 |
| 9 *** | trunk.6.1.block.3.0:fp8 | 14.104 | 0.8885 | 0.9583 | 1.19e-04 | 0.0025 |
| 10 | trunk.5.2.block.0.0:int8 | 14.125 | 0.8880 | 0.9586 | 1.23e-04 | 0.0020 |
| 11 | trunk.5.1.block.0.0:int8 | 14.131 | 0.8890 | 0.9588 | 1.17e-04 | 0.0010 |
| 12 | trunk.6.2.block.2.fc1:int8 | 14.112 | 0.8880 | 0.9585 | 1.23e-04 | 0.0020 |
| 13 | trunk.6.3.block.2.fc1:int8 | 14.119 | 0.8885 | 0.9585 | 1.20e-04 | 0.0015 |
| 14 | trunk.3.0.block.1.0:int8 | 14.484 | 0.8885 | 0.9584 | 1.73e-04 | 0.0045 |
| 15 | trunk.2.0.block.1.0:int8 | 15.295 | 0.8880 | 0.9576 | 5.03e-04 | 0.0040 |
| 16 | trunk.2.1.block.1.0:int8 | 15.636 | 0.8875 | 0.9575 | 6.04e-04 | 0.0045 |
| 17 | trunk.6.0.block.0.0:int8 | 15.646 | 0.8875 | 0.9579 | 6.33e-04 | 0.0045 |
| 18 | trunk.3.0.block.2.fc2:int8 | 15.650 | 0.8880 | 0.9574 | 6.30e-04 | 0.0040 |
| 19 | trunk.5.1.block.3.0:int8 | 15.633 | 0.8870 | 0.9569 | 6.89e-04 | 0.0040 |
| 20 | trunk.6.0.block.3.0:int8 | 15.624 | 0.8865 | 0.9573 | 7.08e-04 | 0.0035 |
| 21 | trunk.4.2.block.3.0:fp8 | 15.612 | 0.8865 | 0.9571 | 7.17e-04 | 0.0035 |
| 22 | trunk.4.0.block.1.0:int8 | 15.662 | 0.8870 | 0.9573 | 9.02e-04 | 0.0040 |
| 23 | trunk.5.2.block.3.0:int8 | 15.661 | 0.8860 | 0.9564 | 1.04e-03 | 0.0040 |
| 24 | trunk.3.1.block.2.fc2:int8 | 15.641 | 0.8855 | 0.9565 | 1.09e-03 | 0.0045 |
| 25 | trunk.5.0.block.3.0:int8 | 15.639 | 0.8865 | 0.9562 | 1.08e-03 | 0.0045 |
| 26 | trunk.7.0.block.2.fc1:int8 | 15.625 | 0.8860 | 0.9573 | 1.08e-03 | 0.0040 |
| 27 | trunk.4.1.block.3.0:int8 | 15.591 | 0.8845 | 0.9567 | 1.12e-03 | 0.0055 |
| 28 | trunk.2.1.block.2.fc2:int8 | 15.604 | 0.8855 | 0.9573 | 1.14e-03 | 0.0045 |
| 29 | trunk.4.1.block.2.fc1:int8 | 15.607 | 0.8850 | 0.9571 | 1.17e-03 | 0.0050 |
| 30 | trunk.4.2.block.2.fc1:int8 | 15.599 | 0.8860 | 0.9570 | 1.13e-03 | 0.0060 |
| 31 | trunk.6.1.block.2.fc1:int8 | 15.621 | 0.8860 | 0.9567 | 1.15e-03 | 0.0060 |
| 32 | trunk.1.0.block.1.fc2:int8 | 15.609 | 0.8860 | 0.9565 | 1.31e-03 | 0.0050 |
| 33 | trunk.5.0.block.2.fc1:int8 | 15.613 | 0.8870 | 0.9573 | 1.24e-03 | 0.0050 |
| 34 | trunk.2.1.block.0.0:int8 | 15.116 | 0.8855 | 0.9567 | 1.37e-03 | 0.0045 |
| 35 | trunk.1.0.block.0.0:int8 | 15.481 | 0.8885 | 0.9580 | 3.59e-03 | 0.0065 |
| 36 | trunk.3.0.block.0.0:int8 | 14.973 | 0.8905 | 0.9567 | 3.71e-03 | 0.0085 |
| 37 | trunk.4.2.block.0.0:int8 | 14.978 | 0.8895 | 0.9565 | 3.57e-03 | 0.0095 |
| 38 | trunk.3.0.block.3.0:int8 | 14.970 | 0.8860 | 0.9560 | 4.58e-03 | 0.0090 |
| 39 | trunk.2.0.block.2.fc2:int8 | 15.002 | 0.8870 | 0.9567 | 4.19e-03 | 0.0090 |
| 40 | trunk.4.0.block.2.fc2:int8 | 15.006 | 0.8860 | 0.9568 | 4.49e-03 | 0.0100 |
| 41 | trunk.4.0.block.3.0:int8 | 14.999 | 0.8860 | 0.9564 | 5.33e-03 | 0.0100 |
| 42 | trunk.5.0.block.0.0:int8 | 15.002 | 0.8865 | 0.9556 | 5.64e-03 | 0.0105 |
| 43 | trunk.2.1.block.3.0:int8 | 15.008 | 0.8860 | 0.9562 | 7.19e-03 | 0.0150 |
| 44 | trunk.4.1.block.0.0:int8 | 15.002 | 0.8850 | 0.9560 | 7.13e-03 | 0.0140 |

## Chosen configuration, full Test set (10,905 images)

### Batch 1: 20 layers quantized

| Variant | Test accuracy | Test AUC | KL vs fp32 | Graph ms | Eager ms | Images/s (20 s) | Idle W | Active W | mJ/image (idle-subtracted) | Weights MB | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock_bf16_folded | 0.8962 | 0.9596 | 2.42e-04 | 0.742 | 4.399 | 1290 | 23.6 | 95.8 | 55.99 | 296.6 | 306 |
| qnet_bf16 | 0.8956 | 0.9588 | 1.53e-04 | 0.441 | 5.457 | 2257 | 21.3 | 90.1 | 30.48 | 305.8 | 321 |
| chosen_mix | 0.8975 | 0.9589 | 7.07e-04 | 0.426 | 6.020 | 2305 | 25.4 | 91.6 | 28.71 | 305.5 | 321 |
| all_int8_eligible | 0.8902 | 0.9432 | 1.85e-02 | 0.501 | 5.953 | 1932 | 19.2 | 83.1 | 33.07 | 305.0 | 577 |

Quantized layers: trunk.6.0.block.1.0=int8, trunk.6.2.block.2.fc2=int8, trunk.6.3.block.2.fc2=int8, trunk.6.1.block.2.fc2=int8, trunk.5.2.block.0.0=int8, trunk.5.1.block.0.0=int8, trunk.4.2.block.2.fc2=int8, trunk.7.0.block.2.fc2=int8, trunk.2.1.block.0.0=int8, trunk.6.0.block.0.0=int8, trunk.7.0.block.1.0=int8, trunk.5.2.block.3.0=fp8, trunk.1.0.block.1.fc2=int8, trunk.3.1.block.0.0=int8, trunk.5.1.block.3.0=fp8, trunk.5.0.block.2.fc2=int8, trunk.3.0.block.0.0=int8, trunk.6.0.block.3.0=int8, trunk.2.0.block.2.fc1=int8, trunk.2.0.block.2.fc2=int8

### Batch 64: 9 layers quantized

| Variant | Test accuracy | Test AUC | KL vs fp32 | Graph ms | Eager ms | Images/s (20 s) | Idle W | Active W | mJ/image (idle-subtracted) | Weights MB | Peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock_bf16_folded | 0.8962 | 0.9596 | 2.42e-04 | 33.425 | 33.818 | 1885 | 28.7 | 127.9 | 52.62 | 297.0 | 857 |
| qnet_bf16 | 0.8960 | 0.9587 | 1.12e-04 | 14.261 | 14.513 | 4429 | 37.0 | 152.6 | 26.10 | 330.5 | 666 |
| chosen_mix | 0.8961 | 0.9589 | 1.26e-04 | 14.116 | 14.339 | 4497 | 28.0 | 153.4 | 27.87 | 329.5 | 665 |
| all_int8_eligible | 0.8902 | 0.9432 | 1.85e-02 | 13.523 | 13.769 | 4685 | 27.3 | 156.1 | 27.51 | 329.8 | 581 |

Quantized layers: trunk.8.0=int8, trunk.6.3.block.0.0=int8, trunk.6.1.block.0.0=int8, trunk.7.0.block.3.0=fp8, trunk.6.2.block.0.0=int8, trunk.6.2.block.3.0=fp8, trunk.6.3.block.3.0=fp8, trunk.7.0.block.0.0=int8, trunk.6.1.block.3.0=fp8

## Model size

| Batch plan | fp32 MB | bf16 MB | chosen mix MB | all-int8 MB | all-W4 group-64 MB |
|---|---|---|---|---|---|
| 1 | 15.96 | 7.98 | 7.25 | 4.20 | 2.23 |
| 64 | 15.96 | 7.98 | 5.69 | 4.20 | 2.23 |
