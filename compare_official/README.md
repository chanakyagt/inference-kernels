# Kernel race: our CUDA scan against the official Mamba kernel

This folder holds the script that races our selective-scan kernel (the core recurrence of a Mamba layer) against the official `mamba-ssm` CUDA kernel and against the kernel shipped in vLLM. The raw data is `../results/official_compare.json`; the rendered tables are `../results/official_compare.md`.

Setup: RTX 4060 Ti (288 GB/s memory bandwidth), torch 2.14.0+cu130, vLLM 0.33.0, official `mamba-ssm` commit e9594ce. Each time is the median of 200 runs, measured as a CUDA graph of 5 back-to-back calls divided by 5, so Python launch cost is excluded. Every method gets the same inputs; the recurrence is the same for all (no `z` gate, no `delta_softplus`). Every output is checked against one fp32 reference. Source: `meta` and `scan` in the JSON.

## Headline: fp32 (float32)

fp32 is the number to quote. Speedup = official time / our time, so above 1 means ours is faster.

- ours_cuda_tuned: the best of a per-shape sweep of CUDA launch settings (channels per block, state split). The sweep chose (16 channels, split 2) in 20 of the 24 rows (`cuda_best_cfg` in the JSON).
- ours_cuda_default: one fixed setting (32 channels, split 2) for every shape. This is the fairer number, because the official kernel has no settings to tune.
- official: the `mamba-ssm` selective-scan CUDA kernel, timed in its own native memory layout.
- vLLM: its scan output is bit-identical to the official kernel in all 24 rows (`max_abs_diff_vs_official.vllm_scan` is exactly 0.0), and the times are close, so it is the same kernel code. It is shown as one extra column and is not an independent second baseline.

| B | E | T | ours_cuda_tuned (ms) | ours_cuda_default (ms) | official (ms) | vLLM (ms) | speedup tuned | speedup default |
|---|---|---|---|---|---|---|---|---|
| 1 | 1024 | 30 | 0.0053 | 0.0068 | 0.0206 | 0.0214 | 3.88 | 3.05 |
| 1 | 1024 | 300 | 0.0307 | 0.0463 | 0.0530 | 0.0776 | 1.73 | 1.15 |
| 1 | 1024 | 3000 | 0.4588 | 0.7796 | 0.4292 | 0.4735 | 0.94 | 0.55 |
| 1 | 2048 | 30 | 0.0049 | 0.0061 | 0.0344 | 0.0362 | 7.00 | 5.60 |
| 1 | 2048 | 300 | 0.0319 | 0.0465 | 0.1019 | 0.1532 | 3.20 | 2.19 |
| 1 | 2048 | 3000 | 0.5218 | 0.8196 | 0.8710 | 0.9482 | 1.67 | 1.06 |
| 4 | 1024 | 30 | 0.0061 | 0.0068 | 0.0657 | 0.0680 | 10.72 | 9.72 |
| 4 | 1024 | 300 | 0.0387 | 0.0498 | 0.1993 | 0.3047 | 5.15 | 4.00 |
| 4 | 1024 | 3000 | 0.6703 | 0.9103 | 1.7395 | 1.9017 | 2.60 | 1.91 |
| 4 | 2048 | 30 | 0.0086 | 0.0086 | 0.1274 | 0.1294 | 14.85 | 14.81 |
| 4 | 2048 | 300 | 0.0627 | 0.0679 | 0.3938 | 0.5937 | 6.28 | 5.80 |
| 4 | 2048 | 3000 | 1.1536 | 1.2614 | 3.4823 | 3.8222 | 3.02 | 2.76 |

B = batch, E = channels, T = time steps.

Reading the table:
- Tuned: faster on 11 of 12 shapes, from 1.67x to 14.85x. It loses on one shape, B1 E1024 T3000, where the official kernel is faster (0.94x).
- Default (32, 2 fixed setting): faster on 11 of 12 shapes, from 1.06x to 14.81x. On the losing shape it is 0.55x. A better fixed setting, (16, 2), is in the rerun section below.
- Why it wins and loses: our kernel gives each channel to a thread and walks time one step after another, which suits short sequences and many channels. The official kernel splits time across threads in a block, which suits one long sequence. This explanation comes from the design of the two kernels, not from hardware counters (Nsight Compute is blocked on this machine).
- Errors against the fp32 reference are about 1e-6 for all methods (`err_vs_fp32_ref` in the JSON).

## Fixed-config rerun, fp32 and bf16, with an idle check before every row (12 November 2026)

This replaces the earlier disturbed bf16 rows. Same shapes, one CUDA setting for every shape: 16 channels per block, state split 2 (`--fixed-cfg 16,2`; `cpb`, `s` in `bench_official.py`). The per-shape sweep chose this setting in 20 of the 24 earlier rows (`cuda_best_cfg`). Before each row the script waits for the GPU to be idle (utilisation at most 5% for three samples 5 s apart; met for 24 of 24 rows, `idle_gate_met`), then runs 1.5 s of matrix multiplication so the GPU clocks are up before timing. Without that warm-up, the first attempt after a 15 s idle wait measured at low clocks (0.154 ms for a 0.031 ms kernel) and was discarded. Raw data: `../results/official_compare_fixed.json`, merged into `../results/official_compare.json` as `scan_fixed` by `merge_fixed.py`; tables also in `../results/official_compare.md`.

**Headline with fixed setting (warm, CUDA graph):** fp32 faster on 11 of 12 shapes, 1.63x to 11.46x; one loss, B1 E1024 T3000 at 0.94x (same as tuned). bf16 faster on all 12 shapes, 1.33x to 11.47x. The earlier fp32 "default" column used (32, 2) and gave 0.55x on the losing shape; (16, 2) is a better fixed setting. Fixed against tuned (fp32): the tuned times are 0% to 23% lower, the largest gap at B4 E2048 T30 (0.0086 against 0.0111 ms, 14.85x against 11.46x).

**Cold L2.** Method: a 64 MB buffer is written before each call, and the call is timed with CUDA events queued behind that write (so the launch latency is hidden). Events add a floor: the same event method without the flush reads 7.2 microseconds for our kernel and 20.5 for the official one at B1 E1024 T30 (fp32), against 13.5 and 25.6 with the flush. So the cold-L2 penalty at the smallest shape is about 6 microseconds for both kernels. Cold speedups are smaller than warm ones: 1.89x at B1 E1024 T30 (warm 3.58x). They flip to losses in three rows: fp32 B1 E1024 T3000 (0.91x, warm 0.94x), and bf16 B1 E1024 T300 (0.89x, warm 1.58x) and T3000 (0.87x, warm 1.33x). Everywhere else we stay ahead cold (1.03x to 4.96x). This shows that part of the warm bf16 advantage at batch 1 depends on the inputs staying in L2.

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


## Footnote on bandwidth above 288 GB/s

The `graph_bw_GBps` fields in the JSON and the "GB/s" columns in `../results/official_compare.md` are the analytic minimum bytes moved divided by the measured time. For our scan kernel they reach 475.3 GB/s, and for single-token steps they exceed 1,300 GB/s, both above the card's 288 GB/s memory bandwidth. This is not a bug: the timing repeats the same inputs, so the data stays in the 32 MB L2 cache (the large on-chip cache in front of main memory) and is not read from DRAM each time. Both kernels get this benefit, so the ratios stand, but the absolute figures are not DRAM bandwidth. The cold-L2 variant above removes this effect for the scan rows.

## Single-token state update (step core)

The comparison is our Triton step kernel against vLLM's Triton step kernel (fp32, 6 shapes). vLLM's time divided by ours is 0.89x to 1.03x; vLLM is slightly faster in 5 of 6 shapes. Source: `step_core` in the JSON.

The official Triton single-token kernel raised `TypeError: Value after * must be an iterable, not int` on all 6 shapes, so there is no official data point for this comparison. Those 6 errors are the only real errors in the run (see below).

## Whole Mamba layer step (our fused layer)

Our fused layer (two Triton launches) against a composed baseline (PyTorch operations plus the official step kernel). Ratio = composed time / our time; below 1 means our fused layer is slower. Source: `step_fused` in the JSON.

| B | E | ours (ms) | composed + official (ms) | ratio |
|---|---|---|---|---|
| 1 | 1024 | 0.0186 | 0.0244 | 1.31 |
| 8 | 1024 | 0.0648 | 0.0493 | 0.76 |
| 64 | 1024 | 0.4727 | 0.0748 | 0.16 |
| 1 | 2048 | 0.1656 | 0.0984 | 0.59 |
| 8 | 2048 | 0.3083 | 0.2168 | 0.70 |
| 64 | 2048 | 2.2715 | 0.3081 | 0.14 |

Our fused layer wins on only 1 of 6 shapes (B1 E1024, 1.31x) and loses on the other 5; at batch 64 it is 6 to 7 times slower. This unit is not an improvement over the composed baseline at larger batch.

## Errors in the JSON

The JSON `errors` list has 558 entries. 552 are `AttributeError: module 'selective_scan_cuda' has no attribute 'scan'` from the first run, caused by a name clash with the official extension. The scan section was then rerun correctly and merged (`--merge`), but the old errors were carried over. The remaining 6 are the official step-kernel TypeErrors above. The stale entries stay in the JSON for the record and are left out of `../results/official_compare.md`.

## How the official kernel was built

`BUILD_OFFICIAL.md` in this folder records the build command and compiler flags: optimised (-O3, `--use_fast_math`, sm_89). The build log was found in the author's WSL home (`~/sota/build.log`), not in this repo.

## Environment variables

- `MAMBA_OFFICIAL_REPO`: path to the official `mamba-ssm` checkout (default `~/sota/mamba_official`).
- `MAMBA_OFFICIAL_BUILD`: path to the built `selective_scan_cuda` extension (default `~/sota/mamba_build`).

## Run

`python compare_official/bench_official.py --wait-idle 600` writes `results/official_compare.json` and `.md`. It needs a GPU, the installs in `../requirements.txt`, vLLM, and the official build above.
