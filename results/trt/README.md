# TensorRT baseline for the v2 detector (13 Nov 2026, RTX 4060 Ti, WSL2)

Model: `_artifacts/bb_runs/ft_effnet_b0_v2` (EfficientNet-B0 + linear head, 256 px). TensorRT 11.4.0.106 (strongly typed; precision set by the exported ONNX dtype; BatchNorm folded into conv before export, 49 layers). Same harness as `results/trained_ladder.md`: latency = best of 3 trials of the median CUDA-event time, input copy included, CUDA graph on every row; sustained = 20 s, idle power subtracted; idle gate passed (3/5/5 % utilisation). All sustained runs stable with clean idle. R6b re-run in the same session. Script: `trained/trt_bench.py`; raw data: `trt_results.json`.

| Version | Acc / AUC (10,905 Test) | flips vs fp32 R0s | lat b1 / b8 / b64 (ms) | img/s b1 / b64 | J/img net b1 / b64 |
|---|---|---|---|---|---|
| R6b (own path, bf16), same session | 0.8958 / 0.9595 (ladder) | 526 (ladder) | 0.627 / 1.842 / 22.34 | 1,506 / 2,759 | 0.0465 / 0.0425 |
| TensorRT fp32 (TF32) | 0.8961 / 0.9586 | 1 | 0.650 / 3.361 / 42.26 | 1,533 / 1,516 | 0.0559 / 0.0784 |
| TensorRT bf16 | 0.8959 / 0.9593 | 31 | 0.565 / 1.851 / 22.70 | 1,816 / 2,781 | 0.0411 / 0.0421 |
| TensorRT fp16 | 0.8961 / 0.9588 | 1 | 0.413 / 1.123 / 15.06 | 2,407 / 4,233 | 0.0326 / 0.0281 |

Reading:
- Same precision (bf16): the hand-optimised path is within about 2-11% of TensorRT (TRT 1.11x faster at b1, 0.98x at b64; energy equal within noise at b64).
- TensorRT fp16 vs R6b: 1.52x lower latency at b1, 1.48x at b64, 1.53x sustained img/s at b64, 1.5x less energy per image, accuracy unchanged (1 flip vs fp32; R6b has 526).
- fp16 vs bf16 inside TensorRT differs 1.37-1.65x although both are 2-byte formats. Cause NOT measured (candidates: TensorRT kernel coverage for fp16 vs bf16, or fp16-accumulate tensor-core rate on GeForce). Do not state a cause until profiled.
- INT8 not run: TensorRT 11 has no implicit INT8 calibration; it needs explicit Q/DQ quantisation (NVIDIA ModelOpt).

## fp16 in the own PyTorch path (same day, separate session; `trained/fp16_check.py`, `fp16_check.json`; idle gate passed, all sustained runs stable + clean idle)
| Version | Acc / AUC | flips vs R0s | lat b1 / b8 / b64 (ms) | img/s b1 / b64 | J/img net b1 / b64 |
|---|---|---|---|---|---|
| R6b (Triton dwconv, bf16) | 0.8958 / 0.9595 | 28 | 0.620 / 1.991 / 22.38 | 1,622 / 2,849 | 0.0440 / 0.0432 |
| R6f (Triton dwconv, fp16) | 0.8961 / 0.9588 | 1 | 0.579 / 1.890 / 22.07 | 1,767 / 2,901 | 0.0427 / 0.0442 |
| R6d (no Triton dwconv, bf16) | 0.8963 / 0.9596 | 27 | 0.785 / 2.971 / 32.86 | 1,232 / 1,934 | 0.0645 / 0.0616 |
| R6g (no Triton dwconv, fp16) | 0.8962 / 0.9588 | 2 | 0.746 / 2.988 / 33.45 | 1,250 / 1,897 | 0.0639 / 0.0606 |

Reading:
- fp16 instead of bf16 in the own path: 1.07x at b1, 1.01x at b64, energy unchanged. So TensorRT fp16's 1.40x (b1: 0.413 vs 0.579 ms) and 1.47x (b64: 15.06 vs 22.07 ms) over the own fp16 path comes from TensorRT's fp16 kernels/fusions, not from the number format.
- The own Triton NHWC depthwise kernel is worth 1.27x at b1 and 1.47x at b64 (R6b vs R6d, same session).
- Next step to locate the gap: per-layer profile of the TensorRT fp16 engine vs the own fp16 path.

## CORRECTION: ladder "flips vs R0s" for the v2 model
`results/trained_ladder.md` lists 523-526 flips for R4b/R6/R6b/R8/R9 (v2). Today R6b gives the same accuracy (0.89583) but 28 flips against `probs_ft_effnet_b0_v2_R0s_10905.npy`. The ladder rows were computed before R0s was re-measured on the 3090-retrain checkpoint, so they compared against the lost checkpoint's predictions. Correct figure: ~27-28 flips for the bf16 paths, 1-2 for fp16. Do not quote 523-526.

## Layer profile, batch 64, fp16 (`trained/profile_layers.py`, `layer_profile.json`; idle gate passed)
Own = R6f eager with CUDA events per MBConv sub-module (sum 21.7 ms; graph run 22.1 ms). TRT = fp16 engine, IProfiler per layer (sum 16.0 ms), layers mapped to PyTorch modules via ONNX name_scopes.

| Layer kind (all stages) | Own ms | TensorRT ms | own / TRT |
|---|---|---|---|
| expand 1x1 conv (+SiLU) | 9.30 | 2.62 | 3.54 |
| depthwise conv (+SiLU) | 4.25 | 6.28 | 0.68 (own faster) |
| squeeze-excite | 3.79 | 3.51 | 1.08 |
| project 1x1 conv | 2.46 | 2.03 | 1.21 |
| stem | 1.42 | 1.49 | 0.96 |
| head conv + pool + fc + residual | 0.48 | 0.09 | - |

Reading:
- The whole 1.47x gap is the expand 1x1 convolutions: 9.30 vs 2.62 ms (stage 2 alone 5.61 vs 1.26 ms). TensorRT runs them as one fp16 implicit-GEMM kernel with the SiLU fused in (`...fprop_implicit_gemm_f16... + PWN(Sigmoid, silu)`).
- The own Triton NHWC depthwise kernel beats TensorRT's depthwise kernels by 1.48x in total (4.25 vs 6.28 ms; faster in 6 of 7 stages).
- Estimate, not measured: own path with TensorRT-speed expand convs ~15.0 ms; TensorRT with the own depthwise kernel ~14.0 ms (~1.14x faster than TensorRT fp16). Needs building (fused 1x1 GEMM+SiLU kernel, or a TensorRT plugin) before it can be claimed.

## Co-design follow-up (13 Nov): two design options + one critical review
Prototype files copied to `fusion_prototype/` (GPU kernels) and `arch_model/` (accelerator model).

**B, fused kernels on the same GPU (keep; build before claiming).** Cause of the 3.5x expand gap, measured: the cuDNN GEMM is fine (0.905 ms, TensorRT's fused kernel 0.922 ms); bias add and SiLU each add a full extra read+write of the 6x-expanded tensor at DRAM speed (1.649 + 1.613 ms on stage 2 block 0). Triton GEMM with bias+SiLU fused in the epilogue: 15-layer expand chain in a CUDA graph 2.78 ms (independent re-run 2.87) vs 9.53 ms (re-run 10.0) today vs TensorRT ~2.62 ms (per-layer profiler, slightly inflated: layers sum to 16.0 vs 15.06 end to end); torch.compile reaches only 6.70 ms (max-autotune 7.04). Project chain with SE-scale prologue + residual epilogue 5.44 -> 2.44 ms. Whole-model projection ~12.6 ms vs TensorRT 15.06 at b64 is NOT measured and may double-count SE/residual time; expand fusion alone ~ parity. Not novel (cuBLASLt/CUTLASS epilogues, DNNFusion, TVM/Ansor): present as profiling-driven engineering. Per-layer eager numbers in pw_b64.json are L2-inflated: quote chain or whole-model numbers only.
To claim: build whole model with fused expand + project kernels; same trt_bench harness (graph, input copy, best of 3, 20 s sustained, idle subtracted); accuracy/AUC/flips on all 10,905; b1 and b64; TensorRT re-run same session at builder optimisation level 5.

**A, model-specific accelerator (drop the ratio).** Modelled fused-MBConv int8 engine: 0.755 mJ/image vs measured 32.6 mJ GPU -> "43x" is core-model vs measured board power (GDDR6, VRM, PCIe), int8 vs fp16 with no GPU int8 baseline and 1.07 pt accuracy loss; range points hand-picked; 40x over a tensor-core GPU would be an extraordinary claim. Safe use: one qualitative clause only ("a first-order analytical model of a fused-MBConv int8 dataflow ... testing it with Timeloop/Accelergy and RTL is a question for graduate school").
