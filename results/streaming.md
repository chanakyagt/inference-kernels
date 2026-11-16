# Streaming inference: recurrent Mamba step vs window recompute vs attention

GPU state at start: {'util_pct': 0.0, 'sm_clock_mhz': 210.0, 'mem_clock_mhz': 405.0, 'max_sm_clock_mhz': 3105.0, 'temp_c': 51.0, 'power_w': 6.8, 'mem_used_mib': 2626.0}
Idle check: {'samples_pct': [0.0, 0.0, 0.0], 'passed': True, 'waited_s': 25.2}

## Correctness: recurrent stepping equals the parallel scan

Trained 4-layer head, 2 sequences of real test-image features, state carried from zero. `max abs diff` is on the 2 output logits.

| dtype | frames | max abs diff vs parallel fp32 | max abs diff vs parallel same dtype | torch step vs Triton step | max abs logit |
|---|---|---|---|---|---|
| float32 | 30 | 1.07e-06 | nan | 3.58e-07 | 0.44 |
| float32 | 300 | 1.67e-06 | nan | 5.36e-07 | 1.00 |
| bfloat16 | 30 | 1.17e-02 | 1.46e-02 | 0.00e+00 | 0.44 |
| bfloat16 | 300 | 2.38e-02 | 3.91e-02 | 9.77e-04 | 1.00 |

Image trunk (R6 stack, batch 1): 1.19 ms per frame; batch 30: 7.16 ms.

## Per-new-frame head latency (batch 1, bf16 weights, ms; window length = frames in the sliding window)

| Window | Mamba x4 recompute (torch scan loop) | Mamba x4 recompute (fused scan) | Mamba x4 recurrent step (eager) | Mamba x4 recurrent step (CUDA graph) | Mamba x1 recompute fused | Mamba x1 step graph | Attn x1 recompute (SDPA) | Attn x1 recompute (materialized weights) | Attn x1 with KV cache |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 11.148 | 3.124 | 3.115 | 0.203 | 0.496 | 0.063 | 0.476 | 0.609 | 0.473 |
| 300 | 210.591 | 5.715 | 1.545 | 0.214 | 0.509 | 0.069 | 0.422 | 0.442 | 0.486 |
| 3000 | - | 10.854 | 2.122 | 0.237 | 2.852 | 0.076 | 0.827 | 3.516 | 0.480 |

## Memory per new frame (MB; transient = extra peak allocation while computing; state = what must persist)

| Window | Mamba x4 recompute transient | Mamba x4 recurrent state | Mamba x4 step transient | Attn recompute SDPA transient | Attn recompute materialized transient | Attn KV cache (persistent) |
|---|---|---|---|---|---|---|
| 30 | 0.8 | 0.275 | 0.06 | 0.2 | 0.2 | 0.06 |
| 300 | 7.9 | 0.275 | 0.06 | 2.1 | 3.4 | 0.59 |
| 3000 | 73.0 | 0.275 | 0.06 | 20.5 | 173.2 | 5.86 |

## Modeled work per new frame (FLOPs, analytic; head only, trunk excluded)

| Window | Mamba x4 recompute | Mamba x4 step | Attn x1 recompute | Attn x1 KV-cache step |
|---|---|---|---|---|
| 30 | 6.9e+08 | 2.3e+07 | 1.04e+08 | 3.47e+06 |
| 300 | 6.9e+09 | 2.3e+07 | 1.21e+09 | 4.02e+06 |
| 3000 | 6.9e+10 | 2.3e+07 | 2.87e+10 | 9.55e+06 |

## 30-frame sliding window, total per new frame (trunk + head, ms)

| Strategy | ms/frame | frames/s |
|---|---|---|
| (a) recompute everything: trunk on 30 frames + 4-layer Mamba, fused scan | 10.28 | 97 |
| (a2) cache old features: trunk on 1 frame + 4-layer Mamba over 30, fused scan | 4.31 | 232 |
| (b) recurrent step: trunk on 1 frame + eager step | 4.30 | 232 |
| (b2) recurrent step: trunk on 1 frame + CUDA-graph step | 1.39 | 718 |
