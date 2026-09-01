# Jester facet evidence

> Generated from immutable benchmark artifacts. `n/m` means not measured; it is never an inferred value.

## Quality and inference cost

| Model | Trained on | Inference | Configuration | General MMLU | Coding semantic | Python HumanEval pass@1 | VRAM allocated | Throughput | p50 latency | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | A40 | RTX 5070 Laptop | `base` | 0.5625 | 0.4375 | 0.4390 | 1.000x | 1.000x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | A40 | RTX 5070 Laptop | `base+coder` | 0.5625 | 0.5104 | 0.3963 | 1.060x | 0.942x | 0.913x | measured |
| Qwen2.5-1.5B-Instruct | A40 | RTX 5070 Laptop | `base+parent+child` | 0.5625 | 0.5521 | 0.3780 | 1.091x | 0.914x | 0.974x | measured |
| Qwen2.5-1.5B-Instruct | A40 | RTX 5070 Laptop | `base+lsr+mi` | 0.5750 | 0.5000 | 0.3780 | 1.091x | 0.659x | 1.286x | measured |
| Qwen2.5-1.5B-Instruct | A40 | NVIDIA A40 | `base` | 0.5625 | 0.5000 | 0.4390 | 1.000x | 1.000x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | A40 | NVIDIA A40 | `base+coder` | 0.5500 | 0.5521 | 0.3841 | 1.060x | 0.935x | 0.943x | measured |
| Qwen2.5-1.5B-Instruct | A40 | NVIDIA A40 | `base+parent+child` | 0.5500 | 0.5833 | 0.3902 | 1.091x | 0.905x | 0.957x | measured |
| Qwen2.5-1.5B-Instruct | A40 | NVIDIA A40 | `base+lsr+mi` | 0.5625 | 0.5521 | 0.3902 | 1.091x | 0.902x | 0.966x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | RTX 5070 Laptop | `base` | n/m | 0.6354 | n/m | 1.000x | 1.000x | 1.000x | partial |
| Gemma-3-1B-IT | RTX 5070 Laptop | RTX 5070 Laptop | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | RTX 5070 Laptop | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | RTX 5070 Laptop | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | NVIDIA A40 | `base` | n/m | 0.6250 | n/m | 1.000x | 1.000x | 1.000x | partial |
| Gemma-3-1B-IT | RTX 5070 Laptop | NVIDIA A40 | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | NVIDIA A40 | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | RTX 5070 Laptop | `base` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | RTX 5070 Laptop | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | RTX 5070 Laptop | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | RTX 5070 Laptop | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | NVIDIA A40 | `base` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | NVIDIA A40 | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | NVIDIA A40 | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |

## Paired quality deltas versus base

| Model | Inference | Configuration | Coding semantic delta | General MMLU delta [bootstrap 95%] | MMLU improved/regressed | HumanEval delta [bootstrap 95%] | HumanEval improved/regressed |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+coder` | 0.0729 | +0.0000 [-0.0500, +0.0500] | 2/2 | -0.0427 [-0.0854, +0.0000] | 4/11 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+parent+child` | 0.1146 | +0.0000 [-0.0500, +0.0500] | 2/2 | -0.0610 [-0.1098, -0.0122] | 4/14 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 0.0625 | +0.0125 [+0.0000, +0.0375] | 1/0 | -0.0610 [-0.1098, -0.0122] | 4/14 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+coder` | 0.0521 | -0.0125 [-0.0625, +0.0375] | 2/3 | -0.0549 [-0.1098, +0.0000] | 7/16 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+parent+child` | 0.0833 | -0.0125 [-0.0625, +0.0375] | 2/3 | -0.0488 [-0.1037, +0.0061] | 7/15 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+lsr+mi` | 0.0521 | +0.0000 [-0.0375, +0.0375] | 1/1 | -0.0488 [-0.1037, +0.0061] | 7/15 |

## Absolute inference performance and allocator memory

| Model | Inference | Configuration | Allocated GiB | Reserved GiB | Design tok/s | HumanEval tok/s | HumanEval p50 ms | HumanEval p95 ms | State wall s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base` | 2.884 | 3.121 | 42.59 | 41.87 | 1375.77 | 3806.64 | 519.25 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+coder` | 3.059 | 3.311 | 39.12 | 39.45 | 1256.43 | 3212.13 | 523.27 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+parent+child` | 3.146 | 3.414 | 38.62 | 38.29 | 1339.69 | 3387.27 | 530.04 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 3.146 | 3.412 | 15.15 | 27.59 | 1769.62 | 6599.86 | 1078.37 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base` | 2.883 | 2.963 | 44.24 | 45.51 | 1280.76 | 3335.61 | 493.31 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+coder` | 3.058 | 3.125 | 42.60 | 42.56 | 1208.24 | 3158.86 | 501.82 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+parent+child` | 3.145 | 3.225 | 40.55 | 41.20 | 1226.30 | 4071.27 | 497.50 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+lsr+mi` | 3.145 | 3.225 | 42.21 | 41.03 | 1237.23 | 4056.15 | 533.13 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base` | 1.920 | 1.973 | 35.22 | 34.98 | 5464.85 | 11230.35 | 1478.67 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | NVIDIA A40 | `base` | 1.870 | 1.924 | 28.11 | 29.17 | 6512.09 | 13322.42 | 1802.60 |
| Gemma-3-1B-IT | NVIDIA A40 | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | NVIDIA A40 | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+coder` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+lsr+mi` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |

## LSR routing

| Model | Inference | Expected decision | Selection recall | Expected abstention | p50 route latency (ms) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | 0.9787 | 0.9787 | n/m | 153.52 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | 0.8433 | 0.9787 | 0.5250 | 110.65 |

## Controlled training

| Run | Hardware | Workers | Rounds | Supervised tokens | Last loss | Compute seconds | tok/s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | not measured |

## Limitations

- All quality runs use one deterministic seed; they do not establish between-seed variance.
- The design suite contains 24 cases and is a project golden, not a public leaderboard.
- The MMLU suite is an 80-record balanced subset and not the official five-shot protocol.
- Only one academic A40 was available; the other devices were occupied and were not touched.
- A measured quality regression remains a valid result and is not replaced by an expectation.
