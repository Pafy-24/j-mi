# Jester facet evidence

> Generated from immutable benchmark artifacts. `n/m` means not measured; it is never an inferred value.

## Parameter cost of the composed facets

| Model | Base parameters | `coder` parameters | `coder.python` parameters | Total inserted | Inserted / base | Composed parameters |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 1,543,714,304 | 93,595,648 | 46,797,824 | 140,393,472 | 9.09% | 1,684,107,776 |
| Gemma-3-1B-IT | 999,885,952 | 53,684,224 | 26,842,112 | 80,526,336 | 8.05% | 1,080,412,288 |
| Llama-3.2-1B-Instruct | 1,235,814,400 | 121,643,008 | 60,821,504 | 182,464,512 | 14.76% | 1,418,278,912 |

## Quality and inference cost

| Model | Trained on | Training rounds | Inference | Configuration | General MMLU [Wilson 95%] | Coding semantic | Python HumanEval pass@1 [Wilson 95%] | VRAM allocated | Throughput | p50 latency | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B-Instruct | A40 | 8 | RTX 5070 Laptop | `base` | 0.5625 [0.4534, 0.6659] | 0.4375 | 0.4390 [0.3653, 0.5155] | 1.000x | 1.000x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | RTX 5070 Laptop | `base+coder` | 0.5625 [0.4534, 0.6659] | 0.5104 | 0.3963 [0.3247, 0.4728] | 1.060x | 0.942x | 0.913x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | RTX 5070 Laptop | `base+parent+child` | 0.5625 [0.4534, 0.6659] | 0.5521 | 0.3780 [0.3074, 0.4543] | 1.091x | 0.914x | 0.974x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | RTX 5070 Laptop | `base+lsr+mi` | 0.5750 [0.4657, 0.6774] | 0.5000 | 0.3780 [0.3074, 0.4543] | 1.091x | 0.659x | 1.286x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | NVIDIA A40 | `base` | 0.5625 [0.4534, 0.6659] | 0.5000 | 0.4390 [0.3653, 0.5155] | 1.000x | 1.000x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | NVIDIA A40 | `base+coder` | 0.5500 [0.4412, 0.6542] | 0.5521 | 0.3841 [0.3132, 0.4604] | 1.060x | 0.935x | 0.943x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | NVIDIA A40 | `base+parent+child` | 0.5500 [0.4412, 0.6542] | 0.5833 | 0.3902 [0.3189, 0.4666] | 1.091x | 0.905x | 0.957x | measured |
| Qwen2.5-1.5B-Instruct | A40 | 8 | NVIDIA A40 | `base+lsr+mi` | 0.5625 [0.4534, 0.6659] | 0.5521 | 0.3902 [0.3189, 0.4666] | 1.091x | 0.902x | 0.966x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base` | 0.2000 [0.1270, 0.3005] | 0.6354 | 0.3841 [0.3132, 0.4604] | 1.000x | 1.000x | 1.000x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+coder` | 0.0000 [0.0000, 0.0458] | 0.0312 | 0.0000 [0.0000, 0.0229] | 1.054x | 0.619x | 2.186x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+parent+child` | 0.0000 [0.0000, 0.0458] | 0.0208 | 0.0000 [0.0000, 0.0229] | 1.081x | 0.048x | 4.023x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+lsr+mi` | 0.1500 [0.0879, 0.2441] | 0.0312 | 0.0061 [0.0011, 0.0337] | 1.081x | 0.670x | 2.722x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | NVIDIA A40 | `base` | 0.2000 [0.1270, 0.3005] | 0.6250 | 0.3902 [0.3189, 0.4666] | 1.000x | 1.000x | 1.000x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+coder` | 0.0000 [0.0000, 0.0458] | 0.0312 | 0.0000 [0.0000, 0.0229] | 1.053x | 0.912x | 2.224x | measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+lsr+mi` | 0.1500 [0.0879, 0.2441] | 0.0312 | 0.0061 [0.0011, 0.0337] | 1.080x | 0.901x | 2.232x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base` | 0.4250 [0.3226, 0.5343] | 0.4896 | 0.3354 [0.2676, 0.4107] | 1.000x | 1.000x | 1.000x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+coder` | 0.1500 [0.0879, 0.2441] | 0.4167 | 0.3171 [0.2507, 0.3918] | 1.098x | 0.924x | 1.071x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+parent+child` | 0.1625 [0.0975, 0.2584] | 0.4062 | 0.2927 [0.2284, 0.3664] | 1.147x | 0.908x | 1.048x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | RTX 5070 Laptop | `base+lsr+mi` | 0.2750 [0.1892, 0.3814] | 0.3854 | 0.2927 [0.2284, 0.3664] | 1.147x | 0.912x | 1.025x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | NVIDIA A40 | `base` | 0.4500 [0.3458, 0.5588] | 0.4688 | 0.3171 [0.2507, 0.3918] | 1.000x | 1.000x | 1.000x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+coder` | 0.1500 [0.0879, 0.2441] | 0.4167 | 0.3171 [0.2507, 0.3918] | 1.098x | 0.900x | 1.091x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+parent+child` | 0.1625 [0.0975, 0.2584] | 0.4062 | 0.2988 [0.2340, 0.3728] | 1.147x | 0.863x | 1.084x | measured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 8 | NVIDIA A40 | `base+lsr+mi` | 0.2625 [0.1786, 0.3682] | 0.4167 | 0.2988 [0.2340, 0.3728] | 1.147x | 0.864x | 1.068x | measured |

## Paired quality deltas versus base

| Model | Inference | Configuration | Coding semantic delta | General MMLU delta [bootstrap 95%] | MMLU improved/regressed | HumanEval delta [bootstrap 95%] | HumanEval improved/regressed |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+coder` | 0.0729 | +0.0000 [-0.0500, +0.0500] | 2/2 | -0.0427 [-0.0854, +0.0000] | 4/11 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+parent+child` | 0.1146 | +0.0000 [-0.0500, +0.0500] | 2/2 | -0.0610 [-0.1098, -0.0122] | 4/14 |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 0.0625 | +0.0125 [+0.0000, +0.0375] | 1/0 | -0.0610 [-0.1098, -0.0122] | 4/14 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+coder` | 0.0521 | -0.0125 [-0.0625, +0.0375] | 2/3 | -0.0549 [-0.1098, +0.0000] | 7/16 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+parent+child` | 0.0833 | -0.0125 [-0.0625, +0.0375] | 2/3 | -0.0488 [-0.1037, +0.0061] | 7/15 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+lsr+mi` | 0.0521 | +0.0000 [-0.0375, +0.0375] | 1/1 | -0.0488 [-0.1037, +0.0061] | 7/15 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+coder` | -0.6042 | -0.2000 [-0.2875, -0.1125] | 0/16 | -0.3841 [-0.4573, -0.3110] | 0/63 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+parent+child` | -0.6146 | -0.2000 [-0.2875, -0.1125] | 0/16 | -0.3841 [-0.4573, -0.3110] | 0/63 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+lsr+mi` | -0.6042 | -0.0500 [-0.1000, -0.0125] | 0/4 | -0.3780 [-0.4512, -0.3049] | 0/62 |
| Gemma-3-1B-IT | NVIDIA A40 | `base+coder` | -0.5938 | -0.2000 [-0.2875, -0.1125] | 0/16 | -0.3902 [-0.4634, -0.3171] | 0/64 |
| Gemma-3-1B-IT | NVIDIA A40 | `base+lsr+mi` | -0.5938 | -0.0500 [-0.1000, -0.0125] | 0/4 | -0.3841 [-0.4634, -0.3110] | 0/63 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+coder` | -0.0729 | -0.2750 [-0.4000, -0.1500] | 5/27 | -0.0183 [-0.0732, +0.0366] | 9/12 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+parent+child` | -0.0833 | -0.2625 [-0.3875, -0.1375] | 6/27 | -0.0427 [-0.1037, +0.0183] | 9/16 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | -0.1042 | -0.1500 [-0.2375, -0.0750] | 0/12 | -0.0427 [-0.1037, +0.0183] | 9/16 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+coder` | -0.0521 | -0.3000 [-0.4250, -0.1750] | 5/29 | +0.0000 [-0.0549, +0.0549] | 10/10 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+parent+child` | -0.0625 | -0.2875 [-0.4125, -0.1500] | 6/29 | -0.0183 [-0.0793, +0.0427] | 11/14 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+lsr+mi` | -0.0521 | -0.1875 [-0.2750, -0.1125] | 0/15 | -0.0183 [-0.0793, +0.0427] | 11/14 |

## Cross-hardware A40 versus RTX

> Quality columns are A40 minus RTX. Performance and allocator columns are A40 divided by RTX for the same model, artifacts and state.

| Model | Configuration | General delta | Coding delta | HumanEval delta | HE throughput ratio | VRAM allocated ratio | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Gemma-3-1B-IT | `base` | 0.0000 | -0.0104 | 0.0061 | 0.834x | 0.974x | measured |
| Gemma-3-1B-IT | `base+coder` | 0.0000 | 0.0000 | 0.0000 | 1.228x | 0.974x | measured |
| Gemma-3-1B-IT | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | not-measured |
| Gemma-3-1B-IT | `base+lsr+mi` | 0.0000 | 0.0000 | 0.0000 | 1.121x | 0.973x | measured |
| Llama-3.2-1B-Instruct | `base` | 0.0250 | -0.0208 | -0.0183 | 1.281x | 1.000x | measured |
| Llama-3.2-1B-Instruct | `base+coder` | 0.0000 | 0.0000 | 0.0000 | 1.247x | 1.000x | measured |
| Llama-3.2-1B-Instruct | `base+parent+child` | 0.0000 | 0.0000 | 0.0061 | 1.216x | 1.000x | measured |
| Llama-3.2-1B-Instruct | `base+lsr+mi` | -0.0125 | 0.0312 | 0.0061 | 1.214x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | `base` | 0.0000 | 0.0625 | 0.0000 | 1.087x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | `base+coder` | -0.0125 | 0.0417 | -0.0122 | 1.079x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | `base+parent+child` | -0.0125 | 0.0312 | 0.0122 | 1.076x | 1.000x | measured |
| Qwen2.5-1.5B-Instruct | `base+lsr+mi` | -0.0125 | 0.0521 | 0.0122 | 1.487x | 1.000x | measured |

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
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+coder` | 2.024 | 2.076 | 32.66 | 21.65 | 11946.02 | 47879.68 | 3426.44 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+parent+child` | 2.075 | 2.131 | 17.42 | 1.69 | 21987.10 | 45782.30 | 36047.48 |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+lsr+mi` | 2.075 | 2.131 | 21.97 | 23.44 | 14876.63 | 25929.37 | 3313.16 |
| Gemma-3-1B-IT | NVIDIA A40 | `base` | 1.870 | 1.924 | 28.11 | 29.17 | 6512.09 | 13322.42 | 1802.60 |
| Gemma-3-1B-IT | NVIDIA A40 | `base+coder` | 1.970 | 2.027 | 26.39 | 26.59 | 14484.56 | 14963.69 | 3043.34 |
| Gemma-3-1B-IT | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | n/m | n/m | n/m |
| Gemma-3-1B-IT | NVIDIA A40 | `base+lsr+mi` | 2.020 | 2.066 | 26.43 | 26.28 | 14531.85 | 14952.66 | 2937.01 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base` | 2.311 | 2.479 | 58.99 | 59.75 | 965.59 | 2184.15 | 447.03 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+coder` | 2.537 | 2.670 | 54.28 | 55.18 | 1034.30 | 2221.06 | 438.87 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+parent+child` | 2.651 | 2.785 | 51.82 | 54.25 | 1012.16 | 2309.84 | 470.96 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 2.651 | 2.785 | 56.38 | 54.47 | 989.48 | 2277.18 | 457.36 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base` | 2.310 | 2.385 | 75.12 | 76.50 | 758.82 | 1742.30 | 366.87 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+coder` | 2.536 | 2.621 | 68.20 | 68.82 | 827.78 | 1849.04 | 340.71 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+parent+child` | 2.650 | 2.740 | 64.95 | 65.99 | 822.83 | 1920.54 | 418.82 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+lsr+mi` | 2.650 | 2.740 | 67.71 | 66.12 | 810.06 | 1918.65 | 362.01 |

## RM process-tree resources

> RSS and CPU totals are conservative sums of per-instance statistics. Peak values are upper bounds because instance maxima may occur in different samples.
> Historical all-zero CPU series are rendered `n/m` because they came from a superseded sampler baseline and are not credible zero-utilization measurements.

| Model | Inference | Configuration | Peak RSS upper bound GiB | Mean CPU sum % | Peak CPU upper bound % | Storage used GiB | RM samples | Resolution | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base` | 2.527 | n/m | n/m | 4.456 | 220 | minute | captured |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+coder` | 2.528 | n/m | n/m | 4.456 | 198 | minute | captured |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+parent+child` | 2.585 | n/m | n/m | 4.457 | 220 | minute | captured |
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 2.584 | n/m | n/m | 4.470 | 418 | minute | captured |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base` | 2.299 | n/m | n/m | 0.521 | 198 | minute | captured |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+coder` | 2.299 | n/m | n/m | 0.522 | 220 | minute | captured |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+parent+child` | 2.069 | n/m | n/m | 0.523 | 198 | minute | captured |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | `base+lsr+mi` | 2.162 | n/m | n/m | 0.524 | 220 | minute | captured |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base` | 2.506 | n/m | n/m | 2.732 | 572 | minute | captured |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+coder` | 2.508 | n/m | n/m | 2.735 | 1276 | minute | captured |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+parent+child` | 4.784 | n/m | n/m | 2.738 | 1912 | minute | captured |
| Gemma-3-1B-IT | RTX 5070 Laptop | `base+lsr+mi` | 0.926 | n/m | n/m | 2.740 | 1232 | minute | captured |
| Gemma-3-1B-IT | NVIDIA A40 | `base` | 2.211 | n/m | n/m | 0.528 | 682 | minute | captured |
| Gemma-3-1B-IT | NVIDIA A40 | `base+coder` | 2.225 | n/m | n/m | 0.533 | 1144 | minute | captured |
| Gemma-3-1B-IT | NVIDIA A40 | `base+parent+child` | n/m | n/m | n/m | n/m | None | None | not-measured |
| Gemma-3-1B-IT | NVIDIA A40 | `base+lsr+mi` | 1.891 | n/m | n/m | 0.594 | 1100 | minute | captured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base` | 2.497 | n/m | n/m | 3.533 | 198 | minute | captured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+coder` | 2.498 | n/m | n/m | 3.533 | 176 | minute | captured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+parent+child` | 2.556 | n/m | n/m | 3.534 | 198 | minute | captured |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | `base+lsr+mi` | 2.648 | n/m | n/m | 3.534 | 176 | minute | captured |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base` | 2.032 | n/m | n/m | 0.532 | 154 | minute | captured |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+coder` | 2.215 | n/m | n/m | 0.533 | 154 | minute | captured |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+parent+child` | 2.311 | n/m | n/m | 0.533 | 176 | minute | captured |
| Llama-3.2-1B-Instruct | NVIDIA A40 | `base+lsr+mi` | 2.183 | n/m | n/m | 0.534 | 154 | minute | captured |

## LSR routing

| Model | Inference | Expected decision | Selection recall | Expected abstention | p50 route latency (ms) |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | RTX 5070 Laptop | 0.9787 | 0.9787 | n/m | 153.52 |
| Qwen2.5-1.5B-Instruct | NVIDIA A40 | 0.8433 | 0.9787 | 0.5250 | 110.65 |
| Gemma-3-1B-IT | RTX 5070 Laptop | 0.8433 | 0.9787 | 0.5250 | 138.28 |
| Gemma-3-1B-IT | NVIDIA A40 | 0.8433 | 0.9787 | 0.5250 | 118.90 |
| Llama-3.2-1B-Instruct | RTX 5070 Laptop | 0.8433 | 0.9787 | 0.5250 | 107.49 |
| Llama-3.2-1B-Instruct | NVIDIA A40 | 0.8433 | 0.9787 | 0.5250 | 99.75 |

## Controlled training

> Speedup and efficiency are relative to `n/m` and are shown only for cohort-invariant controlled runs.

| Run | Hardware | Workers | Rounds | Supervised tokens | First loss | Last loss | Loss delta | Compute seconds | tok/s | Speedup | Scaling efficiency | Worker wait s | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | n/m | not measured |

## Controlled training resource snapshot

> These values are captured immediately after each campaign while the RM topology is still active. They are not labelled as interval peaks.

| Run | RAM GiB | Accelerator memory GiB | CPU % | Channel sent GiB | Channel received GiB | Source |
|---|---:|---:|---:|---:|---:|---|
| n/m | n/m | n/m | n/m | n/m | n/m | not measured |

## Limitations

- All quality runs use one deterministic seed; they do not establish between-seed variance.
- The design suite contains 24 cases and is a project golden, not a public leaderboard.
- The MMLU suite is an 80-record balanced subset and not the official five-shot protocol.
- Only one academic A40 was available; the other devices were occupied and were not touched.
- A measured quality regression remains a valid result and is not replaced by an expectation.
