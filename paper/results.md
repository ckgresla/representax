# Measured Framework Throughput

Ratio of median per-seed examples/s. Dots in the figure are paired seed ratios, not confidence intervals.

| Recipe | GPU native | GPU reference | GPU ratio | TPU native | TPU reference | TPU ratio |
|---|---:|---:|---:|---:|---:|---:|
| Dense retrieval | 305.348 | 211.222 | 1.446 | 2510.213 | 2506.150 | 1.002 |
| Similarity / MPNet | 559.720 | 367.802 | 1.522 | 4045.865 | 808.686 | 5.003 |
| Similarity / BERT | 286.211 | 384.077 | 0.745 | 3439.733 | 829.874 | 4.145 |
| Classification / MPNet | 552.810 | 361.943 | 1.527 | 4124.540 | 773.708 | 5.331 |
| Classification / BERT | 296.207 | 382.917 | 0.774 | 3478.858 | 758.164 | 4.589 |
| Cross-encoder | 670.369 | 441.531 | 1.518 | 2701.598 | 807.870 | 3.344 |
| Late interaction [L] | 262.290 | 40.455 | 6.484 | 2765.341 | 1059.692 | 2.610 |
| Outcome reward | 8.001 | 6.513 | 1.229 | 103.871 | 59.927 | 1.733 |
| Process reward | 56.229 | 6.244 | 9.005 | 85.259 | 24.311 | 3.507 |
| Image-text [C] | 321.020 | 106.223 | 3.022 | 2259.882 | 1062.058 | 2.128 |
| Audio-text | 2.660 | 1.845 | 1.442 | 11.247 | 13.451 | 0.836 |
| Video-text | 3.482 | 1.490 | 2.337 | 80.699 | 8.465 | 9.534 |
| V-JEPA 2.1 [C] | 6.571 | 1.304 | 5.040 | 87.589 | 59.633 | 1.469 |

Dense Inductor control: 316.610 examples/s; native / compiled = 0.964x.

GPU and TPU allocations and negative-pool semantics differ; the scatter compares relative framework rates, not equal-cost hardware performance.

## Held-Out Learning

Three-seed mean and sample SD; nDCG@10.

| Recipe / evaluation | Initial | Final mean | Final SD |
|---|---:|---:|---:|
| Dense / NanoMSMARCO | 0.0730 | 0.5794 | 0.0111 |
| CLIP / text-to-image | 0.7373 | 0.8148 | 0.0023 |
| CLIP / image-to-text | 0.7168 | 0.7665 | 0.0025 |
| Late interaction / NanoMSMARCO | 0.7104 | 0.6925 | 0.0010 |
| Dense / trec-dl-2019 | unmeasured | 0.5404 | 0.0082 |
| Dense / natural-questions | unmeasured | 0.2922 | 0.0011 |

## Multimodal Adaptation

Each arm uses its own initial evaluation; mean +/- sample SD.

| Strategy | Evaluation | Initial | Final | Delta |
|---|---|---:|---:|---:|
| connectors | audiocaps | 0.4378 +/- 0.0000 | 0.4691 +/- 0.0011 | 0.0313 +/- 0.0011 |
| connectors | flickr30k | 0.5724 +/- 0.0000 | 0.7121 +/- 0.0021 | 0.1398 +/- 0.0021 |
| connectors | msrvtt | 0.3731 +/- 0.0000 | 0.4772 +/- 0.0091 | 0.1041 +/- 0.0091 |
| connectors | nanomsmarco | 0.6509 +/- 0.0000 | 0.6509 +/- 0.0000 | 0.0000 +/- 0.0000 |
| connectors-lora | audiocaps | 0.4372 +/- 0.0000 | 0.4799 +/- 0.0010 | 0.0427 +/- 0.0010 |
| connectors-lora | flickr30k | 0.5723 +/- 0.0000 | 0.7466 +/- 0.0044 | 0.1743 +/- 0.0044 |
| connectors-lora | msrvtt | 0.3743 +/- 0.0000 | 0.4984 +/- 0.0014 | 0.1241 +/- 0.0014 |
| connectors-lora | nanomsmarco | 0.6509 +/- 0.0000 | 0.6121 +/- 0.0043 | -0.0387 +/- 0.0043 |
| full | audiocaps | 0.4378 +/- 0.0000 | 0.5028 +/- 0.0027 | 0.0650 +/- 0.0027 |
| full | flickr30k | 0.5724 +/- 0.0000 | 0.6987 +/- 0.0063 | 0.1264 +/- 0.0063 |
| full | msrvtt | 0.3731 +/- 0.0000 | 0.4733 +/- 0.0034 | 0.1002 +/- 0.0034 |
| full | nanomsmarco | 0.6509 +/- 0.0000 | 0.5994 +/- 0.0061 | -0.0515 +/- 0.0061 |

## A100 Strong Scaling

| GPUs | Tokens/s mean | Sample SD | Paired speedup | Efficiency |
|---|---:|---:|---:|---:|
| 1 | 32015.3 | 9.3 | 1.000x | 100.0% |
| 2 | 63000.1 | 68.6 | 1.968x | 98.4% |
| 4 | 121970.6 | 494.9 | 3.810x | 95.2% |
| 8 | 219975.4 | 534.8 | 6.871x | 85.9% |

## Recorded First-Use Costs

Sum of `perf/compilation_and_first_step_seconds` records per run. This includes first execution and may include cache loading; it is not necessarily cold compilation. Missing is not zero. Ranges span the five seeds, and are not confidence intervals.

| Panel | Recipe | Framework | Recorded runs | Seconds min / median / max |
|---|---|---|---:|---:|
| gpu-rtx4090 | audio-text | representax | 5 | 15.96 / 16.29 / 18.07 |
| gpu-rtx4090 | audio-text | reference | 0 | not recorded in this field |
| gpu-rtx4090 | cross-encoder | representax | 5 | 6.28 / 6.33 / 11.07 |
| gpu-rtx4090 | cross-encoder | reference | 0 | not recorded in this field |
| gpu-rtx4090 | dense-retrieval | representax | 5 | 24.08 / 26.18 / 26.61 |
| gpu-rtx4090 | dense-retrieval | reference | 0 | not recorded in this field |
| gpu-rtx4090 | image-text | representax | 5 | 9.57 / 9.81 / 10.00 |
| gpu-rtx4090 | image-text | reference | 0 | not recorded in this field |
| gpu-rtx4090 | late-interaction | representax | 5 | 124.58 / 125.55 / 128.04 |
| gpu-rtx4090 | late-interaction | reference | 0 | not recorded in this field |
| gpu-rtx4090 | outcome-reward | representax | 5 | 34.90 / 37.39 / 39.22 |
| gpu-rtx4090 | outcome-reward | reference | 0 | not recorded in this field |
| gpu-rtx4090 | pair-classification-bert-base | representax | 5 | 5.90 / 6.04 / 10.98 |
| gpu-rtx4090 | pair-classification-bert-base | reference | 0 | not recorded in this field |
| gpu-rtx4090 | pair-classification-mpnet-base | representax | 5 | 16.97 / 17.64 / 29.23 |
| gpu-rtx4090 | pair-classification-mpnet-base | reference | 0 | not recorded in this field |
| gpu-rtx4090 | process-reward | representax | 5 | 5.46 / 5.49 / 8.48 |
| gpu-rtx4090 | process-reward | reference | 0 | not recorded in this field |
| gpu-rtx4090 | semantic-similarity-bert-base | representax | 5 | 5.82 / 5.88 / 10.62 |
| gpu-rtx4090 | semantic-similarity-bert-base | reference | 0 | not recorded in this field |
| gpu-rtx4090 | semantic-similarity-mpnet-base | representax | 5 | 17.17 / 17.28 / 30.88 |
| gpu-rtx4090 | semantic-similarity-mpnet-base | reference | 0 | not recorded in this field |
| gpu-rtx4090 | v-jepa | representax | 5 | 6.16 / 6.22 / 6.28 |
| gpu-rtx4090 | v-jepa | reference | 0 | not recorded in this field |
| gpu-rtx4090 | video-text | representax | 5 | 8.86 / 8.89 / 8.94 |
| gpu-rtx4090 | video-text | reference | 0 | not recorded in this field |
| tpu-v5e-16 | audio-text | representax | 5 | 33.14 / 33.45 / 34.57 |
| tpu-v5e-16 | audio-text | reference | 0 | not recorded in this field |
| tpu-v5e-16 | cross-encoder | representax | 5 | 9.08 / 9.27 / 9.81 |
| tpu-v5e-16 | cross-encoder | reference | 0 | not recorded in this field |
| tpu-v5e-16 | dense-retrieval | representax | 5 | 286.89 / 288.20 / 288.77 |
| tpu-v5e-16 | dense-retrieval | reference | 0 | not recorded in this field |
| tpu-v5e-16 | image-text | representax | 5 | 9.67 / 9.82 / 18.00 |
| tpu-v5e-16 | image-text | reference | 0 | not recorded in this field |
| tpu-v5e-16 | late-interaction | representax | 5 | 49.98 / 50.21 / 63.43 |
| tpu-v5e-16 | late-interaction | reference | 0 | not recorded in this field |
| tpu-v5e-16 | outcome-reward | representax | 5 | 7.40 / 7.62 / 9.76 |
| tpu-v5e-16 | outcome-reward | reference | 0 | not recorded in this field |
| tpu-v5e-16 | pair-classification-bert-base | representax | 5 | 10.93 / 11.06 / 11.98 |
| tpu-v5e-16 | pair-classification-bert-base | reference | 0 | not recorded in this field |
| tpu-v5e-16 | pair-classification-mpnet-base | representax | 5 | 185.31 / 185.57 / 197.60 |
| tpu-v5e-16 | pair-classification-mpnet-base | reference | 0 | not recorded in this field |
| tpu-v5e-16 | process-reward | representax | 5 | 6.96 / 7.00 / 8.61 |
| tpu-v5e-16 | process-reward | reference | 0 | not recorded in this field |
| tpu-v5e-16 | semantic-similarity-bert-base | representax | 5 | 11.08 / 11.12 / 12.17 |
| tpu-v5e-16 | semantic-similarity-bert-base | reference | 0 | not recorded in this field |
| tpu-v5e-16 | semantic-similarity-mpnet-base | representax | 5 | 185.06 / 185.44 / 198.59 |
| tpu-v5e-16 | semantic-similarity-mpnet-base | reference | 0 | not recorded in this field |
| tpu-v5e-16 | v-jepa | representax | 5 | 34.35 / 35.28 / 41.22 |
| tpu-v5e-16 | v-jepa | reference | 0 | not recorded in this field |
| tpu-v5e-16 | video-text | representax | 5 | 12.82 / 15.04 / 16.18 |
| tpu-v5e-16 | video-text | reference | 0 | not recorded in this field |
| gpu-rtx4090-torchinductor | dense-retrieval | representax | 0 | not recorded in this field |
| gpu-rtx4090-torchinductor | dense-retrieval | reference | 0 | not recorded in this field |
