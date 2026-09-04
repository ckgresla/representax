# Experiment 09: ModernBERT dense retrieval

Status: complete systems comparison; not accepted as representation-quality
evidence because the prepared MS MARCO source is duplicate-heavy.

## Contract

- Model: `jhu-clsp/ettin-encoder-150m` at revision
  `45d08642849e5c5701b162671ac811b7654bfd9f`
- Parameters: 149,014,272 in both frameworks
- Training: cosine MNR, BF16 compute with FP32 parameters, batch 128,
  100 updates, maximum length 128
- Representax: GradCache chunk 64 and length buckets 16/32/64/128
- Sentence Transformers: GradCache chunk 128 and TorchInductor
- Seeds and GPU pairs: 17 on 0/1, 42 on 2/3, 73 on 4/5
- Evaluation: NanoMSMARCO before and after training

## Results

| Seed | Representax warm ex/s | ST warm ex/s | Warm ratio | Representax final loss | ST final loss | Representax nDCG@10 | ST nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 333.13 | 617.13 | 0.540x | 3.8308 | 3.8588 | 0.0911 | 0.0872 |
| 42 | 330.24 | 620.62 | 0.532x | 3.8990 | 3.7703 | 0.1143 | 0.1027 |
| 73 | 333.44 | 614.73 | 0.542x | 3.7917 | 3.7564 | 0.1104 | 0.0800 |
| Mean | 332.27 | 617.49 | 0.538x | 3.8405 | 3.7951 | 0.1052 | 0.0900 |

Representax is 1.86x slower at warmed optimizer-step throughput. The loss
curves remain similar: their per-seed Pearson correlations are 0.933, 0.983,
and 0.978, with mean absolute loss differences of 0.066, 0.037, and 0.039.
Mean final nDCG@10 is 0.0153 higher for Representax, but its three-run interval
crosses zero and the data cannot support a quality claim.

Representax is 1.464x faster in the report's short-job amortized statistic
(80.37 versus 54.91 examples/s), which includes first compilation but excludes
evaluation. Do not generalize that number: the six concurrent workers contended
for host compilation resources, and the warmed reference rate is higher.

The canonical aggregate is
`/raid/representax-paper/09-modernbert-dense-retrieval/three-seed-summary.json`.
The runs were produced from Representax commit
`9c02ce8a5b9c6b49ab9a7f0132d1cdd609d4dc5b` with uncommitted experiment and
loader changes, and Sentence Transformers commit
`7d3eb16a65f62045226e08082ade63cbc71c97a4`.

## Fixed-shape GradCache control

A deterministic query/document token pair was repeated across every row of a
batch of 128. Both inputs were all-valid length 128 tensors, so both frameworks
executed exactly 32,768 tokens per optimizer update. The control used BF16
compute, FP32 parameters, identical input fingerprints, cosine MNR, and 13
optimizer updates. Evaluation and dataloading were excluded.

The accepted Representax backend uses `GradCache(implementation="custom_vjp")`.
It caches query and document representations, differentiates MNR once at that
boundary, then replays each encoder chunk into one parameter-gradient
accumulator. Chunks 32 and 64 need no layer rematerialization; chunk 128 uses
full layer rematerialization to fit comfortably on a 24 GB RTX 4090.

| Chunk | Previous RX ex/s | Custom VJP ex/s | VJP / previous | SD | Peak GiB | ST eager ex/s | VJP / eager | ST + Inductor ex/s | VJP / Inductor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 209.32 | 221.83 | 1.060x | 3.17 | 7.58 | 139.99 | 1.585x | 156.79 | 1.415x |
| 64 | 206.20 | 221.41 | 1.074x | 0.39 | 12.12 | 147.12 | 1.505x | 174.09 | 1.272x |
| 128 | OOM (39.55 GiB request) | 260.79 | n/a | 1.26 | 4.84 | 139.32 | 1.872x | 165.75 | 1.573x |

Custom-VJP statistics are four independent processes with ten measured warm
updates each. The previous Representax and each reference value are one process
with ten measured warm updates. Every custom-VJP result has the same input
fingerprint and the same 4.8520298 loss as its matched reference. Because this
performance control repeats one sample and uses a zero learning rate, its exact
loss agreement is not by itself an update-parity result.

On a distinct 128-pair batch with a nonzero learning rate, FP32 custom VJP and
Sentence Transformers agree to a loss difference of 4.77e-7 and a gradient-norm
difference of 0.0013%. Their complete 149M-parameter gradients have cosine
0.99999989 and relative L2 error 0.0478%. This establishes the mathematical
equivalence of the implementations.

BF16 is less numerically stable across the two accelerator stacks. Representax
and Sentence Transformers differ by 0.101% in loss and 2.70% in gradient norm,
but their complete gradients have cosine 0.7569 and relative L2 error 70.7%.
This is not tensor-level BF16 parity. The discrepancy is not specific to the
custom VJP: custom VJP versus rematerialized Representax has cosine 0.9964 and
relative L2 error 8.51%, while rematerialized Representax versus the reference
has cosine 0.7568 and relative L2 error 70.5%. Accordingly, BF16 scientific
equivalence must be judged from replicated loss and held-out quality
trajectories, not claimed from this one-step control.

Two implementation defects were found. The custom backward initially lost the
outer BF16 precision context and replayed FP32 master weights; carrying the
resolved policy into the custom rule roughly doubled throughput. The normal
ModernBERT forward also collected every layer output before selecting the last;
it now returns only the scan carry, while `encode_layers` retains the layerwise
path. Focused FP32 optimizer, stochastic replay, BF16, model, and training tests
pass.

The custom backend exceeds the original chunk-64 result of 1.402x over eager
and 1.184x over Inductor at every tested chunk size. Therefore the earlier
three-seed 0.538x variable-length result is not an intrinsic ModernBERT encoder
or GradCache limit; padding and the previously unequal execution recipes
materially determine it.

The control artifacts are under
`/raid/representax-paper/09-modernbert-dense-retrieval/fixed-shape-control/`.

## Real-data 30-update trajectory

One seed used the same pinned checkpoint, MS MARCO rows, batch 128, maximum
length 128, BF16 compute, FP32 master parameters, AdamW schedule, and 30
optimizer updates in all four configurations. Representax used chunk 64 and
static query/document buckets; both Sentence Transformers configurations used
chunk 128 and native dynamic padding.

| Configuration | Warm ex/s | Custom VJP ratio | 30-step ex/s | Final loss | Loss correlation with custom VJP | Loss MAE |
|---|---:|---:|---:|---:|---:|---:|
| Representax custom VJP | 361.08 | 1.000x | 100.21 | 4.15219 | 1.000 | 0.0000 |
| Representax rematerialized | 341.31 | 1.058x | 100.17 | 4.15219 | 0.977 | 0.0405 |
| Sentence Transformers eager | 391.69 | 0.922x | 376.17 | 4.17523 | 0.989 | 0.0368 |
| Sentence Transformers + Inductor | 667.90 | 0.541x | 20.49 | 4.06847 | 0.956 | 0.0622 |

`Custom VJP ratio` is custom-VJP throughput divided by that row's throughput.
The 30-step rate includes first compilation but excludes evaluation. Inductor's
roughly three-minute compilation dominates that short-job figure; its warm rate
is the relevant long-run result.

Padding explains the eager crossover. The warm Representax batches contain
11,887 real tokens on average but execute 18,432 token positions, leaving 35.5%
padding. Sentence Transformers executes 15,241 positions for the same real
tokens through per-batch dynamic padding, leaving 22.0% padding. Representax
therefore performs 20.9% more encoder work. Normalized by padded positions,
custom VJP processes about 52.0K positions/s versus eager's 46.6K positions/s,
so the Representax computation is approximately 1.115x faster before padding
waste reverses example throughput. Inductor reaches approximately 79.5K padded
positions/s and remains 1.53x faster on that basis.

The two Representax final losses differ by 4.77e-7. Custom VJP also tracks eager
Sentence Transformers closely over all 30 updates (Pearson 0.989, MAE 0.0368).
The result establishes a 1.058x custom-VJP improvement over rematerialized
GradCache, but it does not establish an end-to-end real-data speedup over either
Sentence Transformers configuration. Artifacts are under
`/raid/representax-paper/09-modernbert-dense-retrieval/real-trajectory-30-step/`.

## Padding isolation

Two 30-update controls changed only sequence-shape handling. First, Sentence
Transformers was forced to the same query-16/document-128 tensors used by the
coarse Representax buckets. Second, Representax retained the same examples and
order but selected from finer document buckets at 64, 80, 96, 112, and 128.

| Configuration | Positions/example | Padding | Warm ex/s | Change from original |
|---|---:|---:|---:|---:|
| Representax, coarse buckets | 144.00 | 35.51% | 361.08 | baseline |
| Representax, finer buckets | 129.23 | 28.66% | 393.16 | 1.089x |
| Sentence Transformers eager, native padding | 119.07 | 22.01% | 391.69 | baseline |
| Sentence Transformers eager, fixed shapes | 144.00 | 35.51% | 346.35 | 0.884x |
| Sentence Transformers + Inductor, native padding | 119.07 | 22.01% | 667.90 | baseline |
| Sentence Transformers + Inductor, fixed shapes | 144.00 | 35.51% | 583.96 | 0.874x |

This establishes padding as the cause of the eager crossover. On matched fixed
shapes, coarse-bucket Representax is 1.043x faster than eager Sentence
Transformers. With practical shape policies, finer-bucket Representax is within
0.4% of the native eager reference. Inductor remains 1.70x faster than
finer-bucket Representax, so its remaining advantage is not explained by
padding.

The finer Representax run compiled four training signatures and spent 92.0
seconds in first use, so its 30-step rate including compilation is only 38.16
examples/s. That cold-start cost is separate from the measured warm throughput
and must remain visible in reporting. Shape-only loss trajectories remain close:
the fine/coarse Representax correlation is 0.985, fixed/native eager is 0.983,
and fixed/native Inductor is 0.979. Artifacts, including the two preserved
failed collator-wrapper attempts, are under
`/raid/representax-paper/09-modernbert-dense-retrieval/padding-ablation-30-step/`.
