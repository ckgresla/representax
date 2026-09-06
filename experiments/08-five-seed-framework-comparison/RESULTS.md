# Five-seed framework decision sweep

## Result

All 65 paired runs completed. Each of the 13 recipes ran five seeds and 20
optimizer updates per framework: 100 updates per framework per recipe and
2,600 accepted optimizer updates in total.

Throughput is measured from one optimizer completion to the next and includes
input wait. First-use compilation and intervals containing the midpoint
checkpoint or resume are retained in the raw logs but excluded from the
steady-state rate.

| Recipe | Reference | Representax ex/s | Reference ex/s | Median ratio [min, max] | Representax loss change | Reference loss change | Direction agreement |
|---|---|---:|---:|---:|---:|---:|---:|
| Dense retrieval | Sentence Transformers | 337.54 | 460.07 | 0.740x [0.731, 0.762] | -0.1275 | -0.1277 | 5/5 |
| Semantic similarity, MPNet | Sentence Transformers | 417.52 | 357.72 | 1.172x [1.144, 1.187] | +0.0131 | +0.0134 | 5/5 |
| Semantic similarity, BERT Base | Sentence Transformers | 256.35 | 381.71 | 0.673x [0.640, 1.130] | -0.0939 | -0.0952 | 5/5 |
| Pair classification, MPNet | Sentence Transformers | 384.44 | 355.25 | 1.082x [1.032, 1.091] | -0.0174 | -0.0172 | 5/5 |
| Pair classification, BERT Base | Sentence Transformers | 254.11 | 380.18 | 0.666x [0.654, 0.684] | -0.0135 | -0.0136 | 5/5 |
| Cross-encoder | Sentence Transformers | 667.08 | 396.28 | 1.705x [1.429, 1.722] | -1.5308 | -1.5534 | 5/5 |
| Late interaction | PyLate | 134.98 | 54.11 | 2.452x [2.072, 2.555] | +0.2428 | -0.1649 | not comparable |
| Outcome reward | TRL | 7.97 | 6.79 | 1.175x [1.173, 1.186] | -0.1542 | -0.2188 | 5/5 |
| Process reward | TRL | 55.00 | 16.76 | 3.309x [3.263, 3.392] | -0.4340 | -0.4468 | 5/5 |
| Image-text | Sentence Transformers | 319.80 | 103.28 | 3.083x [2.539, 4.332] | -2.6304 | -2.6361 | 5/5 |
| Audio-text | Sentence Transformers | 2.66 | 1.77 | 1.500x [1.444, 1.531] | -0.2331 | -0.1006 | 5/5 |
| Video-text | Sentence Transformers | 3.56 | 1.43 | 2.447x [2.006, 2.503] | +0.1205 | +0.1923 | 5/5 |
| V-JEPA 2.1 | facebookresearch/vjepa2 | 6.73 | 1.29 | 5.285x [5.104, 5.341] | -0.1203 | -0.1203 | 5/5 |

The displayed rates are medians across seeds. The ratio is the median of each
seed's Representax/reference ratio, so it is not necessarily the ratio of the
two displayed rates.

## Interpretation

Representax is faster in 10 of 13 comparisons. The largest repeatable gains are
V-JEPA 2.1, process reward modeling, image-text retrieval, late interaction,
video-text retrieval, and cross-encoder training. Audio-text is a stable 1.50x
win, while MPNet similarity, MPNet pair classification, and outcome reward are
smaller wins.

The three losses are optimized dense MPNet against TorchInductor and both BERT
Base controls. Their input wait is effectively zero, so prefetching and
preprocessing are not the cause; the remaining gaps are in model/compiler
execution. BERT similarity seed 7 is a reference-throughput outlier, but the
five-seed median still shows a clear loss.

All 12 loss-comparable recipes agree on loss direction for all five seeds.
Late interaction uses different loss reductions and its raw loss directions
disagree, so its throughput result must not be promoted without shared held-out
quality. Twenty updates measure short loss dynamics, not convergence.

The result supports continuing with a selective systems paper. It does not
support a claim that Representax is universally faster. Longer runs should be
limited to winning rows with shared Representax evaluation, while dense and
BERT remain explicit limitations.

## Execution notes

- Median Representax input wait is below 0.1% for every recipe after bounded
  prefetch and lazy preprocessing.
- The 3B audio graph used roughly 300-370 GiB of host memory during native XLA
  compilation. The launcher therefore serializes audio-text jobs.
- Two completed continuous audio runs hit a reporting-only `resumed` check
  after training, evaluation, checkpoints, and export had finished. Their
  reports were recovered from durable metrics. The condition is fixed, and the
  final audio seed also passed the worker's exact reload check.
- LeJEPA is omitted because the current reference path is an experiment-owned
  timm oracle using stable-pretraining components, not the external
  stable-pretraining training loop. NQ and MIRACL are lifecycle checks without
  external paired trainers.

## Revisions

- Representax launch commit: `9c02ce8a5b9c6b49ab9a7f0132d1cdd609d4dc5b`
- Sentence Transformers 5.6.1: `7d3eb16a65f62045226e08082ade63cbc71c97a4`
- PyLate 1.6.0: `346ea2b9c273256919809ac35b3a055970780c9c`
- TRL 1.10.0: `a7be897f5c8d7b52161f9f8a47d8e6242456b898`
- facebookresearch/vjepa2: `204698b45b3712590f06245fbfba32d3be539812`

The aggregate is
`/raid/representax-paper/08-five-seed-framework-comparison/summary.json`; the
rendered artifact is the adjacent `results.md`. Each seed directory contains
raw logs, metrics, reports, checkpoints, exports, and content-addressed
reference provenance.
