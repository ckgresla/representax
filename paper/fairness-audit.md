# Paired Training Fairness Audit

Audit date: 2026-09-16. This is a configuration/source audit with focused CPU
diagnostics, not a new proof of numerical equivalence for every model.
Historical evidence is preserved under correction history when complete,
validated replacement panels are promoted. Passing an inventory check does
not establish identical minibatches.

## Live Rerun Readiness (2026-09-17 02:22 UTC)

All 50 corrected GPU runs have completed and passed artifact/timing validation.
Five-seed median native/reference ratios are 3.9776x late interaction, 1.4388x
outcome reward, 3.2867x process reward, 2.1262x audio-text and 3.4395x video-text.
The TPU process panel is promoted; all five native outcome runs are collected,
and the per-parameter reduction fix passed all 22 reference updates for seed 7
with final replica equality. The remaining outcome references are running.
All ten TPU late-interaction jobs finished, but promotion is held while checking
unexpected seed-dependent reference loss logs against the deterministic reader.
A local first-batch diagnostic gives global loss 0.02829 with FP32 scoring and
0.02798 with BF16 scoring, whereas the TPU logs start at 0.03198--0.07509.
This local probe alone does not identify the TPU cause. The saved timings are
not discarded, but completion and replica identity alone do not resolve this gate.

### Earlier Readiness Gates

All ten TPU process-reward runs are complete and hash-verified locally
(10/50 TPU cells), including final replica checks for all five references.
GPU process reward has also completed all ten paired runs and passed
evidence-import checks. GPUs 0-3 now execute
late interaction, outcome reward, video-text and audio-text, respectively.
Production library code is unchanged. Experiment corrections are
frozen in commits `23a61b0`, `328aa74`, `026e80f`, `8c9fc7e`, and `cdcb104`;
every run records its actual clean source revision. Nine focused CPU checks pass.

The first TPU outcome reference exhausted HBM while materializing reduced
FP32 gradients (2.05 GiB requested, 1.46 GiB available). Its replacement uses
microbatch two instead of four, retaining global batch 128 and four local
accumulation rounds. Commits `10981d9` and `f576e9a` also release obsolete
gradient buffers before the synchronization barrier. This is an experiment-only
execution correction, not a library or objective change. The focused CPU test
checks gradient replacement before synchronization and mean reduction before
clipping; ten applicable tests pass. Full TPU capacity and replica checks still
gate acceptance. The failed attempt is retained separately.
Microbatches two and one both subsequently failed in the same optimizer
materialization graph (4.90 GiB requested, 4.64 GiB available). Thus lowering
activation memory did not resolve that boundary. Commit `143321e` gives each
reduced gradient independent tensor ownership before the barrier, rather than
retaining views into the full flattened buffer; its focused CPU test checks
the ownership and synchronization order. It restores microbatch two and is
still a capacity hypothesis until the real TPU run passes. No failed timing
is accepted as a completed comparison.
That attempt completed one optimizer update but then exceeded HBM during
the next gradient materialization (4.63 GiB requested, 4.16 GiB available).
Commit `f86dddf` removes full-model gradient concatenation and reduces each
parameter's gradient functionally before synchronization and clipping.
Eleven applicable CPU tests pass, including single-/multiple-parameter mean
reduction and ownership checks. It is staged, not yet accepted on TPU.

The first corrected TPU process reference completed 22 updates but failed final
replica identity. It is excluded, not a usable timing result. Functional
gradient averaging before clipping, with explicit gradient reassignment,
resolved this gate: the seed-7 replacement completed all 22 updates with
identical final trainable-parameter hashes on all 16 ranks. The same assertions
remain mandatory for each subsequent reference run.

The TPU-derived launcher omitted the flat GPU native metrics path and wrote an
empty outer metrics file. Complete native logs remain at `run/metrics.jsonl`.
The paper importer verifies the original outer hash, then reads, hashes and
records this actual log path without modifying either original artifact. It
requires all 22 finite training updates. Both GPU frameworks exclude the
post-checkpoint interval (update 12), in addition to the two first-use updates:
19 identical warm intervals remain. Raw metrics and explicit analysis exclusions
are kept. TRL losses omitted from the outer timing log are read by update index
from the hash-verified summary's complete loss history, without changing raw logs.
Promoted process-reward panels: GPU medians 55.1760 versus 16.7875 examples/s
(3.2867x), TPU medians 85.2563 versus 9.93935 (8.5777x). These are short-input,
matched-padding controls, not long-context reasoning training. TPU final losses
are close in each pair (native/reference for seeds 7, 42, 773, 1234, 2026):
0.33298/0.33151, 0.30470/0.30461, 0.32421/0.32442, 0.32669/0.32718,
0.35490/0.34938. This is a short-run numerical observation, not convergence proof.

GPU startup exposed two execution-porting errors: direct audio training was
incorrectly enabled on GPU, and the historical GPU media chunk size of one had
not been preserved from its later source revision. Both are now corrected.
The outcome reference's export probe also required FP32 reload and identical
BF16 autocast after changing to FP32 master weights. Failed attempts are retained.

Duplicate-free late-interaction data exposed a previously latent limit mismatch:
checkpoint metadata allows 48/300 tokens, whereas this recipe uses 32/256.
A derived checkpoint config now applies 32/256 to native tokenization as well as
PyLate; original weights are unchanged. This does not increase padded shapes.
The sixteen longest queries and sixteen longest positives in the actual table
produce exactly matching token IDs and attention masks in both processors.

The real audio three-update diagnostic now reports identical gradient norms on
all 16 ranks after functional flattened gradient reduction and explicit
reassignment, unlike the in-place reduction. This is not yet a final-parameter
identity proof for audio. The corrected runner checks trainable-parameter hashes
after training, outside measured step timing. Process reward has passed this
gate; the complete audio run remains queued.
The standalone global-MNR loss/gradient/update oracle was also rerun and passed.

The corrected GPU late-interaction panel is now complete and promoted. Median
rates are 219.4986 versus 55.1840 examples/s (3.9776x); final mean NanoMSMARCO
nDCG@10 is 0.7150934 versus 0.7088870, from 0.7123384 and 0.7088870 initially.
The five reference loss histories are identical, consistent with a deterministic
reader and zero-dropout checkpoint. These are repeated timing observations,
not independent sampled trajectories. Native final loss is 0.020575--0.020662;
reference final loss is 0.020425. Scoring precision still differs, and 50 queries
do not establish convergence equivalence.

The corrected GPU video panel is complete and promoted: median 3.4370 versus
0.99928 examples/s (3.4395x). Reference per-seed rates span 0.7958--1.3206
examples/s; this variation is retained, not replaced by the fastest seed.
GPU replacement jobs run on separate devices, with up to four concurrent
processes sharing CPU, memory and storage resources. These are not isolated-host
measurements; no causal explanation of the observed reference variation is
established here. Native and reference within each seed use the same GPU.

The corrected GPU outcome-reward panel is complete and promoted: median
7.80953 versus 5.42778 examples/s (1.43881x). Both use FP32 master weights and
AdamW moments from the same BF16-rounded base and scalar head; microbatch four,
32 accumulation rounds and global batch 128 are unchanged. Final paired losses
differ by at most 0.00805. This is short-run numerical agreement, not proof of
equivalent convergence. The separate TPU reference memory gate remains open.

PyLate's global-loss oracle now passes against a same-backend, full-global-batch
XLA oracle: loss 45.1423492 on both, maximum gradient error 3.814697e-6, and zero
AdamW update error. The CPU loss is 45.1617317. HLO shows highest precision for
the fixture's encoder matmuls, but default precision for PyLate's einsum scorer
and its backward dots. Thus distributed semantics pass; strict CPU/XLA arithmetic
parity is not claimed. Full Trainer replica identity remains a separate gate.

A physical-HLO probe confirms that, with an XLA tensor allocated before
Accelerator construction, its BF16 flag alone leaves FP32-weight dot products
in FP32. Explicit XLA autocast gives BF16 dot products while preserving FP32
master weights and Adam moments. The isolated full-tuning reference wrappers
now request explicit XLA BF16 autocast, without Accelerate's global BF16 flag;
actual full-model capacity and timing remain to be checked. Audio/video retain
their explicitly BF16 frozen bases and FP32 adapters. This probe alone is not
evidence about every historical reference's physical precision.

## Scope

`methods.json` contains 265 runs: 13 recipes x two accelerators x two frameworks
x five seeds, plus five GPU dense TorchInductor controls. Each ordinary paired
cell has seeds 7, 42, 773, 1234, 2026 and matching preparation-manifest hashes.
We inspected recorded configs and historical execution code, not current defaults.
All 693 source artifacts recorded in `methods.json` were present and matched
their stored hashes. Small diagnostic JSON reports are copied verbatim into
`paper/audit/`; original artifacts remain under the paths below.

Historical sources: GPU `7b7d27e3d8a61197b9a4e88094a096ff17192be5`;
ordinary TPU `63207d0f14eedb7c0646c4e5000ea3c0d4fe972c`; audio TPU
`5b0d216375a0f3ef9de3fcc6504ef45d19a30b93`; video/V-JEPA TPU
`3491fb008f94253f3521e2f4fde29d3957f57727`. The corrected GPU process-reward
reference has separate provenance in `evidence.json` and `methods.json`.

## Decisions by Recipe

| Recipe | GPU | TPU | Action |
|---|---|---|---|
| Dense retrieval | Batch 2,048, one negative pool | Batch 2,048; 128 candidates/chip | No new mismatch found; GPU chunk sizes intentionally differ. |
| BERT semantic similarity | Batch 256, length 128, cosine regression | Same objective/batch/length | No new mismatch found. |
| MPNet semantic similarity | Batch 256, length 128, cosine regression | Same objective/batch/length | No new mismatch found. |
| BERT pair classification | Batch 256, cosine contrastive margin 0.5 | Same objective/batch/length | No new mismatch found. |
| MPNet pair classification | Batch 256, cosine contrastive margin 0.5 | Same objective/batch/length | No new mismatch found. |
| Cross encoder | Batch 128; native buckets/reference dynamic padding | Batch 128; both fixed length 512 | No new mismatch found; GPU uses best-native padding. |
| Late interaction | Different batch construction and cached-gradient normalization | Native global pool 512 vs reference local pool 32; different batch construction | Corrected paired controls needed for strict matched-training claims. |
| Outcome reward | Native FP32 master/state vs reference BF16 weights/state | Same precision discrepancy | Corrected precision control needed; also align newly initialized reward heads for trajectory claims. |
| Process reward | Corrected padding matches; new reference head is constructed before seeding | Same initialization-order issue | Share and seed the scalar head; verify replica identity. No additional length correction. |
| Image-text | Batch 512; both chunk eight | Batch 512/local pool 32; native chunk eight vs direct reference | No new objective mismatch found; repeated-caption and input-timing qualifications remain. |
| Audio-text | Native FP32 versus reference BF16 adapters; different LoRA initialization policies | Same precision/initialization issues; historical chunk padding overhead | Correct reference precision/initialization; global-negative direct TPU comparison in progress. |
| Video-text | Same adapter precision/initialization discrepancies | Same discrepancies; native chunk two pads seven to eight | Correct reference precision/initialization; capacity-check direct/chunk-seven as an execution control. |
| V-JEPA | Batch one | Batch 128 | Shared initialization/masks/objective within each accelerator; no rerun just to equalize GPU/TPU batches. |

## Confirmed Late-Interaction Issues

### TPU Negative Pool

The historical late-interaction GradCache path never dispatches to the
device-local MNR grouping helper. Its task ignores `negative_scope`, and its
scorer replicates candidate document representations. Native training therefore
scores 512 x 512 pairs; PyLate's direct Contrastive loss, with gathering disabled,
scores 16 independent 32 x 32 pools. Config/summary strings saying `local` do not
describe native execution. The relevant code is identical at `63207d0` and
`5b0d216`.

`experiments/10-cross-accelerator-framework-comparison/audit_local_negative_scope.py`
reproduces this using the historical training step on CPU meshes:

| Devices | Observed configured-local loss | True global loss | Expected grouped-local loss |
|---|---:|---:|---:|
| 2 | 0.831924915 | 0.831924975 | 0.608224988 |
| 4 | 0.831924915 | 0.831924975 | 0.330651045 |

Reducing to local pools means 16 times fewer score pairs, **not** a predicted
16x end-to-end speedup. Encoder/backward/optimizer work remains. Alternatively,
global gathering can be enabled in the reference after verifying gradients.

### Batch Construction on Both Accelerators

Native late interaction sets `shuffle=False`. Its reference does not install the
sequential sampler used by the other Sentence Transformers recipes; the pinned
default uses `RandomSampler`. Replaying those readers over the frozen 11,264 rows
gives only 11 distinct pairs in the first native batch of 512, versus 205 in a
seed-7 reference sampler reconstruction (not a saved batch trace). This is a
substantial difference in false negatives, not just a scalar loss reduction.

A corrected comparison should materialize one shared, duplicate-controlled batch
schedule and use it in both frameworks. Historical similar held-out GPU endpoints
remain observations, but are not evidence of equivalent training dynamics.

### PyLate 1.6.0 Cached Gradient Normalization

The pinned cached loss backpropagates each scoring chunk with summed cross entropy,
then divides only the detached reported loss by batch size. The replay cache
therefore contains sum gradients while the displayed loss is a mean.
`audit_reference_numerics.py cached-loss` confirms equal forward losses and exactly
5x embedding gradients at batch five, including a partial final chunk.

At historical batch 512, the factor is 512. Gradient clipping can cancel this
factor when both gradients exceed the clip threshold, but not always. All five
saved reference logs have 5--7 of 22 updates whose logged norm divided by 512 is
below one; thus clipping does not justify treating every update as unchanged.
This diagnostic does not establish the complete parameter-update difference.
Use a narrowly documented reference fix or direct loss for a corrected control;
do not silently patch the upstream reference and continue naming it unchanged.

## Outcome-Reward Precision

Historical TRL construction explicitly loads `dtype="bfloat16"` for full tuning.
A CPU reconstruction using the same checkpoint and installed TRL confirms all
596,050,944 trainable parameters remain BF16 after Trainer construction.
An AdamW dtype probe confirms BF16 parameters produce BF16 first/second moments.
Native `init_train_state` promotes all trainable leaves to the configured FP32
master dtype before constructing Optax state. BF16 forward computation in both
frameworks does not make those optimizer regimes identical.

The native reward head and the Transformers head are also separately initialized;
equal integer seeds across JAX and Torch do not imply identical weights. For a
strict trajectory control, share the initial scalar head and the BF16-rounded
base weights, then keep FP32 master parameters/moments in both. Merely loading the
original checkpoint directly as FP32 would also change the rounded starting base.
TRL does seed before constructing this model from its checkpoint string; this is
not the missing-seed issue described below. Both readers use the stored
`chosen_ids`/`rejected_ids`, rather than independently retokenizing preferences.

## Multimodal Adapter Precision and Initialization

Audio and video references load the frozen base as BF16 and call Transformers'
`add_adapter` API. This leaves the trainable LoRA parameters BF16, whereas native
LoRA parameters and optimizer moments are FP32. A tiny model using the installed
Transformers 5.6.0 / PEFT 0.19.1 path reproduces this, and the real 16-rank audio
canary independently reports BF16 trainable parameters before correction. This
is not the behavior of a different PEFT API that automatically promotes adapters.

There are two separate initialization differences:

1. Reference adapters are created before Trainer applies the run seed. Thus the
   recorded seed does not fully specify these newly initialized parameters.
   Historical TPU replica identity is unverified, not proven divergent.
2. Native `LoRALinear.from_linear` uses A drawn from a normal distribution with
   standard deviation `input_dimension**-0.5`. PEFT's default is Kaiming uniform,
   here uniform on `[-input_dimension**-0.5, input_dimension**-0.5]`. Its variance
   is one third as large. B is zero in both; equal initial adapter outputs do
   not imply identical gradient geometry or subsequent updates.

The isolated audio reference now seeds before construction, promotes trainable
adapters to FP32, and initializes A with the native distribution/scale and B to
zero. Full trainable-parameter hashes must agree across all 16 TPU ranks.
JAX and Torch still draw independently; the correction matches initialization
policy, not bitwise initial A matrices. Exact trajectory tests would additionally
share the same adapter tensors. Video and GPU audio need the corresponding fix.

## Process-Reward Initialization

The reference constructs `ScalarStepClassifier` before the Trainer sets the
seed. The base Qwen3 causal-language-model checkpoint has no scalar `score`
weights, so this creates a new random head. It also leaves Transformers'
default label count in place and selects the first score, whereas native
training uses a one-row scalar head. This does not invalidate the completed
256-token padding correction, but it is another uncontrolled initialization
choice. Set `num_labels=1`, seed before construction, share the initial scalar
head, and verify replica hashes before a corrected matched-initialization run.
Do not claim the historical TPU replicas actually differed without evidence.

## Audio Rerun

Historical audio pools were deliberately matched to the configured reference
integration, not to an inherent Sentence Transformers limitation. PJRT gradient
synchronization did not initialize the `torch.distributed` group used by its
embedding gather. The isolated rerun registers XLA's process group and uses the
unmodified ST global-negative MNR objective. All 16 ranks passed a full-precision
48-pair loss/gradient/AdamW check; each computes a 3 x 48 score block.

The real native direct-MNR canary completed. The first reference attempt exceeded
device memory (6.74 GiB requested with 6.68 GiB available). A second canary adds an
explicit post-backward execution boundary before the optimizer, with all its time
included. It passed and exposed the BF16 adapter issue. A further six-update
canary passed with FP32 trainable weights, 48 global candidates, 3 x 48 score
blocks, and matching adapter hashes on every rank. The CPU-initialized normal-A
six-update canary also completed, but its Trainer logs contain differing gradient
norms across ranks for the same update. This is an unresolved runtime gate:
initial-parameter equality and a standalone loss oracle do not prove correct
Trainer reduction/clipping. Tiny Accelerate-path diagnostics stalled when
materializing results, both with and without the extra execution boundary;
they were terminated and are inconclusive, not passing numerical checks.
The later functional flattened-reduction diagnostic completed three updates
with identical norms across ranks; see the live readiness section above for
the remaining final-parameter verification gate.
Do not extrapolate this warning to historical runs without reproducing it there.
The five-seed replacement queues have started; see live readiness above.
The device-initialization attempt was canceled during prolonged compilation;
initialization now creates tensors on CPU before device placement.
These are capacity/semantic gates,
not final throughput results. The global rerun changes the negative pool from
three to 48, so it cannot isolate the historical slowdown's cause by itself.

Artifacts: `/raid/representax-paper/fairness-audit-20260916/` and
`/raid/representax-paper/10-cross-accelerator-framework-comparison/tpu-v5e-16-audio-execution-20260916/`.
Replacement outputs are under `gpu-fairness-20260916` and
`tpu-fairness-20260916` within the Experiment 10 artifact directory.

## Repair and Rerun Plan

| Group | Required correction | Replacement measurements |
|---|---|---|
| Late interaction | Shared duplicate-controlled batch schedule; consistent negative pool; mean-loss cached gradients | Both frameworks, five seeds on GPU and TPU. |
| Outcome reward | FP32 master weights/moments after BF16 base loading; common initial scalar head | Both frameworks on both platforms if head initialization changes; five seeds per cell. |
| Process reward | Explicit one-row common head; seed before construction; replica assertion | Both frameworks on both platforms if head initialization changes; keep corrected GPU padding. |
| Audio | FP32 adapters, seeded matched initialization policy; direct global MNR on TPU | Both TPU frameworks, five seeds. Corrected GPU reference; reuse native only after contract verification. |
| Video | FP32 adapters and seeded matched initialization policy; avoid partial-chunk overhead where feasible | Corrected references on GPU/TPU; native rerun if execution/pool or initialization changes. |

Do not automatically rerun dense retrieval, sentence-pair controls, cross-encoder,
image-text or V-JEPA on the strength of these findings. No new configuration
mismatch was identified in those checked paths; this is not a universal numerical
certification. Preserve old evidence and record source patches for every repair.

## Rerun Budget (2026-09-16)

Conservative scope is 50 TPU jobs: five affected recipes x both frameworks x
five seeds. Reusing verified native audio/video results could reduce this to 45,
but a shared-initialization refresh is budgeted for all 50. These are short paired
training controls, not repetitions of the longer native adaptation studies.

| TPU recipe | Jobs | Historical combined launch-to-completion minutes |
|---|---:|---:|
| Late interaction | 10 | 19.51 |
| Outcome reward | 10 | 27.85 |
| Process reward | 10 | 25.02 |
| Audio-text | 10 | 55.37 |
| Video-text | 10 | 185.59 |
| Total | 50 | 313.35 |

Times come from each frozen run's `started_at`/`completed_at` records, not warm
throughput extrapolation. They exclude separate asset staging and readiness
diagnostics. Changed precision, graph structure and negative pools may change
runtime or memory use. The author approved six additional slice-hours at
2026-09-16 23:29:27 UTC, not the earlier proposed 8--12 hours. This is a hard
runtime limit, not a guaranteed completion time. Resolve the Trainer
runtime gate before launching the queue; if it affects other reference paths,
reassess scope rather than quietly reusing them.

The live Cloud Billing catalog (Compute Engine SKU `A9FF-6143-F24B`, effective
2026-09-16 07:00 UTC) quotes v5e Americas spot at $0.342218/chip-hour, or
$5.475488/hour for this 16-chip allocation. Six additional hours are $32.85 in
TPU compute, excluding storage, network, taxes and any credit eligibility
restrictions. This replaces the earlier unapproved $75 proposal. Spot prices
and availability can change; no on-demand fallback is included.

Billing account `0150FD-77ECA0-6313D4` is open and linked to this project.
Remaining promotional credits are **unverified**: the available account API
does not provide the balance, and no BigQuery datasets were listed in the
experiment project. Check the account's Credits page before claiming remaining
funds. The verified `representax-fairness-tpu-cleanup-20260916.timer` force-deletes
the allocation at 2026-09-17 05:29:27 UTC (September 16, 10:29:27 PM Pacific).
The original two-hour timer was stopped only after verifying the new timer.
Copy completed evidence throughout the run; no automatic deadline extension.

The same conservative GPU scope is 50 jobs on the local tinybox. Their historical
combined durations total 267.07 GPU-minutes (4.45 single-GPU-hours); independent
runs can share GPUs 0--3, subject to compilation and I/O contention. No additional
Google TPU uptime is needed for those local jobs.

## Other Qualifications

GPU/TPU batch equality is not required; equality within the framework pair is.
Encoder chunk size and rematerialization may differ without changing the objective.
TPU video has the same avoidable partial-chunk overhead found in historical audio,
but this alone is not evidence that its comparison computes a different loss.
Data wait includes the visible timed path; pre-materialized reference inputs and
short timing windows remain limitations. Missing pure cold-compilation measurements
cannot be recovered by renaming first-step latency, and need not trigger a full
throughput rerun. This audit does not certify every kernel or evaluator.
