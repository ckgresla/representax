# Representax: Scalable Representation Learning in JAX

Working manuscript, 11 September 2026. Abstract: [abstract.md](abstract.md).
This is a claim-bounded first draft, not submission-ready typesetting. The
roadmap remains in `todo.org`; this directory contains manuscript material.

## 1. Introduction

Representation-learning research requires more than an encoder and a loss.
Researchers must connect heterogeneous data sources, control trainable
parameters, preserve sufficiently large negative pools, evaluate learned
representations, and execute the same training logic across accelerators.
These requirements recur across retrieval, multimodal adaptation, reward
modeling, and joint-embedding predictive objectives.

Representax provides a shared JAX training interface for these workloads.
Models and processors define the representation boundary; tasks define the
learning objective; configuration selects data, optimization and execution.
The contribution is the integration and composability of these facilities for
research, not a new compiler or representation-learning objective.

We evaluate three questions: (1) how training throughput compares with maintained
reference implementations, (2) whether the framework supports useful held-out
learning in complete adaptation runs, and (3) how a fixed training workload
scales across GPUs. Results are favorable in many measured settings but not
uniformly: backend choice, input processing, compilation and interconnect
constraints materially affect the outcome.

## 2. System

Describe the model/processor, task, data-loader and optimizer interfaces with a
single minimal executable example, selected from the existing public API.
Highlight task composition, trainable-parameter selection, custom-VJP GradCache,
homogeneous source-per-batch multimodal sampling, preprocessing/prefetch,
checkpoint/resume and explicit sharding. Distinguish ordinary MLM gradient
accumulation from representation-gradient caching for contrastive objectives.

Do not claim every task supports every execution strategy. In particular,
late interaction's rematerialized GradCache support does not establish support
for the dense custom-VJP path. The capability table must separate implemented
features from combinations exercised in the paper.

## 3. Experimental Protocol

**Systems comparisons.** Experiment 10 contains 13 concrete recipe rows, five
seeds per framework, on single RTX 4090 GPUs and a 16-chip v5e allocation.
There are 130 measured framework runs per accelerator panel. The dense GPU
Inductor control is additional; it is not a compiled baseline for all recipes.
GPU and TPU panels are matched within their own hardware and recipe contracts,
not assumed identical across accelerators. For example, dense TPU uses matched
per-device negative pools; the single-GPU negative pool covers the full batch.

Compute each run's throughput as examples divided by complete measured
optimizer-step intervals, including input wait. Report steps/s as well. Retain
the existing ratio of median per-seed rates for the main table. Show individual
paired seed ratios rather than treating successive updates as independent
experimental repetitions. Twenty warm updates per seed are a bounded timing
screen, not proof of long-run stationarity; native GPU late interaction retains
15 usable intervals after first-use exclusions. Compilation, initial loading,
evaluation and checkpoint costs are separate, not silently free.

**Learning studies.** Experiments 11 and 13 each use three independent native
training seeds. Experiment 14 uses three seeds for each of three adaptation
strategies. Held-out retrieval metrics establish finite-budget learning, not
asymptotic convergence or SOTA performance. Report each dataset and retrieval
direction separately. Loss correlation is descriptive and does not establish
parameter, gradient or learned-solution equivalence.

**Scaling.** Experiment 15 uses a controlled, randomly initialized 505,072,128
parameter ModernBERT MLM, length 512, BF16 compute, FP32 parameters/AdamW and
full activation rematerialization. Hold 256 sequences (131,072 input tokens)
per optimizer update fixed. Local microbatch 16 gives 16/8/4/2 accumulation
rounds on 1/2/4/8 A100 SXM4 GPUs; synchronize gradients after accumulation.
Each seed runs 21 updates, with the final 20 used for throughput. Seeds 7, 42
and 773 each use identical token blocks and masking across GPU counts.
WikiText-2 supplies repeated real text; this is not a complete pretraining run
or a reproduction of the released ModernBERT training recipe.

## 4. Framework and Accelerator Comparison

Use [results.md](results.md), [framework-throughput](figures/framework-throughput.pdf)
and [cross-accelerator](figures/cross-accelerator.pdf). Among the 12 rows other
than historical late interaction, native throughput exceeds the default
reference in 10 GPU rows; the two BERT Base sentence-pair controls are slower.
On TPU, 10 of these rows are faster, dense retrieval is approximately tied,
and audio-text is slower. These are descriptive counts, not significance
tests or a single aggregated framework score.

The GPU MPNet dense result is 305.35 examples/s versus 211.22 for eager Sentence
Transformers (1.446x), but the compiled reference reaches 316.61 examples/s
(native/reference 0.964x). Do not substitute an earlier optimized ETTIN or
repeated-sample control into this panel.

TPU image-text and V-JEPA have large first-seed cold-filesystem effects. Native
data streaming and reference pre-materialization also differ in where input
work occurs. Keep those rows explicitly qualified, display every seed, and do
not treat their medians as isolated hardware or compiler effects. Historical
late interaction has duplicated training pairs and differing loss reductions;
retain its timing as a qualified historical observation, not evidence of
equivalent learning. No additional rerun is implied by retaining these limits.

The cross-accelerator scatter shows backend-dependent relative performance.
It is not a Pareto frontier or equal-cost GPU/TPU comparison: the horizontal
and vertical axes represent different hardware allocations and, where
documented, different recipe parameters.

## 5. End-to-End Learning

Dense adaptation starts from ETTIN Encoder 150M and uses duplicate-free MS MARCO.
Report initial/final NanoMSMARCO, and final TREC DL 2019 and Natural Questions
transfer separately; the latter two have no initial measurements in this
evidence inventory, so no transfer-improvement claim is made.

CLIP ViT-B/32 fine-tuning on COCO improves mean Flickr30k nDCG@10 from 0.7373
to 0.8148 for text-to-image and 0.7168 to 0.7665 for image-to-text.

Jina v5 Omni Nano supports text-anchored joint adaptation over COCO, AudioCaps,
MSR-VTT and MS MARCO. Image, audio and video retrieval improve under all three
strategies. Connector-only preserves the frozen text tower's text score, while
connector-plus-LoRA and full fine-tuning reduce text retrieval performance.
Use each strategy's own initial scores, since initial LoRA media scores differ
slightly. Connector-only omits text-only training and samples three sources
uniformly; the other strategies sample four. Thus this is a useful study of
adaptation recipes, not an isolated causal ablation of parameter selection or
an equal-work comparison of their aggregate throughputs.

Retain the revised late-interaction negative result in the limitations or
appendix: NanoMSMARCO nDCG@10 falls from 0.7104 to 0.6925 despite decreasing
training loss. This is a small-panel, reproducible regression. Recipe-induced
forgetting is a hypothesis, not a demonstrated cause.

## 6. Strong Scaling

Mean input throughput rises from 32,015 tokens/s on one A100 to 63,000,
121,971 and 219,975 on two, four and eight devices. Mean within-seed speedups
are 1.968x, 3.810x and 6.871x; corresponding efficiencies are 98.4%, 95.2% and
85.9%. This supports near-linear strong scaling through four GPUs and high
efficiency on eight for this workload. It does not establish that every
Representax workload scales similarly.

All 252 optimizer updates are finite and non-skipped. Token and masking hashes
match across device counts within each seed. BF16 trajectories are close but
not identical; maximum observed cross-device loss spread is about 0.0523.
The library source is identical across runs. Two harness revisions differ in
HLO inspection and resume handling, not the model or training update.

Cold compilation for the selected signatures took approximately 126-148
seconds each; subsequent executable-cache loading took approximately 14 seconds.
Keep the initial 8-GPU post-compilation inspection failure in the record; its
retry used that compiled cache. Compiled memory is approximately 32.5-33.6 GiB
per device, distinct from the allocator's preallocated pool.

The six-4090 workstation's weaker scaling is useful interconnect-constrained
context. Its profile attributes most of the four-GPU gap from ideal to exposed
gradient collectives, while encoder computation scales well. The A100-versus-
4090 difference is not an isolated NVLink ablation: GPUs, hosts and selected
microbatches differ. FSDP/hybrid capacity remains unmeasured and is not a result
of this DDP experiment.

## 7. Design Analysis

Select only existing, matched ablations for custom-VJP GradCache, padding and
prefetch that directly explain an end-to-end result. Put detailed lowering and
compilation diagnostics in the appendix. A controlled repeated-sample throughput
win must not be presented as a measured real-data speedup. Do not add new
experiments to satisfy this outline.

## 8. Capabilities and Limitations

Use [capabilities.md](capabilities.md). Multi-host TPU execution is observed;
multi-node GPU scaling is not. Explicit sharding is an API capability, while
the accepted scaling panel is replicated DDP. Fixed-budget quality gains do
not prove full convergence; 20-update comparisons do not establish long-run
reliability. State reference modifications and integrations explicitly, retain
negative outcomes and startup costs, and avoid novelty claims about capabilities
also provided by reference frameworks.

## 9. Related Work and Conclusion

Before submission, add verified primary citations for JAX, Sentence
Transformers, PyLate/ColBERT, GradCache, CLIP, Jina Omni, JEPA, PyTorch/XLA,
and the datasets. Do not infer priority from this working draft.

Representax demonstrates that a shared JAX training interface can support a
broad representation-learning program, useful multimodal adaptation, and
efficient multi-GPU execution. Its value is a foundation for subsequent
research; performance and learning outcomes remain workload-dependent.
