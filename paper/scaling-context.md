# Scaling in Context

Research note, 11 September 2026. Published numbers below are context, not
baselines we ran or a leaderboard comparison. They differ in model, hardware,
precision, workload, topology, baseline and timing protocol.

## Our Result

The Lambda machine had **eight A100 SXM4 GPUs, each with 40 GB memory**.
This is a single GPU node; multi-host evidence comes separately from the TPU
panel. The model is a controlled ModernBERT encoder with an MLM head, not GPT.

| GPUs | Mean within-seed speedup | Parallel efficiency |
|---:|---:|---:|
| 1 | 1.000x | 100.0% |
| 2 | 1.968x | 98.4% |
| 4 | 3.810x | 95.2% |
| 8 | 6.871x | 85.9% |

Source: the frozen Experiment 15 records in `evidence.json`. Model, global
batch, sequence length and masking are fixed across GPU counts. Each cell
has three seeds and twenty measured warm optimizer updates per seed.

## What Linear Means

For fixed-work strong scaling, speedup is `T(1) / T(N)` and efficiency is
`speedup / N`. Perfect scaling requires `T(N) = T(1) / N`. A useful conceptual
model is `T(N) = T_compute(1)/N + T_exposed_communication(N) + T_other(N)`;
this is an explanation, not a fit to our traces. Real kernels also change
efficiency with shapes, so even the compute term need not scale exactly.

Near-linear scaling is an established attainable outcome for well-configured
training, not an automatic guarantee. Exact linearity is not the usual
acceptance criterion. Apparent superlinear speedup can arise from changed
kernel efficiency or memory behavior, or an inadequate baseline; it requires
investigation rather than being intrinsically impossible.

With weak scaling, local work stays fixed while the global batch grows. It
can be easier to sustain throughput efficiency because each device retains
enough work to amortize communication. This answers a different question from
our fixed-global-batch experiment.

## Primary-Source Examples

| Work | Published observation | Why it is not our baseline |
|---|---|---|
| Goyal et al., *Accurate, Large Minibatch SGD* | About 90% efficiency from 8 to 256 GPUs; ResNet-50 | Batch grows from 256 to 8,192; P100-based system and different task |
| Li et al., *PyTorch Distributed* | Near-linear scaling up to 256 GPUs when appropriately configured; untuned cases are substantially worse | Includes synchronization-frequency experiments and different per-iteration work; not our fixed-batch A100 MLM panel |
| Shoeybi et al., *Megatron-LM*, Figure 5 | 77% efficiency for eight-way model parallelism; 74% for combined model/data parallelism on 512 GPUs | Explicitly weak-scales model size from a 1.2B baseline to 8.3B; V100 system, different parallelism |

Goyal et al. explicitly enlarge the minibatch as devices increase and show
both speed and retained accuracy. This is a useful example of a strong result
that does not need 100% scaling efficiency. [Paper](https://arxiv.org/pdf/1706.02677)

Li et al. report latency as a function of GPU count, examine gradient
communication and synchronization frequency, and show how network and
configuration affect scaling. Their broad near-linear statement must not be
read as a guarantee for every plotted workload. [Sections 5.1-5.3](https://arxiv.org/html/2006.15704v1#S5)

Use Figure 5 rather than mixing numbers across Megatron versions or its
abstract: the plotted model-only and model-plus-data efficiencies are 77%
and 74%. These are not fixed-model, fixed-batch strong-scaling results.
[Section 4.1 and Figure 5](https://arxiv.org/pdf/1909.08053)

Our result is credible evidence of efficient single-node strong scaling for
this workload. It does not show that Representax beats those systems, and
near-linear scaling alone is not the novelty. The contribution is a unified
research toolkit whose empirical story includes broad execution coverage,
useful held-out adaptation and efficient scaling.

Suggested manuscript sentence:

> Representax achieves near-linear strong scaling through four A100 GPUs and
> retains 85.9% parallel efficiency on eight, without increasing the global
> training batch.
