# Exact GradCache execution

GradCache is an execution schedule for a representation objective. It is not a
second loss. In Representax, direct diagnostics and cached training both
evaluate `MNRTask` through one shared row-level MNR primitive; the execution
choice determines which encoder activations and score rows survive until the
backward pass.

```python
from representax.tasks.retrieval import MNRTask
from representax.train import GradCache, build_train_step

step = build_train_step(
    MNRTask(scale=20.0, symmetric=True),
    optimizer,
    execution=GradCache(
        query_chunk_size=16,
        document_chunk_size=8,
        loss_row_chunk_size=16,
    ),
)
```

The logical batch and negative population do not change. Query and document
chunk sizes bound separate encoder replay passes. `loss_row_chunk_size` instead
tiles rows of the representation-level similarity loss after both encoders have
run; it bounds score-matrix memory without encoding another microbatch.

The user-facing `GradCacheConfig.micro_batch_size` supplies the default for all
three bounds. Optional query and document overrides exist for asymmetric or
multimodal towers whose encoder costs differ; `loss_row_chunk_size` is an
orthogonal override for the shared representation-level loss.

## Named and arbitrary sharded execution

DDP and FSDP are named defaults over the same `ShardingPlan`; explicit
model-path partition rules produce that same plan. Sharding is separate from
GradCache: the task and loss keep identical scientific semantics while the plan
changes physical parameter, Optax-state, batch, and gradient placement.

```python
import jax

from representax.tasks.retrieval import MNRTask
from representax.train import (
    GradCache,
    ShardingPlan,
    build_train_step,
)

mesh = jax.make_mesh((len(jax.devices("gpu")),), ("data",))
plan = ShardingPlan.fsdp(
    state,
    optimizer,
    mesh,
    parameter_axis_name="data",
    data_axis_name="data",
)
step = build_train_step(
    MNRTask(scale=20.0),
    optimizer,
    plan=plan,
    execution=GradCache(query_chunk_size=16),
)
state = plan.place_state(state)
batch = plan.place_batch(batch)
key = jax.device_put(key, plan.replicated_sharding)
result = step(state, batch, key)
```

The positive relation is row-sharded with its global document axis intact.
Each rank encodes only its local query and document records; GradCache gathers
compact representations, validity vectors, and relation rows to recover the
exact global negative population. FSDP keeps model and Optax state sharded at
rest, AllGathers parameters for the forward, and ReduceScatters their
cotangents. Same-axis data/FSDP uses a gradient sum because each device sees
distinct examples; a parameter-only axis uses a mean because its batch is
replicated. Global loss and task metrics are explicitly reduced, and
FSDP/custom `shard_map(check_vma=True)` checks the declared varying/manual axes
rather than trusting unchecked output placement. Named DDP instead performs
one explicit bounded-bucket synchronization after backward and disables VMA
transposition so implicit per-parameter reductions cannot be sunk into encoder
replay.

FSDP parameter materialization is separately configurable as `model` or
`layer`. The latter bounds full-parameter live ranges but repeats layer
communication for every GradCache encoder replay; it is therefore a capacity
choice whose throughput depends strongly on the device interconnect.

Two- and four-GPU DDP acceptance is recorded in
[`distributed-grad-cache-modernvbert-20260814`](../benchmarks/results/distributed-grad-cache-modernvbert-20260814/README.org).
The named/custom sharding and physical FSDP profile is recorded in
[`fsdp-modernvbert-20260819`](../benchmarks/results/fsdp-modernvbert-20260819/README.org).

### Process-local input across JAX processes

Initialize JAX distributed before any device query or computation. Each process
then constructs only its local query and document payload rows. The relation
retains one local query axis and the complete global document axis, so a
positive may live on another process without replicating token or media
payloads:

```python
from representax.tasks.retrieval import process_local_retrieval_batch

local_batch = process_local_retrieval_batch(
    query=local_queries,
    document=local_documents,
    positive_mask=local_queries_to_global_documents,
)
batch = plan.place_process_local_batch(local_batch)
state = plan.place_replicated(state)
key = plan.place_replicated(key)
result = step(state, batch, key)
```

`place_process_local_batch` uses
`jax.make_array_from_process_local_data`; no process needs another process's
input records on its host. Replicated state and keys use the same constructor
when the mesh is not fully addressable from one process. The compiled GradCache
program itself is unchanged.

The accepted process-boundary result uses two JAX processes with two RTX 4090s
each and includes a query whose only positive is owned by the other process:
[`multiprocess-grad-cache-modernvbert-20260814`](../benchmarks/results/multiprocess-grad-cache-modernvbert-20260814/README.org).
This is a one-physical-host validation of process-local construction and JAX
collectives, not evidence for inter-host transport or failure behavior.

`build_train_step` keeps state donation off by default because its generic API
cannot prevent a caller from comparing, retrying, or branching from the input
state after an update. A linear training loop may set `donate_state=True` even
with asynchronous Orbax checkpoints: Orbax V1 completes its blocking
device-to-host snapshot before the save call returns, then writes that
independent snapshot in the background.

## Algorithm

For each query and document role, the native JAX path:

1. pads the execution batch to a static number of chunks;
2. encodes those chunks with `lax.scan`;
3. wraps the scan body in `jax.checkpoint` with `nothing_saveable`;
4. evaluates the canonical MNR row formula in bounded score-matrix tiles; and
5. lets JAX's transposition replay one encoder chunk at a time during backward.

Only the compact representations remain live across the encoder scan and only
one score-row tile remains live across the objective scan. Explicit
per-role and per-chunk PRNG keys make a replay of a stochastic encoder use the
same random values as its graphless forward. The resulting parameter gradient
then enters the same clipping, finite-check, Optax update, and logging path used
by direct training.

This is the compact JAX form of the three-stage algorithm in Gao et al.:
graphless representations, representation cotangents from the full contrastive
loss, then encoder replay with those cotangents. It avoids PyTorch backward
hooks and avoids maintaining a second cached MNR formula.

## Reference and performance contract

The correctness oracle is direct Representax MNR. Tests compare the loss,
metrics, parameter gradients through their global norm and clipped update,
optimizer state, and updated parameters. Separate replay tests cover stochastic
encoders and partial padded chunks.

The external reference and target to beat is Sentence Transformers'
`CachedMultipleNegativesRankingLoss`, pinned by version or Git revision in each
benchmark artifact. A matched comparison must use the same checkpoint, inputs,
pooling, scale, precision, optimizer, logical batch, negative population, and
encoder chunk geometry. It reports compile-plus-first execution separately from
warmup and steady-state measurements, along with the OOM boundary, throughput,
framework allocator peak, and process GPU-memory peak.

Performance regressions warn rather than masquerading as numerical failures.
Acceptance requires Representax to fit every batch in the declared matched
matrix, improve steady-state throughput, and avoid a material peak-memory
regression on the primary workload.

The first pinned result is
[`grad-cache-modernvbert-20260813`](../benchmarks/results/grad-cache-modernvbert-20260813/README.org).
At sequence length 512 and chunk four, native JAX is faster at every matched
batch from 8 through 1,024 and keeps its allocator peak nearly flat.  The final
scanned, chunk-two acceptance result is
[`grad-cache-final-20260813`](../benchmarks/results/grad-cache-final-20260813/README.org):
Representax is 16.1% to 26.7% faster through batch 1,024, uses less
process-visible device memory, and remains within 1.03% of the reference's live
allocator peak.  A persistent compilation-cache hit reduces compile plus first
execution from 38.5 to 3.6 seconds.

## Pallas boundary

Pallas is not used to orchestrate encoder replay: an arbitrary Equinox encoder
forward and backward remain ordinary JAX. If profiling shows that the
representation objective becomes material at very large logical batches, a
later custom VJP may fuse normalized similarity, streaming log-sum-exp, and
representation cotangents in Pallas. The existing row-tiled native-JAX path is
the fallback and numerical oracle it must beat.

## Sources

- [Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup](https://arxiv.org/abs/2101.06983)
- [Sentence Transformers GradCache engine](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/base/losses/gradcache.py)
- [Sentence Transformers cached MNR](https://github.com/huggingface/sentence-transformers/blob/main/sentence_transformers/sentence_transformer/losses/cached_multiple_negatives_ranking.py)
- [Original GradCache JAX transform](https://github.com/luyug/GradCache/blob/main/src/grad_cache/cachex/functional.py)
