# Architecture

Representax separates four concerns that are often coupled in training
frameworks:

```text
upstream artifacts -> lazy data recipe -> task -> model capability
                                              |
scientific specification -> execution plan -> compiled training step
```

The ordinary single-device path adds a deliberately small host boundary:

```text
Grain recipe -> mapped examples -> static task batches -> device placement
                                                        -> compiled train step
                                                        -> local run records
                                                        -> async checkpoints
```

The trainer does not tokenize, decode media, interpret task examples, construct
models, or select objectives. It consumes model-ready batches and coordinates
only iteration, deterministic random keys, placement, execution, measurement,
and observability. See [Training](training.md).

## Models and tasks

Native model integrations are complete Equinox PyTrees. A model may implement
one or more capabilities: encoding, token-level features, scalar scoring, or
prediction heads. Tasks consume the capability they need and own their batch
interpretation, loss, metrics, and evaluation semantics.

Retrieval is the first concrete task. Query and document routing remains in the
retrieval task rather than being generalized away. The generic task protocol
does not require an encoder, leaving room for classification, pairwise and
listwise reward modeling, process reward modeling, distillation, and JEPA-style
self-supervision.

## Data

A data recipe points at immutable upstream artifacts and names the mapping code
that converts each raw record to a task example. A mixture is a sampling policy
over sources; one source is the one-element form of the same policy.

Hydra-Zen composes frozen Pydantic configuration values in Python and applies
typed CLI overrides. Grain provides lazy random access, deterministic mapping
and mixing, batching, prefetching, and native iterator checkpointing.
Representax does not require users to materialize an intermediate
framework-specific dataset.

The run manifest records the complete recipe and source revisions plus a data
fingerprint covering resolved mapper/resolver module digests, batch collation,
batching semantics, and the Grain version. The recipe and mapper code remain
Git-tracked without duplicating source data.

## Domain configuration and parameter roles

`JobConfig` is organized by model, task, loss, optimization, data, training,
logging, and checkpointing. `TrainingConfig` unifies global training semantics
with the efficiency parameters that realize them. Values or nested config sets
are annotated `Scientific[T]` or `Execution[T]`; generic projections form the
resume fingerprint and future Profilax search space without duplicating a
second scientific/execution object tree.

The immutable task registry associates each task identity with its Pydantic
config and batch type. A separate immutable loss registry associates loss
identity with its config, runtime builder, compatible task kinds, and supported
training strategies. This keeps retrieval semantics distinct from MNR and lets
`training.grad_cache` be optional but capability-gated.

Logical mesh config mirrors the serializable `jax.make_mesh` arguments: axis
shapes and names unpack directly into JAX, while concrete topology-bound devices
remain runtime state. Mesh names alone do not define array placement. Named DDP,
named FSDP, or ordered model-path rules resolve model-shaped parameter,
gradient, Optax-state, batch, and output `PartitionSpec` trees through one
`ShardingPlan`. The current loop validates global batch against the resolved
data-axis size and its model-ready Grain source instead of inferring semantics
from names such as `fsdp` or `tensor`.
Replicated DDP gradients and materialized FSDP parameters are coalesced into
bounded dtype/axis-compatible buckets. FSDP can materialize a whole model call
or one supported scanned layer at a time; the boundary is an execution
parameter whose throughput/memory tradeoff belongs in the future Profilax
search space.
Packing remains deferred until its segment IDs, position handling, attention
masking, and example-boundary preservation are implemented for compatible data,
models, and tasks.

Activation rematerialization is an explicit three-way execution choice rather
than a model change. See [Activation rematerialization](rematerialization.md)
for the policy contract and the measured ModernVBERT default.

An execution planner may search only the execution-annotated paths and must
validate that the resulting configuration preserves the scientific projection. The
protocol is intentionally small enough for a future standalone `profilax` package to
implement it over arbitrary compiled JAX programs.

## Test contract

Tests live in an out-of-tree hierarchy that mirrors the source domains. A model
family therefore keeps its native unit tests, upstream oracle, numerical parity,
compiled training integration, and performance comparison together. Test lanes
are orthogonal pytest markers rather than the primary directory structure.

An integration is supported only when the relevant lanes pass:

- `unit`: contracts and numerical building blocks without upstream frameworks;
- `runtime`: compiled forward, backward, update, checkpoint, and resume;
- `parity`: optional upstream forward, gradient, update, and export comparisons;
- `distributed`: collective correctness and multi-host-compatible layouts; and
- `performance`: compile time, steady-state throughput, and peak-memory evidence.

Upstream PyTorch dependencies are confined to the optional parity environment.
They are not installed in ordinary Representax training environments.

Numerical support is necessary but not sufficient for a production model
integration. On pinned reference hardware, matched native and upstream programs
must also compare compile time, synchronized steady-state throughput, and peak
device memory. The detailed measurement contract is in [Testing](testing.md).

Hugging Face checkpoint handling follows the model-owned adapter design in
[Hugging Face interoperability](hugging-face.md). Native Equinox modules are
the runtime; upstream implementations are references used to establish
numerical support.
