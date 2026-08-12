# Architecture

Representax separates four concerns that are often coupled in training
frameworks:

```text
upstream artifacts -> lazy data recipe -> task -> model capability
                                              |
scientific specification -> execution plan -> compiled training step
```

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

Hydra-Zen composes recipe values in Python. Grain will provide lazy random
access, deterministic mapping and mixing, sharding, packing, batching,
prefetching, and iterator checkpointing. Representax does not require users to
materialize an intermediate framework-specific dataset.

The run manifest will record the recipe fingerprint and source revisions. The
recipe and mapper code remain Git-tracked, so a repository revision identifies
the complete mapping program without duplicating source data.

## Scientific and execution specifications

`ScientificSpec` contains experiment semantics: objective, global batch,
negative scope, data mixture, limits, optimizer schedule, and numerical
requirements. `ExecutionPlan` contains topology-dependent mechanisms: mesh,
per-device batch, accumulation, GradCache chunks, rematerialization, packing,
prefetching, and buffer donation.

An execution planner may search only the latter space and must validate that
the resulting plan preserves the scientific specification. The protocol is
intentionally small enough for a future standalone `profilax` package to
implement it over arbitrary compiled JAX programs.

## Test contract

An integration is supported only when the relevant lanes pass:

- `unit`: contracts and numerical building blocks without upstream frameworks;
- `runtime`: compiled forward, backward, update, checkpoint, and resume;
- `parity`: optional upstream forward, gradient, update, and export comparisons;
- `distributed`: collective correctness and multi-host-compatible layouts; and
- `performance`: compile time, steady-state throughput, and peak-memory evidence.

Upstream PyTorch dependencies are confined to the optional parity environment.
They are not installed in ordinary Representax training environments.
