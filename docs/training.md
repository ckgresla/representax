# Training

Representax keeps the compiled numerical program small and the orchestration
explicit:

1. `build_train_step` closes over a task and Optax transformation and returns
   one compiled, model-neutral Equinox update;
2. `run_training` owns Grain iteration, device-placement dispatch, reporting,
   checkpointing, and failure cleanup on the host.

This is the ordinary JAX boundary. Python is useful for I/O and lifecycle work;
the repeated forward, backward, finite-update gate, and optimizer update belong
inside the compiled step.

The design follows JAX's
[Training Cookbook](https://docs.jax.dev/en/latest/the-training-cookbook.html):
JIT calls and `jax.device_put` remain asynchronously dispatched, Grain prefetch
can overlap accelerator execution, telemetry from iteration N is consumed only
after iteration N+1 has been dispatched, and host-only timing uses Python rather
than enqueueing incidental `jax.numpy` work outside the train step.

## Configuration boundary

User-facing configurations are frozen Pydantic models. `TrainingConfig` is the
single argument consumed by the host loop and contains:

- `ScientificConfig`: trajectory-defining task identity, global batch, maximum
  steps, random seed, negative scope, and numerical tolerance;
- `ExecutionConfig`: topology-dependent mechanisms such as mesh axes,
  per-device batch, accumulation, GradCache chunks, rematerialization, packing,
  prefetch, and donation;
- `RuntimeConfig`: host mechanics such as console cadence and reporter queue
  capacity; and
- `CheckpointConfig`: cadence, retention, final-save policy, and asynchronous
  publication.

There is no second semantics object: `TrainingConfig.scientific` is the single
training-level scientific contract. Model, optimizer, task, and data choices
remain explicit sibling members of `RunConfig`; execution may change only when
it preserves the scientific contract.

These models are declarative, validated, serializable, and compatible with
Hydra-Zen composition and CLI overrides. They contain no live Equinox model,
Optax transformation, Grain iterator, or arbitrary runtime object. Builders
read them while constructing those objects. Configurations do not pass through
`jax.jit` and do not need an Equinox mirror.

## Host loop and synchronization

For every iteration, the host loop:

1. requests the next already-collated Grain batch;
2. enqueues device placement;
3. derives a deterministic key by folding the absolute iteration into the run
   seed;
4. dispatches the compiled update; and
5. queues its metrics for asynchronous reporting.

The loop deliberately does not call `block_until_ready` for placement or every
step. It synchronizes the first execution of each new shape/dtype signature so
compilation plus first execution is recorded honestly. Thereafter, the reporter
worker requests all metric leaves for an iteration in one `jax.device_get` while
the host continues dispatching later work. This follows JAX's asynchronous
execution model and lets host reporting overlap accelerator work.

`perf/placement_enqueue_seconds` and `perf/step_dispatch_seconds` measure host
enqueue cost, not device execution. Exact steady-state throughput and memory are
measured by the dedicated performance lane with explicit synchronization.

An attempted iteration and an accepted optimizer update are distinct. The
compiled step forms one ordinary Equinox/Optax proposed state, then uses Optax's
tree-selection primitive to retain the previous state when any forward,
loss-metric, or gradient value is non-finite. A skipped update still advances
the absolute iteration and therefore its random key, but not `TrainState.step`.

## Run artifacts and reporters

Local files are the source of truth:

- `run.json` contains configuration, data contracts, fingerprints, and terminal
  status;
- `events.jsonl` is the complete ordered lifecycle and metric stream; and
- `metrics.jsonl` is the exact metric-row projection of that stream.

Metric values use service-neutral W&B-style namespaces: `train/loss`,
`valid/loss`, and `perf/...`. A metric row contains `iteration`,
`optimizer_step`, and a `metrics` mapping, so a future W&B reporter can call
`wandb.log(row["metrics"], step=row["optimizer_step"])` without renaming fields.

`RunLogger` owns one bounded, ordered worker queue. That worker materializes
device metrics once, appends the local JSONL source of truth, and fans the same
row out through the small `Reporter` protocol. Disk reporting is implemented;
tests exercise another reporter; W&B and TensorBoard adapters are deferred. A
full queue applies bounded backpressure instead of allowing unbounded host
memory growth.

At a checkpoint boundary, `RunLogger.cursor()` drains the queue, flushes and
fsyncs both local streams, flushes downstream reporters, and returns byte and
sequence offsets. Normal metric reporting remains asynchronous; checkpoint
durability is the intentional synchronization boundary.

## Asynchronous checkpoints and exact resume

Orbax owns array snapshotting and storage. Representax owns the training-state,
publication, and compatibility contract. A checkpoint contains:

- the Equinox model and Optax state;
- the accepted optimizer step and absolute attempted iteration;
- the base random key used for per-iteration `fold_in`;
- Grain's native iterator state; and
- durable byte offsets plus sequence position for both append-only logs.

The associated data fingerprint covers the complete artifact recipe and source
revisions, declared mapper paths, digests of the resolved mapper and resolver
modules, the batch mapper implementation, batch size and remainder policy, and
the Grain version. Changing one of these values rejects resume before restoring
the saved iterator cursor. This prevents an old cursor from being interpreted
under changed preprocessing. Grain's native `set_state` seeks to the saved
iterator position; Representax does not spin through or repeat earlier batches.

Representax uses Orbax V1's sequence-oriented training `Checkpointer` and
`save_checkpointables_async`. Only one save may be in flight. A newer save waits
only if its predecessor is still outstanding, emitting explicit backpressure
events. Orbax takes a donation-safe host snapshot before its asynchronous
response returns, so a linear training loop can donate the snapshotted device
state while storage and publication finish in the background.

A checkpoint is restorable only after Orbax metadata, a fingerprinted
`checkpoint.json`, and `REPRESENTAX_COMPLETE` agree. The `latest` pointer moves
only after publication is complete. Resume validates the scientific configuration, the
data contract, and model/optimizer structures; restores training and iterator
state; truncates post-checkpoint log rows; and continues with the same next batch
and random key as an uninterrupted run.

## Deliberately deferred

The current host loop does not yet construct arbitrary named sharding plans,
run validation, or provide concrete W&B/TensorBoard adapters. Grain owns lazy
reading, mapping, batching, prefetch, and iterator state. Task-owned collation
is the bridge from data examples to the compiled batch contract.

MNR can use exact GradCache execution through the single-device
`build_train_step` boundary or the accepted two- and four-device data-parallel
boundary. Physical multi-host execution, FSDP-style model-state sharding, and
arbitrary sharding configurations remain separately scoped roadmap work.

The existing `DataParallel` plan already uses a named mesh, replicated state,
batch-axis sharding, `jax.make_array_from_process_local_data` for process-local
rows, and explicit collective semantics. Completing the cookbook's high-
performance sharding picture means wiring `ExecutionConfig` into placement and
step construction, initializing state directly into its declared sharding, and
adding measured FSDP and tensor/hybrid plans. Representax will also benchmark
JAX `Ref`-based state mutation against canonical functional Equinox/Optax plus
buffer donation before changing the model-state contract.
