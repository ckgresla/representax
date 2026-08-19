# Training

Representax keeps the compiled numerical program small and the orchestration
explicit:

1. `build_train_step` closes over a task and Optax transformation and returns
   one compiled, model-neutral Equinox update;
2. `run_training` owns Grain iteration, device-placement dispatch, reporting,
   checkpointing, and failure cleanup on the host.
3. `run_job` is the canonical configured entry point and builds both layers,
   validation, model selection, and inference export from one `JobConfig`.

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

User-facing configurations are frozen Pydantic models organized by the domains
users actually configure. `JobConfig` owns model, task, loss, optimization,
data, training, logging, checkpointing, evaluation, and export configs.
`TrainingConfig` combines
the scientific training values and their efficiency-only realization: global
batch, maximum steps, seed, logical mesh, physical batch decomposition,
optional GradCache, activation rematerialization, and donation. Input threading
and prefetch live in each `DataConfig`, because training and validation sources
may need different host-side execution plans.

Scientific and execution are field roles rather than parallel configuration
trees. `Scientific[T]` and `Execution[T]` metadata can mark one value or a whole
nested config. Generic projection code derives the scientific fingerprint and
the execution search space without duplicating values. Logging and
checkpointing are persisted operational contracts but are not hyperparameter
roles. Study-specific labels such as tuned, fixed, or nuisance remain a future
overlay over parameter paths; they are not intrinsic field properties.

Task and loss are distinct registered configs. `RetrievalConfig` selects the
query/document batch semantics; `MNRConfig` owns scale, symmetry, Matryoshka
dimensions, and local versus global negative scope. The loss registry declares
compatible tasks and training strategies. A populated `training.grad_cache`
therefore selects exact cached differentiation only when the configured loss
advertises that capability; `None` selects direct differentiation.

For losses that reduce independently over examples, configured gradient
accumulation reshapes one logical batch into equal microbatches, evaluates them
with one compiled `lax.scan`, averages their gradients, clips once, and performs
one Optax update. The loss registry capability-gates this path. Cross-example
objectives such as MNR reject ordinary accumulation because it would change the
negative population; exact GradCache remains their bounded-memory execution.

`MeshConfig` stores portable logical axis shapes and names; concrete JAX
`Device` objects are never serialized. The runtime materializes an automatic
JAX mesh from those values and the devices assigned to the job.
`training.sharding` is a discriminated union of named DDP, named FSDP, and exact
custom partition rules. All three resolve to one model-shaped `ShardingPlan`
containing batch, parameter, gradient, Optax-state, and output layouts. The host
loop validates the scientific global batch against the resolved data-axis size
and the model-ready Grain source. Packing remains absent until a task-, data-,
and model-compatible segment/masking contract is implemented.

Named DDP and default full-model FSDP execute as global JAX programs whose
communication follows from their declared input, output, and model-call
shardings. The optional `materialization_boundary="layer"` mode uses bounded
explicit parameter gathers to shorten the full-parameter live range; it
requires a model to name its layer stack and fails during plan construction
when that capability is absent. These are execution choices: they do not change
the task, loss, global batch, or optimizer semantics.

These models are declarative, validated, serializable, and compatible with
Hydra-Zen composition and CLI overrides. They contain no live JAX mesh, Equinox model,
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
only after publication is complete. Resume validates the scientific parameter
projection, the data contract, and model/optimizer structures; restores training
and iterator state; truncates post-checkpoint log rows; and continues with the
same next batch and random key as an uninterrupted run.

## Evaluation, model selection, and inference publication

`EvaluationRunner` is the shared offline and in-training loss-evaluation
boundary. It caches JAX executables by batch structure and shape, aggregates
unequal final batches by the task's exact loss denominator (or example count),
and emits service-neutral `valid/...` metrics. `EvaluationConfig` controls
start/end and periodic cadence, a bounded
number of batches, the primary metric, and min/max selection. Training performs
evaluation on a separate Grain iterator, leaving the resumable training cursor
untouched.

Representax computes the metric; Orbax receives the scalar mapping and composes
latest-N with best-N preservation. Representax publishes an additional durable
`best` pointer only after the checkpoint is complete. At the end of a successful
job it restores only the selected model when necessary and atomically publishes
`final-model/`:

- `native/job.json` plus `native/model.eqx` reconstruct any configured Equinox
  model without optimizer or loader state;
- the optional `huggingface/` directory retains source tokenizer/config assets,
  writes the trained adapter state, and must reload with exact tensors before
  publication; and
- `manifest.json` plus `REPRESENTAX_COMPLETE` fingerprint the complete bundle.

The same `EvaluationRunner` is available through `representax.train.evaluate`
for offline evaluation of a loaded inference bundle.

## Deliberately deferred

The configured runtime does not yet provide concrete W&B/TensorBoard adapters.
Grain owns lazy reading, mapping, batching, prefetch, and iterator state.
Task-owned collation is the bridge from data examples to the compiled batch
contract.

Single-host DDP, FSDP, hybrid data/model meshes, and arbitrary model-path
partition rules execute through the same `build_sharded_train_step` boundary.
Two- and four-device topology gates cover exact updates, StableHLO collectives,
and asynchronous Orbax restore; physical GPUs additionally cover complete
ModernVBERT updates, memory placement, and NCCL execution. Physical multi-host
acceptance remains deferred until suitable hardware is available. ModernVBERT
supports both whole-model and scanned-layer FSDP materialization. The accepted
four-GPU profile records the resulting DDP/FSDP throughput-memory frontier;
larger sequence/model capacity sweeps remain open paper evidence.

Representax will also benchmark JAX `Ref`-based state mutation against canonical
functional Equinox/Optax plus buffer donation before changing the model-state
contract.
