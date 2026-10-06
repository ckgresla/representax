# Training

Representax keeps the compiled numerical program small and the orchestration
explicit:

1. `build_train_step` closes over a task and Optax transformation and returns
   one compiled, model-neutral Equinox update;
2. the internal host loop owns Grain iteration, device-placement dispatch,
   reporting, checkpointing, and failure cleanup;
3. `run_job` is the sole public end-to-end entry point and builds both layers,
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

`training.precision` explicitly separates FP32 master/checkpoint/Optax state
from transient compute and activation dtypes. The BF16 mixed preset restores
representations, reductions, gradients, losses, and metrics to FP32. Training
and validation consume the same policy. See the
[mixed-precision contract](precision.md) for model-use boundaries, GradCache,
FSDP communication, and numerical acceptance.

`training.adapter` is an optional scientific model transformation. The first
accepted recipe, `QuantizedLoRAConfig`, replaces selected native linear
projections with packed INT4 frozen bases and FP32 low-rank adapters before the
optimizer and compiled step are constructed. A model-shaped trainable filter
then drives ordinary Optax initialization, differentiation, DDP/FSDP planning,
checkpointing, and export; there is no adapter-specific trainer. See
[low-bit adapters](adapters.md) for the representation, export contract, and
physical acceptance evidence.

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

Named DDP, FSDP, and custom layouts execute as global JAX programs whose
communication follows from declared state, input, output, and primitive-use
shardings. Shared linear and normalization primitives request replicated
parameters only at their exact use sites; reverse-mode derives the matching
sharded gradient communication. These are execution choices: they do not change
the task, loss, global batch, or optimizer semantics, and no model-specific FSDP
hook is required.

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

Data execution is fail-closed when configured. While `next()` is blocked,
`data_wait_heartbeat_seconds` emits repeated `data_wait_heartbeat` lifecycle
events; `data_wait_timeout_seconds` raises `DataStarvationError`, closes the
iterator, and records the failed run. Fatal deadlines use POSIX main-thread
signals so they can interrupt the blocked call rather than merely notice it
afterward.

`perf/data_wait_seconds`, `perf/preprocess_seconds`,
`perf/host_batch_bytes`, `perf/prefetch_ready_batches`, and
`perf/prefetch_capacity` describe the host input path.
`perf/placement_enqueue_seconds` and `perf/step_dispatch_seconds` measure host
enqueue cost, not device execution. One bounded asynchronous completion
observer emits `perf/device_input_idle_seconds_lower_bound` only when the
immediately preceding update is known complete before the next batch arrives;
zero is deliberately inconclusive. It adds neither a per-step barrier nor an
unbounded queue of retained states. Exact steady-state throughput, utilization,
and memory remain dedicated performance-lane measurements with explicit
synchronization and profiler evidence.

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

Metric values use service-neutral W&B-style namespaces: `train/loss` for
optimization, `valid/loss` for validation coupled to training, `eval/loss` for
standalone or test evaluation, and `perf/...` for systems measurements. Every
metric row contains `iteration` and a `metrics` mapping; training rows also
contain `optimizer_step`. Downstream reporters consume these without renaming
fields.

`RunLogger` owns one bounded, ordered worker queue. That worker materializes
device metrics once, appends the local JSONL source of truth, and fans the same
row out through the small `Reporter` protocol. Disk reporting is always
available. The optional W&B reporter is initialized and driven on this worker,
records terminal success or failure, and requires `representax[wandb]`; it does
not enter the task, evaluator, or compiled-step APIs. A full queue applies
bounded backpressure instead of allowing unbounded host memory growth.

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

`EvaluationRunner` is the shared offline and in-training evaluation boundary.
It caches JAX executables by batch structure and shape and keeps exact host-side
reducers for corpus metrics. In-training runners emit `valid/...`; standalone
runners remap the same reducer outputs to `eval/...` without changing their
values.
Compatible evaluators share one compiled traversal; a one-output pipeline can
overlap device inference with reduction of the preceding batch. The inventory
covers loss, similarity, classification, regression/MSE, triplet, reranking,
reward, paraphrase mining, information retrieval, and LeJEPA collapse
diagnostics. BEIR-format query, corpus, and qrel sources map into the generic
information-retrieval evaluator; NanoBEIR is a revision-pinned example rather
than a separate evaluator. `EvaluationConfig`
controls start/end and periodic cadence, a bounded number of batches, the
primary metric, and min/max selection. Training performs evaluation on a
separate Grain iterator, leaving the resumable training cursor untouched.

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

W&B uses its native step axis rather than logging iteration as a metric.
Training rows additionally expose `train/optimizer_step`; standalone evaluation
rows do not invent an optimizer coordinate and instead report elapsed work under
`perf/...`. Static run configuration records the JAX backend, process index and
world size, local and global device counts, visible accelerator models, and the
configured training mesh/sharding policy. W&B's system monitor remains
responsible for time-varying accelerator utilization and memory telemetry.

## Deliberately deferred

TensorBoard and additional reporter adapters remain optional future additions.
Grain owns lazy reading, mapping, batching, prefetch, and iterator state.
Task-owned collation is the bridge from data examples to the compiled batch
contract.

Single-host DDP, FSDP, hybrid data/model meshes, and arbitrary model-path
partition rules execute through the same `build_train_step` boundary.
Two- and four-device topology gates cover exact updates, StableHLO collectives,
and asynchronous Orbax restore; physical GPUs additionally cover complete
ModernVBERT updates, memory placement, NCCL execution, and a 1.9795B-parameter
two-GPU capacity point. Physical multi-host acceptance remains deferred until
suitable hardware is available.

Representax will also benchmark JAX `Ref`-based state mutation against canonical
functional Equinox/Optax plus buffer donation before changing the model-state
contract.
