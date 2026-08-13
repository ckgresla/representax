# Training

Representax has two training boundaries:

1. `build_train_step` closes over a task and Optax transformation and returns
   one compiled, model-neutral Equinox update;
2. `run_training` consumes model-ready batches and owns the ordinary
   single-device host loop.

This split is inspired by the parts of Lasso that held up well under real
training: the loop remains topology-neutral, its dependencies are narrow, and
logging describes the same update stream that actually ran.

## Loop contract

`ScientificSpec` owns semantics the runtime must not alter: task identity,
global batch size, total iterations, and random seed. `TrainingLoopConfig` owns
only host mechanics, currently console cadence. `CheckpointConfig` owns save
cadence, retention, final-save policy, and whether publication is asynchronous.
Grain batch sources expose their exact global batch size, which must match the
scientific value before the run directory is created.

For every iteration, the loop:

1. waits for the next already-collated batch;
2. places it on the default device and waits for placement to complete;
3. derives a deterministic key by folding the absolute iteration into the
   scientific seed;
4. calls the compiled task step and synchronizes before measuring it;
5. writes one structured metric record; and
6. optionally prints the same record at a separately configured cadence.

The first execution of every new batch shape/dtype signature is recorded as
`compilation_and_first_step_seconds`. It is deliberately not reported as
steady-state throughput. Later executions use `compiled_step_seconds` and
`examples_per_second`; end-to-end measurements include input wait and device
placement as well.

An attempted iteration and an accepted optimizer update are distinct. A
non-finite update leaves the model and Optax state unchanged, increments no
`TrainState.step`, writes the attempted iteration, and emits a
`nonfinite_update_skipped` event. Random keys still advance by absolute
iteration, so a skipped update cannot replay stochastic work.

## Run artifacts

Local files are the source of truth; external services are optional mirrors
through the small `EventSink` protocol.

- `run.json` contains the scientific loop inputs and terminal status;
- `events.jsonl` is the complete ordered lifecycle and metric stream; and
- `metrics.jsonl` is the exact metric-row projection of that stream.

Each row is flushed immediately. Successful runs end with `training_finished`;
exceptions, including premature iterator exhaustion, end with
`training_failed`, update `run.json` to `failed`, close the iterator and logger,
and are re-raised.

## Asynchronous checkpoints and exact resume

Orbax owns array snapshotting and storage. Representax owns the training-state,
publication, and resume contract. A checkpoint contains:

- the Equinox model and Optax state;
- the accepted optimizer step and absolute attempted iteration;
- the base random key used for per-iteration `fold_in`;
- Grain's iterator state; and
- byte offsets plus sequence position for both append-only JSONL streams.

Only one save may be in flight. Scheduling a checkpoint first waits for an
older save only when that save is still outstanding, emitting explicit
`checkpoint_backpressure_started` and `checkpoint_backpressure_finished`
events. Orbax snapshots the immutable JAX arrays, after which training may
continue while disk I/O and publication finish on a background thread.

The checkpoint is restorable only after Orbax metadata, a fingerprinted
`checkpoint.json`, and `REPRESENTAX_COMPLETE` all agree. The `latest` pointer is
atomically replaced only after that complete marker is durable. Background
write failures surface on the training thread at the next save or close.

Checkpoint snapshot and backpressure durations are lifecycle events; they are
not folded into `compiled_step_seconds` or reported as accelerator throughput.
This keeps compute timing honest without making checkpoint overhead invisible.

To resume, construct the same `ScientificSpec`, model/Optax state template, and
Grain source, task, and optimizer program, then call
`run_training(..., checkpoint=config, resume=True)`. Representax validates the
scientific fingerprint and state structures, restores all training and iterator
state, truncates log rows newer than the checkpoint cursor, and emits
`checkpoint_restored` followed by `training_resumed`. The next update therefore
uses the same batch and random key as uninterrupted training. Shape-dependent
compilation is measured again after process restart rather than misclassified
as steady-state work. Fingerprinting arbitrary Python task and optimizer
closures remains part of the deferred full recipe/configuration snapshot.

## Deliberately deferred

The current loop does not include distributed sharding, validation, external
experiment integrations, full configuration/environment snapshots, media
preprocessing, or a model registry. Grain owns lazy reading, mapping, batching,
prefetch, and iterator position. Task-owned collation is the only bridge from
data examples to the compiled batch contract.

Single-device MNR may use [exact GradCache execution](grad-cache.md) by passing a
step built with `execution=GradCache(...)` to this loop. Distributed GradCache
remains roadmap work.
