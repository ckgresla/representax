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
only host mechanics, currently console cadence and the initial iteration.
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

## Deliberately deferred

This first loop does not include distributed sharding, GradCache, validation,
checkpoint/resume, external experiment integrations, configuration snapshots,
media preprocessing, or a model registry. Grain owns lazy reading, mapping,
batching, and prefetch. Task-owned collation is the only bridge from data
examples to the compiled batch contract.
