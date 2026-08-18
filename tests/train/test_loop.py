"""End-to-end single-device training-loop tests."""

import json
import threading
import time

import jax
import jax.numpy as jnp
import optax
import pytest

from representax.config import DataConfig, EvaluationConfig, EvaluatorConfig
from representax.data import build_grain_iterator, mix, source
from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask
from representax.train import (
    EvaluationRunner,
    LoggingConfig,
    MetricRecord,
    RunLogger,
    StepMetrics,
    StepResult,
    TrainState,
    build_train_step,
    init_train_state,
    run_training,
)
from tests.train.toy_retrieval import (
    TOY_BATCH_SIZE,
    TOY_FEATURE_DIMENSION,
    TOY_OUTPUT_DIMENSION,
    TOY_STEPS,
    build_toy_retrieval_batches,
    toy_job_config,
)


class ClosingBatches:
    def __init__(self, values):
        self._iterator = iter(values)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.closed = True


class RecordingReporter:
    def __init__(self):
        self.rows = []
        self.closed = False

    def write(self, row):
        self.rows.append(row)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class KeyPresenceEvaluator:
    name = "deterministic"

    def evaluate_batch(self, model, batch, *, key=None):
        del model, batch
        return jnp.asarray(key is None, dtype=jnp.float32)

    def initialize(self):
        return 0.0

    def accumulate(self, accumulator, output):
        return accumulator + float(output)

    def finalize(self, accumulator):
        return {"valid/deterministic/key_is_none": accumulator}


def _state() -> TrainState:
    model = DenseEncoder(2, 2, key=jax.random.key(0), normalize=False)
    return TrainState(
        model=model,
        optimizer_state=optax.EmptyState(),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def _job(*, logging=None, checkpointing=None, **kwargs):
    return toy_job_config(
        logging=LoggingConfig() if logging is None else logging,
        checkpointing=checkpointing,
        **kwargs,
    )


def _step(state, batch, key):
    del key
    return StepResult(
        state=TrainState(
            model=state.model,
            optimizer_state=state.optimizer_state,
            step=state.step + 1,
        ),
        metrics=StepMetrics(
            loss=jnp.mean(batch),
            task={"batch_mean": jnp.mean(batch)},
            gradient_global_norm=jnp.asarray(2.0),
            clipped_gradient_global_norm=jnp.asarray(1.0),
            update_global_norm=jnp.asarray(0.1),
            numeric_finite=jnp.asarray(True),
            skipped_update=jnp.asarray(False),
        ),
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_loop_writes_every_metric_and_closes_batches(tmp_path, capsys):
    batches = ClosingBatches([jnp.asarray([1.0, 3.0]), jnp.asarray([2.0, 4.0])])
    reporter = RecordingReporter()
    result = run_training(
        state=_state(),
        step=_step,
        batches=batches,
        job=_job(
            global_batch_size=2,
            max_steps=2,
            seed=17,
            logging=LoggingConfig(console_every=2),
        ),
        run_directory=tmp_path / "run",
        reporters=(reporter,),
    )

    assert result.completed_iterations == 2
    assert int(result.state.step) == 2
    assert batches.closed is True
    assert reporter.closed is True
    assert "train iteration=2 step=2" in capsys.readouterr().out

    run = json.loads((tmp_path / "run" / "run.json").read_text())
    metrics = _read_jsonl(tmp_path / "run" / "metrics.jsonl")
    events = _read_jsonl(tmp_path / "run" / "events.jsonl")
    assert run["status"] == "completed"
    assert run["completed_iterations"] == 2
    assert run["scientific"]["training"]["global_batch_size"] == 2
    assert run["execution"]["data"]["prefetch_buffer_size"] == 16
    assert run["config"]["logging"] == {
        "console_every": 2,
        "reporter_queue_size": 16,
    }
    assert [row["iteration"] for row in metrics] == [1, 2]
    assert metrics == [row for row in events if row["category"] == "metric"]
    assert len(reporter.rows) == len(events)
    assert metrics[0]["metrics"]["perf/compilation_and_first_step_seconds"] is not None
    assert "perf/compilation_and_first_step_seconds" not in metrics[1]["metrics"]
    assert metrics[0]["metrics"]["train/batch_mean"] == 2.0
    assert all("/" in name for row in metrics for name in row["metrics"])
    assert [row["event"] for row in events].count("executable_first_use_started") == 1
    assert events[-1]["event"] == "training_finished"


def test_training_loop_records_exhaustion_as_a_real_failure(tmp_path):
    batches = ClosingBatches([jnp.asarray([1.0, 3.0])])
    with pytest.raises(RuntimeError, match="batch source exhausted"):
        run_training(
            state=_state(),
            step=_step,
            batches=batches,
            job=_job(
                global_batch_size=2,
                max_steps=2,
                seed=17,
            ),
            run_directory=tmp_path / "failed-run",
        )

    run = json.loads((tmp_path / "failed-run" / "run.json").read_text())
    events = _read_jsonl(tmp_path / "failed-run" / "events.jsonl")
    assert batches.closed is True
    assert run["status"] == "failed"
    assert run["completed_iterations"] == 1
    assert events[-1]["event"] == "training_failed"
    assert events[-1]["error_type"] == "RuntimeError"


def test_training_loop_rejects_grain_batch_size_drift(tmp_path):
    artifact = source("memory://toy", map=_identity)
    batches = build_grain_iterator(
        mix(artifact, shuffle=False),
        batch_size=2,
        resolvers={"memory": _resolver},
        mappers={artifact.mapper: _identity},
    )

    with pytest.raises(ValueError, match="global_batch_size differs"):
        run_training(
            state=_state(),
            step=_step,
            batches=batches,
            job=_job(
                global_batch_size=4,
                max_steps=1,
                seed=17,
            ),
            run_directory=tmp_path / "must-not-exist",
        )

    assert not (tmp_path / "must-not-exist").exists()


def _skipped_step(state, batch, key):
    del batch
    return StepResult(
        state=state,
        metrics=StepMetrics(
            loss=jnp.asarray(jnp.nan),
            task={"key": key},
            gradient_global_norm=jnp.asarray(jnp.nan),
            clipped_gradient_global_norm=jnp.asarray(jnp.nan),
            update_global_norm=jnp.asarray(0.0),
            numeric_finite=jnp.asarray(False),
            skipped_update=jnp.asarray(True),
        ),
    )


def test_skipped_updates_advance_deterministic_iteration_keys(tmp_path):
    def run(name):
        result = run_training(
            state=_state(),
            step=_skipped_step,
            batches=ClosingBatches([jnp.asarray([1.0, 3.0]), jnp.asarray([1.0, 3.0])]),
            job=_job(
                global_batch_size=2,
                max_steps=2,
                seed=19,
                logging=LoggingConfig(console_every=10),
            ),
            run_directory=tmp_path / name,
        )
        return result, _read_jsonl(tmp_path / name / "metrics.jsonl")

    first_result, first = run("first")
    second_result, second = run("second")

    assert first_result.completed_iterations == 2
    assert int(first_result.state.step) == 0
    assert int(second_result.state.step) == 0
    assert [row["optimizer_step"] for row in first] == [0, 0]
    assert first[0]["metrics"]["train/key"] != first[1]["metrics"]["train/key"]
    assert [row["metrics"]["train/key"] for row in first] == [
        row["metrics"]["train/key"] for row in second
    ]


def test_training_evaluation_uses_deterministic_inference(tmp_path):
    validation_source = source("memory://validation", map=_identity)
    evaluation = EvaluationConfig(
        data=DataConfig(recipe=mix(validation_source, shuffle=False)),
        batch_size=2,
        evaluators=(EvaluatorConfig(name="deterministic"),),
        on_start=True,
        on_end=False,
        primary_metric="valid/deterministic/key_is_none",
        primary_metric_mode="max",
        save_best=False,
    )
    job = _job(
        global_batch_size=2,
        max_steps=1,
        seed=19,
        logging=LoggingConfig(console_every=10),
    ).model_copy(update={"evaluation": evaluation})

    run_training(
        state=_state(),
        step=_step,
        batches=ClosingBatches([jnp.asarray([1.0, 3.0])]),
        job=job,
        run_directory=tmp_path / "deterministic-evaluation",
        evaluation_runners=(EvaluationRunner(KeyPresenceEvaluator()),),
        evaluation_batches=lambda: [jnp.asarray([2.0, 4.0])],
    )

    metrics = _read_jsonl(tmp_path / "deterministic-evaluation" / "metrics.jsonl")
    evaluation_metrics = [row for row in metrics if row["event"] == "evaluation"]
    assert evaluation_metrics[0]["metrics"]["valid/deterministic/key_is_none"] == 1.0


def _identity(record):
    return record


def _resolver(_artifact):
    return [1.0, 2.0]


@pytest.mark.runtime
def test_grain_recipe_drives_compiled_updates_end_to_end(tmp_path):
    batches = build_toy_retrieval_batches(seed=23)
    model = DenseEncoder(
        TOY_FEATURE_DIMENSION,
        TOY_OUTPUT_DIMENSION,
        key=jax.random.key(0),
    )
    optimizer = optax.adamw(learning_rate=0.03, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    result = run_training(
        state=state,
        step=build_train_step(
            MNRTask(scale=5.0, symmetric=True),
            optimizer,
        ),
        batches=batches,
        job=_job(
            global_batch_size=TOY_BATCH_SIZE,
            max_steps=TOY_STEPS,
            seed=23,
            logging=LoggingConfig(console_every=TOY_STEPS),
        ),
        run_directory=tmp_path / "grain-run",
    )

    assert result.completed_iterations == TOY_STEPS
    assert int(result.state.step) == TOY_STEPS
    assert isinstance(result.state.model, DenseEncoder)
    assert not jnp.array_equal(
        result.state.model.projection.weight,
        model.projection.weight,
    )
    metrics = _read_jsonl(tmp_path / "grain-run" / "metrics.jsonl")
    assert len(metrics) == TOY_STEPS
    assert all(row["metrics"]["train/numeric_finite"] for row in metrics)
    losses = [row["metrics"]["train/loss"] for row in metrics]
    assert losses[-1] < losses[0] * 0.5


def test_reporter_writes_do_not_block_the_training_thread(tmp_path):
    class BlockingReporter(RecordingReporter):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()

        def write(self, row):
            self.entered.set()
            self.release.wait(timeout=5)
            super().write(row)

    reporter = BlockingReporter()
    logger = RunLogger(
        tmp_path / "async-run",
        manifest={},
        reporters=(reporter,),
        queue_size=2,
    )
    try:
        logger.metrics(
            MetricRecord(
                iteration=1,
                values={"train/loss": 1.0, "train/skipped_update": False},
            )
        )
        assert reporter.entered.wait(timeout=2)
        started = time.perf_counter()
        logger.metrics(
            MetricRecord(
                iteration=2,
                values={"train/loss": 0.5, "train/skipped_update": False},
            )
        )
        assert time.perf_counter() - started < 0.1
    finally:
        reporter.release.set()
        logger.close()

    assert [row["iteration"] for row in reporter.rows] == [1, 2]
