"""Asynchronous checkpoint publication and exact-resume tests."""

from __future__ import annotations

import json
import threading

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models import DenseEncoder
from representax.planning import ScientificSpec
from representax.tasks.retrieval import MNRTask
from representax.train import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointWriteError,
    IncompleteCheckpointError,
    RunLogger,
    TrainState,
    build_train_step,
    make_train_state,
    run_training,
    science_fingerprint,
    training_checkpointables,
    validate_complete_checkpoint,
)
from tests.train.toy_retrieval import (
    TOY_BATCH_SIZE,
    TOY_FEATURE_DIMENSION,
    TOY_OUTPUT_DIMENSION,
    TOY_STEPS,
    build_toy_retrieval_batches,
)


class _MixedPrecisionModel(eqx.Module):
    trainable_master: jax.Array
    frozen_compute: jax.Array
    compute_dtype: object = eqx.field(static=True)


class _ControlledResponse:
    def __init__(self, path, release, *, error=None):
        self.path = path
        self.release = release
        self.error = error
        self.entered = threading.Event()

    def result(self):
        self.entered.set()
        self.release.wait(timeout=10)
        if self.error is not None:
            raise self.error
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "_CHECKPOINT_METADATA").touch()
        return True


class _ControlledCheckpointer:
    def __init__(self, root, *, failures=()):
        self.root = root
        self.failures = set(failures)
        self.responses = {}
        self.closed = False

    def save_checkpointables_async(self, iteration, *_args, **_kwargs):
        response = _ControlledResponse(
            self.root / str(iteration),
            threading.Event(),
            error=(RuntimeError("disk failed") if iteration in self.failures else None),
        )
        self.responses[iteration] = response
        return response

    def load_checkpointables(self, *_args, **_kwargs):
        raise AssertionError("restore is not used by this fake")

    def close(self):
        self.closed = True


def _state(input_dimension=2, output_dimension=2, optimizer=None):
    model = DenseEncoder(
        input_dimension,
        output_dimension,
        key=jax.random.key(0),
        normalize=False,
    )
    optimizer = (
        optax.adamw(learning_rate=0.01) if optimizer is None else optimizer
    )
    return make_train_state(model, optimizer)


def _checkpointables(iteration, state=None):
    state = _state() if state is None else state
    return training_checkpointables(
        state=state,
        iteration=iteration,
        rng=jax.random.key(7),
        data_state={"next_index": iteration * 2},
        logging_cursor={
            "events_bytes": iteration * 10,
            "metrics_bytes": iteration * 5,
            "sequence": iteration,
        },
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_checkpoint_manager_keeps_one_save_in_flight_and_backpressures(tmp_path):
    events = []
    checkpointer = _ControlledCheckpointer(tmp_path / "run" / "checkpoints")
    manager = CheckpointManager(
        tmp_path / "run",
        science_fingerprint="sha256:test",
        asynchronous=True,
        event=lambda name, **fields: events.append((name, fields)),
        checkpointer=checkpointer,
    )

    first = manager.save(1, _checkpointables(1))
    assert first.asynchronous is True
    assert checkpointer.responses[1].entered.wait(timeout=2)

    second_finished = threading.Event()

    def save_second():
        manager.save(2, _checkpointables(2))
        second_finished.set()

    thread = threading.Thread(target=save_second)
    thread.start()
    assert not second_finished.wait(timeout=0.1)
    assert any(name == "checkpoint_backpressure_started" for name, _ in events)

    checkpointer.responses[1].release.set()
    assert second_finished.wait(timeout=2)
    assert checkpointer.responses[2].entered.wait(timeout=2)
    checkpointer.responses[2].release.set()
    thread.join(timeout=2)
    manager.close()

    assert checkpointer.closed is True
    assert [name for name, _ in events].count("checkpoint_saved") == 2
    assert all(
        fields["duration_seconds"] >= 0
        for name, fields in events
        if name
        in {
            "checkpoint_snapshot_finished",
            "checkpoint_backpressure_finished",
        }
    )
    assert (tmp_path / "run" / "checkpoints" / "latest").is_file()


def test_checkpoint_manager_surfaces_background_write_failure(tmp_path):
    checkpointer = _ControlledCheckpointer(
        tmp_path / "run" / "checkpoints",
        failures={1},
    )
    manager = CheckpointManager(
        tmp_path / "run",
        science_fingerprint="sha256:test",
        asynchronous=True,
        checkpointer=checkpointer,
    )
    manager.save(1, _checkpointables(1))
    checkpointer.responses[1].release.set()

    with pytest.raises(CheckpointWriteError, match="disk failed"):
        manager.wait()
    manager.close()


def test_incomplete_checkpoint_is_rejected(tmp_path):
    incomplete = tmp_path / "checkpoints" / "3"
    incomplete.mkdir(parents=True)
    (incomplete / "_CHECKPOINT_METADATA").touch()

    with pytest.raises(IncompleteCheckpointError, match="incomplete checkpoint"):
        validate_complete_checkpoint(incomplete)


def test_run_logger_truncates_post_checkpoint_tail_on_resume(tmp_path):
    run = tmp_path / "run"
    manifest = {
        "task": "test/task",
        "global_batch_size": 2,
        "max_steps": 3,
        "seed": 7,
    }
    logger = RunLogger(run, manifest=manifest)
    logger.event("durable", iteration=1)
    cursor = logger.cursor()
    logger.event("stale", iteration=2)
    logger.finish("failed")
    logger.close()

    resumed = RunLogger(run, manifest=manifest, resume_cursor=cursor)
    resumed.event("training_resumed", iteration=1)
    resumed.close()

    rows = _read_jsonl(run / "events.jsonl")
    assert [row["event"] for row in rows] == ["durable", "training_resumed"]
    assert [row["sequence"] for row in rows] == [0, 1]


def _assert_array_trees_equal(left, right):
    left_leaves = [leaf for leaf in jax.tree.leaves(left) if eqx.is_array(leaf)]
    right_leaves = [leaf for leaf in jax.tree.leaves(right) if eqx.is_array(leaf)]
    assert len(left_leaves) == len(right_leaves)
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.runtime
def test_orbax_retains_latest_complete_checkpoints_and_restores_state(tmp_path):
    science = ScientificSpec(
        task="retrieval/mnr",
        global_batch_size=2,
        max_steps=3,
        seed=29,
    )
    initial = _state(input_dimension=4)
    manager = CheckpointManager(
        tmp_path / "run",
        science_fingerprint=science_fingerprint(science),
        keep=2,
        asynchronous=True,
    )
    for iteration in range(1, 4):
        manager.save(iteration, _checkpointables(iteration, initial))
    manager.close()

    root = tmp_path / "run" / "checkpoints"
    assert not (root / "1").exists()
    assert (root / "2" / "REPRESENTAX_COMPLETE").is_file()
    assert (root / "3" / "REPRESENTAX_COMPLETE").is_file()

    resumed = CheckpointManager(
        tmp_path / "run",
        science_fingerprint=science_fingerprint(science),
        keep=2,
    )
    restored = resumed.restore_training_state(initial)
    assert restored.iteration == 3
    assert restored.data_state == {"next_index": 6}
    assert int(restored.state.step) == 0
    resumed.close()

    different_model = DenseEncoder(
        4,
        2,
        key=jax.random.key(0),
        normalize=True,
    )
    incompatible_state = make_train_state(
        different_model,
        optax.adamw(learning_rate=0.01),
    )
    incompatible_structure = CheckpointManager(
        tmp_path / "run",
        science_fingerprint=science_fingerprint(science),
        keep=2,
    )
    with pytest.raises(IncompleteCheckpointError, match="structure differs"):
        incompatible_structure.restore_training_state(incompatible_state)
    incompatible_structure.close()

    incompatible = CheckpointManager(
        tmp_path / "run",
        science_fingerprint="sha256:different-science",
        keep=2,
    )
    with pytest.raises(IncompleteCheckpointError, match="scientific specification"):
        incompatible.restore_training_state(initial)
    incompatible.close()


@pytest.mark.runtime
def test_orbax_preserves_mixed_training_dtypes(tmp_path):
    model = _MixedPrecisionModel(
        trainable_master=jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        frozen_compute=jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4),
        compute_dtype=jnp.bfloat16,
    )
    state = TrainState(
        model=model,
        optimizer_state={"moment": jnp.ones_like(model.trainable_master)},
        step=jnp.asarray(5, dtype=jnp.int32),
    )
    science = ScientificSpec(
        task="test/mixed-precision",
        global_batch_size=2,
        max_steps=6,
        seed=31,
    )
    manager = CheckpointManager(
        tmp_path / "mixed-run",
        science_fingerprint=science_fingerprint(science),
        keep=1,
        asynchronous=False,
    )
    manager.save(
        5,
        training_checkpointables(
            state=state,
            iteration=5,
            rng=jax.random.key(31),
            data_state={"next_index": 10},
            logging_cursor={
                "events_bytes": 100,
                "metrics_bytes": 80,
                "sequence": 5,
            },
        ),
    )
    manager.close()

    resumed = CheckpointManager(
        tmp_path / "mixed-run",
        science_fingerprint=science_fingerprint(science),
        keep=1,
    )
    restored = resumed.restore_training_state(state)
    resumed.close()

    assert restored.state.model.trainable_master.dtype == jnp.float32
    assert restored.state.model.frozen_compute.dtype == jnp.bfloat16
    assert restored.state.optimizer_state["moment"].dtype == jnp.float32
    _assert_array_trees_equal(restored.state, state)


@pytest.mark.runtime
def test_donated_training_with_async_orbax_resumes_exactly(tmp_path):
    science = ScientificSpec(
        task="retrieval/mnr",
        global_batch_size=TOY_BATCH_SIZE,
        max_steps=TOY_STEPS,
        seed=29,
    )
    optimizer = optax.adamw(learning_rate=0.03, weight_decay=0.0)
    def fresh_initial():
        return _state(
            input_dimension=TOY_FEATURE_DIMENSION,
            output_dimension=TOY_OUTPUT_DIMENSION,
            optimizer=optimizer,
        )

    train_step = build_train_step(
        MNRTask(scale=5.0, symmetric=True),
        optimizer,
        donate_state=True,
    )
    uninterrupted = run_training(
        state=fresh_initial(),
        step=train_step,
        batches=build_toy_retrieval_batches(seed=29),
        science=science,
        run_directory=tmp_path / "uninterrupted",
    )

    calls = 0

    def crash_after_checkpoint(state, batch, key):
        nonlocal calls
        calls += 1
        # Iteration four has been snapshotted asynchronously and iteration five
        # has donated that state before this simulated process interruption.
        if calls == 6:
            raise RuntimeError("simulated interruption")
        return train_step(state, batch, key)

    run = tmp_path / "resumed"
    checkpoint = CheckpointConfig(every=4, keep=2, asynchronous=True)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_training(
            state=fresh_initial(),
            step=crash_after_checkpoint,
            batches=build_toy_retrieval_batches(seed=29),
            science=science,
            run_directory=run,
            checkpoint=checkpoint,
        )

    resumed = run_training(
        # A real restart constructs a fresh abstract/state template. Reusing the
        # donated object from the interrupted run is invalid JAX ownership.
        state=fresh_initial(),
        step=train_step,
        batches=build_toy_retrieval_batches(seed=29),
        science=science,
        run_directory=run,
        checkpoint=checkpoint,
        resume=True,
    )

    assert resumed.resumed is True
    assert resumed.completed_iterations == TOY_STEPS
    _assert_array_trees_equal(resumed.state, uninterrupted.state)
    uninterrupted_metrics = _read_jsonl(
        tmp_path / "uninterrupted" / "metrics.jsonl"
    )
    resumed_metrics = _read_jsonl(run / "metrics.jsonl")
    assert [row["iteration"] for row in resumed_metrics] == list(
        range(1, TOY_STEPS + 1)
    )
    np.testing.assert_array_equal(
        [row["loss"] for row in resumed_metrics],
        [row["loss"] for row in uninterrupted_metrics],
    )
    assert resumed_metrics[-1]["loss"] < resumed_metrics[0]["loss"] * 0.5
    events = _read_jsonl(run / "events.jsonl")
    assert [row["event"] for row in events].count("training_resumed") == 1
    assert not any(row["event"] == "training_failed" for row in events)
    assert [row["sequence"] for row in events] == list(range(len(events)))
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["resume_count"] == 1
    assert manifest["checkpoint"]["every"] == 4
    assert "error" not in manifest
    assert "error_type" not in manifest
