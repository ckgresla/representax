"""Exercise the actual hard-negative recipe and its rectangular cached updates."""

import importlib
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from tests.tasks.test_late_interaction import (
    _assert_array_trees_close,
    _TokenBatch,
    _TokenEncoder,
)

from representax.config import JobConfig
from representax.tasks.late_interaction import LateInteractionTask
from representax.tasks.retrieval import retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state


def _experiment():
    return importlib.import_module("experiments.12-late-interaction-convergence.run")


def test_logged_configuration_is_the_executable_configuration(monkeypatch):
    experiment = _experiment()
    monkeypatch.setattr(experiment, "GLOBAL_BATCH_SIZE", 128)
    monkeypatch.setattr(experiment, "GRAD_CACHE_MICRO_BATCH", 4)
    monkeypatch.setattr(experiment, "WARMUP_STEPS", 25)
    job = experiment.job_config(42)
    restored = JobConfig.model_validate_json(job.model_dump_json())
    record = experiment.contract(restored)
    assert restored == job
    assert restored.training.seed == 42
    assert (
        record["global_batch_size"] == restored.training.batch.micro_batch_size == 128
    )
    assert record["grad_cache_micro_batch_size"] == 4
    assert record["optimization"]["warmup_steps"] == 25
    assert restored.optimization.schedule.parameters["decay_steps"] == 1000
    assert restored.data.collate.target.endswith(":TripletCollator")
    assert restored.checkpointing.should_save(100, final=False)
    assert restored.checkpointing.should_save(250, final=False)


def test_worker_passes_the_original_python_job_to_training(tmp_path, monkeypatch):
    from experiments.preflights import late_interaction as late

    from representax import train
    from representax.train import job as job_module

    experiment = _experiment()
    monkeypatch.setattr(experiment, "DATA", tmp_path / "data")
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path / "output")
    data = experiment.DATA / "seed-7"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n")
    (experiment.DATA / "manifest.json").write_text(
        json.dumps(
            {"seed_files": {"7": {"sha256": late._sha256(data / "train.jsonl")}}}
        )
    )
    job = experiment.job_config(7)
    constructed = []

    def construct(seed):
        constructed.append(seed)
        return job

    class ReachedTraining(Exception):
        pass

    def capture(actual, *args, **kwargs):
        assert actual is job
        raise ReachedTraining

    monkeypatch.setattr(experiment, "job_config", construct)
    monkeypatch.setattr(experiment, "_git_state", lambda: {"commit": "test"})
    monkeypatch.setattr(
        experiment.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"")
    )
    monkeypatch.setattr(job_module, "load_model", lambda *a, **k: (object(), object()))
    monkeypatch.setattr(
        late, "_representax_evaluation", lambda *a, **k: {"metrics": {}}
    )
    monkeypatch.setattr(train, "run_job", capture)
    with pytest.raises(ReachedTraining):
        experiment.worker(7)
    assert constructed == [7]
    recorded = json.loads((experiment.OUTPUT / "runs/seed-7/launch.json").read_text())
    assert recorded["job"] == job.model_dump(mode="json")


def test_triplet_collator_retains_negatives_and_masks_duplicate_candidates():
    class Processor:
        def __call__(self, texts, *, route):
            return jnp.arange(len(texts))[:, None]

    collate = _experiment().TripletCollator(processor=Processor())
    batch = collate(
        [
            {"query": "q1", "positive": "p1", "negative": "p2"},
            {"query": "q2", "positive": "p2", "negative": "n2"},
        ]
    )
    np.testing.assert_array_equal(batch.positive_mask, [[1, 0, 0, 0], [0, 1, 1, 0]])
    np.testing.assert_array_equal(batch.document_valid, [1, 1, 0, 1])
    assert batch.query.shape[0] == 2
    assert batch.document.shape[0] == 4


def test_preparation_keeps_filtered_negatives_only_when_requested():
    from experiments.preflights.late_interaction import select_training_rows

    rows = [{"query": "q", "positive": "p", "negative": "n"}]
    assert "negative" not in select_training_rows(rows, count=1)[0]
    assert (
        select_training_rows(rows, count=1, negative_field="negative")[0]["negative"]
        == "n"
    )


def test_asymmetric_hard_negative_grad_cache_matches_direct_update():
    model = _TokenEncoder(key=jax.random.key(7))
    optimizer = optax.adamw(learning_rate=3e-6, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    batch = retrieval_batch(
        query=_TokenBatch(
            jax.random.normal(jax.random.key(8), (4, 3, 4)),
            jnp.ones((4, 3), dtype=bool),
        ),
        document=_TokenBatch(
            jax.random.normal(jax.random.key(9), (8, 5, 4)),
            jnp.ones((8, 5), dtype=bool),
        ),
        positive_mask=jnp.eye(4, 8, dtype=bool),
    )
    task = LateInteractionTask(temperature=0.02, symmetric=False)
    direct = build_train_step(task, optimizer, max_grad_norm=1.0, donate_state=False)
    cached = build_train_step(
        task,
        optimizer,
        max_grad_norm=1.0,
        donate_state=False,
        execution=GradCache(
            query_chunk_size=2, document_chunk_size=3, loss_row_chunk_size=2
        ),
    )
    actual = cached(state, batch, jax.random.key(10))
    expected = direct(state, batch, jax.random.key(10))
    _assert_array_trees_close(actual.metrics, expected.metrics)
    _assert_array_trees_close(actual.state, expected.state)


def test_fp32_maxsim_matches_manual_scoring():
    from experiments.preflights.late_interaction import _representax_maxsim

    queries = [np.array([[0.9991, 0.042]], dtype=np.float32)]
    documents = [np.array([[0.9992, 0.041]], dtype=np.float32)]
    scores, metadata = _representax_maxsim(queries, documents, score_dtype="float32")
    np.testing.assert_allclose(
        scores[0, 0], (queries[0] @ documents[0].T).item(), rtol=1e-6
    )
    assert metadata["score_dtype"] == "float32"
