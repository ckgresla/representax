"""Artifact-to-inference-model training job acceptance test."""

from __future__ import annotations

import json
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax import load_inference_bundle
from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    EmbeddingSimilarityEvaluatorConfig,
    EvaluationConfig,
    EvaluatorConfig,
    ExportConfig,
    JobConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.core import Encoder, ModelBundle, Route, encode
from representax.data import mix, source
from representax.precision import resolve_precision_policy
from representax.tasks import build_task
from representax.tasks.pairwise import (
    CosineRegressionConfig,
    PairwiseConfig,
    pairwise_batch,
)
from representax.train import (
    build_batches,
    build_collate,
    build_model,
    build_model_bundle,
    evaluate,
    run_job,
)


class RematerializedModule(eqx.Module):
    weight: jax.Array
    rematerialization: str = eqx.field(static=True)


def rematerialized_model(*, key, rematerialization):
    return RematerializedModule(
        weight=jax.random.normal(key, (2, 2)),
        rematerialization=rematerialization,
    )


class _Processor:
    def batch(self, model, artifacts, *, route=Route.GENERIC, seed=None):
        del model, route, seed
        return tuple(artifacts)


def bundled_model(*, key):
    return ModelBundle(
        model=rematerialized_model(key=key, rematerialization="none"),
        processor=_Processor(),
    )


class BundleAwareCollator:
    def __init__(self, *, bundle):
        self.bundle = bundle

    def __call__(self, examples):
        return self.bundle.batch(examples)


def identity(record):
    return record


def collate_pairwise_records(examples: Sequence[dict]):
    return pairwise_batch(
        left=jnp.asarray([example["left"] for example in examples]),
        right=jnp.asarray([example["right"] for example in examples]),
        labels=jnp.asarray([example["label"] for example in examples]),
    )


def _write_pairs(path, *, count: int) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for index in range(count):
            angle = 0.3 + 0.17 * index
            row = {
                "left": [float(np.cos(angle)), float(np.sin(angle))],
                "right": [float(np.cos(angle + 0.1)), float(np.sin(angle + 0.1))],
                "label": 0.85 + 0.01 * (index % 3),
            }
            stream.write(json.dumps(row) + "\n")


def _data(path) -> DataConfig:
    return DataConfig(
        distribution=mix(source(str(path), map=identity), shuffle=False),
        collate=ComponentConfig(target="tests.train.test_job.collate_pairwise_records"),
        num_threads=0,
        prefetch_buffer_size=0,
    )


def test_job_model_builder_injects_activation_rematerialization():
    model = build_model(
        ModelConfig(target="tests.train.test_job.rematerialized_model"),
        key=jax.random.key(3),
        activation_rematerialization="selective",
    )

    assert isinstance(model, RematerializedModule)
    assert model.rematerialization == "selective"


def test_job_builder_preserves_bundled_processor_for_data_collation():
    bundle = build_model_bundle(
        ModelConfig(target="tests.train.test_job.bundled_model"),
        key=jax.random.key(7),
    )
    collate = build_collate(
        DataConfig(
            distribution=mix(
                source("memory://unused", map="tests.train.test_job.identity")
            ),
            collate=ComponentConfig(target="tests.train.test_job.BundleAwareCollator"),
        ),
        bundle=bundle,
    )

    assert isinstance(bundle.model, RematerializedModule)
    assert bundle.model.weight.shape == (2, 2)
    assert collate is not None
    assert collate(("first", "second")) == ("first", "second")


@pytest.mark.runtime
def test_run_job_trains_evaluates_selects_and_exports_from_disk(tmp_path):
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    _write_pairs(train_path, count=12)
    _write_pairs(valid_path, count=4)
    run_directory = tmp_path / "run"
    job = JobConfig(
        name="disk-to-inference",
        model=ModelConfig(
            target="representax.models.DenseEncoder",
            parameters={"input_dimension": 2, "output_dimension": 2},
        ),
        task=PairwiseConfig(),
        loss=CosineRegressionConfig(),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 0.01, "weight_decay": 0.0},
            ),
            max_gradient_norm=None,
        ),
        data=_data(train_path),
        training=TrainingConfig(
            global_batch_size=4,
            max_steps=3,
            seed=31,
            batch=BatchConfig(
                micro_batch_size=2,
                gradient_accumulation_steps=2,
            ),
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(every=1, keep=2, asynchronous=True),
        evaluation=EvaluationConfig(
            data=_data(valid_path),
            batch_size=4,
            evaluators=(
                EvaluatorConfig(),
                EmbeddingSimilarityEvaluatorConfig(
                    similarity_functions=("cosine",),
                    main_similarity="cosine",
                ),
            ),
            every_steps=1,
            on_start=True,
            on_end=True,
            save_best=True,
        ),
        export=ExportConfig(selection="best"),
    )

    result = run_job(job, run_directory)

    assert result.completed_iterations == 3
    assert result.best_iteration in {0, 1, 2, 3}
    assert result.best_metrics is not None
    assert "valid/loss" in result.best_metrics
    assert "valid/similarity/spearman_cosine" in result.best_metrics
    assert result.inference_bundle == run_directory / "final-model"
    assert (run_directory / "checkpoints" / "best").is_file()
    model, restored_job = load_inference_bundle(run_directory / "final-model")
    assert restored_job == job
    inputs = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    assert isinstance(model, Encoder)
    assert isinstance(result.selected_model, Encoder)
    np.testing.assert_array_equal(
        encode(model, inputs, route=Route.GENERIC),
        encode(result.selected_model, inputs, route=Route.GENERIC),
    )

    assert job.evaluation is not None
    offline = evaluate(
        model,
        build_task(job.task, job.loss),
        build_batches(job.evaluation.data, batch_size=4),
        precision=resolve_precision_policy(job.training.precision),
    )
    np.testing.assert_allclose(
        offline.metrics["valid/loss"],
        result.best_metrics["valid/loss"],
        rtol=1e-6,
    )
