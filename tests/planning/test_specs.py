"""Domain configuration and parameter-role tests."""

from typing import Any

import jax
import pytest
from pydantic import ValidationError

from representax.config import (
    BatchConfig,
    ComponentConfig,
    GradCacheConfig,
    JobConfig,
    LoggingConfig,
    MeshConfig,
    ModelConfig,
    OptimizationConfig,
    ParameterRole,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks import build_task
from representax.tasks.retrieval import MNRConfig, MNRTask, RetrievalConfig
from representax.train import GradCache, build_loss_execution, scientific_fingerprint


def _training(**overrides: Any) -> TrainingConfig:
    values: dict[str, Any] = {
        "global_batch_size": 64,
        "max_steps": 100,
        "seed": 7,
        "mesh": MeshConfig(axis_shapes=(4,), axis_names=("data",)),
        "batch": BatchConfig(
            micro_batch_size=4,
            gradient_accumulation_steps=4,
        ),
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _job(**overrides: Any) -> JobConfig:
    values: dict[str, Any] = {
        "name": "test-job",
        "model": ModelConfig(target="tests.models.ToyEncoder"),
        "task": RetrievalConfig(),
        "loss": MNRConfig(),
        "optimization": OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-3},
            )
        ),
        "data": mix(source("file:///tmp/data.jsonl", map="tests.data.identity")),
        "training": _training(),
    }
    values.update(overrides)
    return JobConfig(**values)


@pytest.mark.parametrize("policy", ["none", "selective", "full"])
def test_training_config_accepts_activation_rematerialization_policies(policy):
    training = _training(activation_rematerialization=policy)

    assert training.activation_rematerialization == policy


def test_training_config_rejects_unknown_activation_rematerialization_policy():
    with pytest.raises(
        ValueError,
        match="Input should be 'none', 'selective' or 'full'",
    ):
        _training(activation_rematerialization="automatic")


def test_mesh_config_names_logical_axes_without_assigning_sharding_semantics():
    training = _training(
        mesh=MeshConfig(
            axis_shapes=(4, 2),
            axis_names=("fsdp", "tensor"),
        ),
    )

    assert training.mesh.device_count == 8
    assert training.mesh.axis_names == ("fsdp", "tensor")


def test_mesh_config_unpacks_directly_into_jax_make_mesh(monkeypatch):
    calls = []
    sentinel = object()

    def make_mesh(axis_shapes, axis_names):
        calls.append((axis_shapes, axis_names))
        return sentinel

    monkeypatch.setattr(jax, "make_mesh", make_mesh)
    config = MeshConfig(
        axis_shapes=(128, 4),
        axis_names=("fsdp", "tensor"),
    )

    assert jax.make_mesh(**config.model_dump()) is sentinel
    assert calls == [((128, 4), ("fsdp", "tensor"))]


def test_job_round_trips_registered_configs_and_is_frozen():
    job = _job(loss=MNRConfig(scale=9.0, symmetric=True))

    restored = JobConfig.model_validate_json(job.model_dump_json())

    assert restored == job
    assert isinstance(restored.task, RetrievalConfig)
    assert isinstance(restored.loss, MNRConfig)
    with pytest.raises(ValidationError, match="frozen"):
        restored.training.seed = 9  # ty: ignore[invalid-assignment]


def test_parameter_roles_project_domain_config_without_parallel_trees():
    job = _job(
        training=_training(
            grad_cache=GradCacheConfig(micro_batch_size=8),
        ),
        logging=LoggingConfig(console_every=5),
    )

    scientific = job.parameters(ParameterRole.SCIENTIFIC)
    execution = job.parameters(ParameterRole.EXECUTION)

    assert set(scientific) == {
        "data",
        "loss",
        "model",
        "optimization",
        "task",
        "training",
    }
    assert scientific["training"] == {
        "global_batch_size": 64,
        "max_steps": 100,
        "seed": 7,
    }
    assert set(execution) == {"training"}
    training_execution = execution["training"]
    assert isinstance(training_execution, dict)
    assert training_execution["grad_cache"] == {
        "micro_batch_size": 8,
        "query_micro_batch_size": None,
        "document_micro_batch_size": None,
        "loss_row_chunk_size": None,
    }
    assert "logging" not in scientific
    assert "logging" not in execution


def test_scientific_fingerprint_ignores_execution_only_changes():
    baseline = _job()
    retuned_execution = _job(
        training=_training(
            mesh=MeshConfig(axis_shapes=(8,), axis_names=("data",)),
            batch=BatchConfig(
                micro_batch_size=2,
                gradient_accumulation_steps=4,
            ),
            grad_cache=GradCacheConfig(micro_batch_size=2),
            activation_rematerialization="selective",
            prefetch_depth=8,
        )
    )
    changed_science = _job(
        training=_training(
            global_batch_size=128,
            batch=BatchConfig(
                micro_batch_size=8,
                gradient_accumulation_steps=4,
            ),
        )
    )

    assert scientific_fingerprint(retuned_execution) == scientific_fingerprint(baseline)
    assert scientific_fingerprint(changed_science) != scientific_fingerprint(baseline)


def test_structured_task_loss_and_grad_cache_build_runtime_objects():
    loss = MNRConfig(
        scale=7.0,
        symmetric=True,
        dimensions=(2, 4),
        dimension_weights=(1.0, 2.0),
        negative_scope="global",
    )
    task = build_task(RetrievalConfig(), loss)
    loss_execution = build_loss_execution(
        GradCacheConfig(
            micro_batch_size=4,
            document_micro_batch_size=8,
            loss_row_chunk_size=2,
        )
    )

    assert isinstance(task, MNRTask)
    assert task.scale == 7.0
    assert task.dimensions == (2, 4)
    assert isinstance(loss_execution, GradCache)
    assert loss_execution.query_chunk_size == 4
    assert loss_execution.resolved_document_chunk_size == 8
    assert loss_execution.resolved_loss_row_chunk_size == 2
