"""Domain configuration and parameter-role tests."""

from typing import Any

import jax
import pytest
from pydantic import ValidationError

from representax.config import (
    BatchConfig,
    ComponentConfig,
    CustomShardingConfig,
    DDPConfig,
    FSDPConfig,
    GradCacheConfig,
    JobConfig,
    LoggingConfig,
    MeshConfig,
    ModelConfig,
    OptimizationConfig,
    ParameterRole,
    PartitionRuleConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks import build_task
from representax.tasks.modifiers import (
    AdaptiveLayerModifierConfig,
    MatryoshkaModifierConfig,
    MatryoshkaTask,
)
from representax.tasks.pairwise import CosineRegressionConfig, PairwiseConfig
from representax.tasks.retrieval import MNRConfig, MNRTask, RetrievalConfig
from representax.train import GradCache, build_loss_execution, scientific_fingerprint


def _training(**overrides: Any) -> TrainingConfig:
    values: dict[str, Any] = {
        "global_batch_size": 64,
        "max_steps": 100,
        "seed": 7,
        "mesh": MeshConfig(axis_shapes=(4,), axis_names=("data",)),
        "batch": BatchConfig(
            micro_batch_size=64,
            gradient_accumulation_steps=1,
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


def test_mesh_config_preserves_logical_axis_names_for_sharding():
    training = _training(
        mesh=MeshConfig(
            axis_shapes=(4, 2),
            axis_names=("fsdp", "tensor"),
        ),
        sharding=DDPConfig(axis="fsdp"),
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


def test_named_and_custom_sharding_configs_round_trip():
    ddp = _job(training=_training(sharding=DDPConfig(axis="data")))
    fsdp = _job(
        training=_training(
            sharding=FSDPConfig(
                data_axis="data",
                minimum_parameter_elements=1024,
            )
        )
    )
    custom = _job(
        training=_training(
            mesh=MeshConfig(
                axis_shapes=(2, 2),
                axis_names=("data", "model"),
            ),
            sharding=CustomShardingConfig(
                data_axis="data",
                parameter_axes=("model",),
                parameter_rules=(
                    PartitionRuleConfig(
                        pattern=r"\.layers\..*\.weight$",
                        axes=("model", None),
                    ),
                ),
            ),
        )
    )

    restored_ddp = JobConfig.model_validate_json(ddp.model_dump_json())
    restored_fsdp = JobConfig.model_validate_json(fsdp.model_dump_json())
    restored_custom = JobConfig.model_validate_json(custom.model_dump_json())

    assert isinstance(restored_ddp.training.sharding, DDPConfig)
    assert isinstance(restored_fsdp.training.sharding, FSDPConfig)
    assert restored_fsdp.training.sharding.resolved_parameter_axis == "data"
    assert isinstance(restored_custom.training.sharding, CustomShardingConfig)
    assert restored_custom.training.sharding.parameter_rules[0].axes == (
        "model",
        None,
    )


def test_custom_sharding_rejects_invalid_rules():
    with pytest.raises(ValidationError, match="regular expression"):
        PartitionRuleConfig(pattern="[", axes=("model",))
    with pytest.raises(ValidationError, match="cannot reuse"):
        PartitionRuleConfig(
            pattern="weight",
            axes=("model", ("tensor", "model")),
        )
    with pytest.raises(ValidationError, match="at least one parameter rule"):
        CustomShardingConfig(parameter_axes=("model",), parameter_rules=())
    with pytest.raises(ValidationError, match="absent from parameter_axes"):
        _training(
            sharding=CustomShardingConfig(
                parameter_axes=("data",),
                parameter_rules=(
                    PartitionRuleConfig(pattern="weight", axes=("model",)),
                ),
            )
        )
    with pytest.raises(ValidationError, match="absent from the mesh"):
        _training(sharding=FSDPConfig(parameter_axis="model"))


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
        "loss_modifiers",
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
    assert set(execution) == {"data", "training"}
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
                gradient_accumulation_steps=1,
            ),
            grad_cache=GradCacheConfig(micro_batch_size=2),
            activation_rematerialization="selective",
        )
    )
    changed_science = _job(
        training=_training(
            global_batch_size=128,
            batch=BatchConfig(
                micro_batch_size=128,
                gradient_accumulation_steps=1,
            ),
        )
    )

    assert scientific_fingerprint(retuned_execution) == scientific_fingerprint(baseline)
    assert scientific_fingerprint(changed_science) != scientific_fingerprint(baseline)


def test_gradient_accumulation_is_capability_gated_by_loss_semantics():
    with pytest.raises(ValidationError, match="does not decompose exactly"):
        _job(
            training=_training(
                batch=BatchConfig(
                    micro_batch_size=16,
                    gradient_accumulation_steps=4,
                )
            )
        )

    job = _job(
        task=PairwiseConfig(),
        loss=CosineRegressionConfig(),
        training=_training(
            batch=BatchConfig(
                micro_batch_size=16,
                gradient_accumulation_steps=4,
            )
        ),
    )

    assert job.training.batch.gradient_accumulation_steps == 4


def test_structured_task_loss_and_grad_cache_build_runtime_objects():
    loss = MNRConfig(
        scale=7.0,
        symmetric=True,
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
    assert task.symmetric
    assert task.negative_scope == "global"
    assert isinstance(loss_execution, GradCache)
    assert loss_execution.query_chunk_size == 4
    assert loss_execution.resolved_document_chunk_size == 8
    assert loss_execution.resolved_loss_row_chunk_size == 2


def test_loss_modifiers_round_trip_and_build_as_scientific_job_config():
    modifier = MatryoshkaModifierConfig(
        dimensions=(4, 2),
        weights=(1.0, 0.5),
    )
    job = _job(loss_modifiers=(modifier,))

    restored = JobConfig.model_validate_json(job.model_dump_json())
    task = build_task(
        restored.task,
        restored.loss,
        modifiers=restored.loss_modifiers,
    )

    assert restored.loss_modifiers == (modifier,)
    assert isinstance(task, MatryoshkaTask)
    assert task.dimensions == (4, 2)
    assert job.parameters(ParameterRole.SCIENTIFIC)["loss_modifiers"] == [
        {
            "kind": "matryoshka",
            "dimensions": [4, 2],
            "weights": [1.0, 0.5],
            "dimensions_per_step": -1,
        }
    ]


def test_matryoshka_supports_grad_cache_but_adaptive_layers_do_not():
    job = _job(
        loss_modifiers=(MatryoshkaModifierConfig(dimensions=(4, 2)),),
        training=_training(grad_cache=GradCacheConfig(micro_batch_size=2)),
    )
    task = build_task(job.task, job.loss, modifiers=job.loss_modifiers)
    execution = build_loss_execution(job.training.grad_cache)

    execution.validate(task)
    with pytest.raises(
        ValidationError,
        match="loss modifier 'adaptive_layer' does not support training strategy",
    ):
        _job(
            loss_modifiers=(AdaptiveLayerModifierConfig(),),
            training=_training(grad_cache=GradCacheConfig(micro_batch_size=2)),
        )
