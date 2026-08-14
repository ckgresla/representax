"""Task and loss registry contracts."""

from typing import Literal

import pytest
from pydantic import ValidationError

from representax.config import (
    BatchConfig,
    ComponentConfig,
    GradCacheConfig,
    JobConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from representax.data import mix, source
from representax.tasks import (
    BUILTIN_LOSSES,
    BUILTIN_TASKS,
    LossConfig,
    LossDefinition,
    TaskConfig,
    TaskDefinition,
)
from representax.tasks.retrieval import RetrievalBatch


class _CustomTaskConfig(TaskConfig):
    kind: Literal["test/custom"] = "test/custom"


class _CustomLossConfig(LossConfig):
    kind: Literal["test/loss"] = "test/loss"
    weight: float


def _job_data():
    return {
        "name": "custom",
        "model": ModelConfig(target="tests.Model"),
        "task": {"kind": "test/custom"},
        "loss": {"kind": "test/loss", "weight": 2.5},
        "optimization": OptimizationConfig(
            optimizer=ComponentConfig(target="optax.sgd")
        ),
        "data": mix(source("file:///tmp/data", map="tests.identity")),
        "training": TrainingConfig(
            global_batch_size=8,
            max_steps=2,
            seed=1,
            batch=BatchConfig(micro_batch_size=8),
        ),
    }


def test_extended_registries_parse_custom_configs_without_global_mutation():
    tasks = BUILTIN_TASKS.extended(
        TaskDefinition(
            kind="test/custom",
            config_type=_CustomTaskConfig,
            batch_type=dict,
        )
    )
    losses = BUILTIN_LOSSES.extended(
        LossDefinition(
            kind="test/loss",
            config_type=_CustomLossConfig,
            build=lambda config: (config.kind, config.weight),
            task_kinds=frozenset({"test/custom"}),
            training_strategies=frozenset({"direct"}),
        )
    )
    job = JobConfig.model_validate(
        _job_data(),
        context={"task_registry": tasks, "loss_registry": losses},
    )

    assert isinstance(job.task, _CustomTaskConfig)
    assert isinstance(job.loss, _CustomLossConfig)
    assert losses.build(job.loss) == ("test/loss", 2.5)
    assert "test/custom" not in BUILTIN_TASKS.definitions
    assert "test/loss" not in BUILTIN_LOSSES.definitions


def test_unknown_task_kind_is_a_configuration_error():
    data = _job_data()
    data["task"] = {"kind": "missing/task"}
    with pytest.raises(ValidationError, match="is not registered"):
        JobConfig.model_validate(data)


def test_unknown_loss_kind_is_a_configuration_error():
    data = _job_data()
    data["task"] = {"kind": "retrieval"}
    data["loss"] = {"kind": "missing/loss"}
    with pytest.raises(ValidationError, match="is not registered"):
        JobConfig.model_validate(data)


def test_grad_cache_is_rejected_when_loss_does_not_support_it():
    tasks = BUILTIN_TASKS.extended(
        TaskDefinition(
            kind="test/custom",
            config_type=_CustomTaskConfig,
            batch_type=dict,
        )
    )
    losses = BUILTIN_LOSSES.extended(
        LossDefinition(
            kind="test/loss",
            config_type=_CustomLossConfig,
            build=lambda config: config,
            task_kinds=frozenset({"test/custom"}),
            training_strategies=frozenset({"direct"}),
        )
    )
    data = _job_data()
    data["training"] = data["training"].model_copy(
        update={"grad_cache": GradCacheConfig(micro_batch_size=2)}
    )

    with pytest.raises(ValidationError, match="does not support training strategy"):
        JobConfig.model_validate(
            data,
            context={"task_registry": tasks, "loss_registry": losses},
        )


def test_builtin_registries_declare_batch_and_training_contracts():
    task = BUILTIN_TASKS.definition("retrieval")
    loss = BUILTIN_LOSSES.definition("mnr")

    assert task.batch_type is RetrievalBatch
    assert loss.task_kinds == frozenset({"retrieval"})
    assert loss.training_strategies == frozenset({"direct", "grad_cache"})
