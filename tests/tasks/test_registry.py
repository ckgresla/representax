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
    build_task,
)
from representax.tasks.classification import (
    PairClassificationConfig,
    SoftmaxClassificationConfig,
    SoftmaxClassificationTask,
)
from representax.tasks.contrastive_tension import (
    ContrastiveTensionConfig,
    ContrastiveTensionExamplesConfig,
    ContrastiveTensionInBatchConfig,
    ContrastiveTensionInBatchTask,
    ContrastiveTensionPairsConfig,
    ContrastiveTensionTask,
)
from representax.tasks.guided import GISTConfig, GISTTask, GuidedRetrievalConfig
from representax.tasks.mega_batch import (
    MegaBatchConfig,
    MegaBatchMarginConfig,
    MegaBatchMarginTask,
)
from representax.tasks.reconstruction import (
    DenoisingAutoEncoderConfig,
    DenoisingAutoEncoderTask,
    DenoisingConfig,
)
from representax.tasks.regularization import (
    GlobalOrthogonalRegularizationTask,
    GORConfig,
    RegularizationConfig,
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


def _build_custom_loss(task: TaskConfig, config: LossConfig):
    del task
    if not isinstance(config, _CustomLossConfig):
        raise TypeError("custom loss requires _CustomLossConfig")
    return config.kind, config.weight


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
            build=_build_custom_loss,
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
    assert losses.build(job.task, job.loss) == ("test/loss", 2.5)
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
            build=lambda task, config: config,
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


@pytest.mark.parametrize(
    ("task_config", "loss_config", "runtime_type"),
    (
        (GuidedRetrievalConfig(), GISTConfig(), GISTTask),
        (
            PairClassificationConfig(),
            SoftmaxClassificationConfig(),
            SoftmaxClassificationTask,
        ),
        (
            ContrastiveTensionPairsConfig(),
            ContrastiveTensionConfig(),
            ContrastiveTensionTask,
        ),
        (
            ContrastiveTensionExamplesConfig(),
            ContrastiveTensionInBatchConfig(),
            ContrastiveTensionInBatchTask,
        ),
        (
            RegularizationConfig(),
            GORConfig(),
            GlobalOrthogonalRegularizationTask,
        ),
        (
            DenoisingConfig(),
            DenoisingAutoEncoderConfig(pad_token_id=0),
            DenoisingAutoEncoderTask,
        ),
        (MegaBatchConfig(), MegaBatchMarginConfig(), MegaBatchMarginTask),
    ),
)
def test_extended_dense_configs_build_registered_runtime_tasks(
    task_config,
    loss_config,
    runtime_type,
):
    assert isinstance(build_task(task_config, loss_config), runtime_type)
