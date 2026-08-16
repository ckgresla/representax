"""Immutable registries for structured task and loss configurations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from representax.core import Route

from .config import LossConfig, TaskConfig
from .distillation.batch import (
    DistributionDistillationBatch,
    EmbeddingDistillationBatch,
    MarginDistillationBatch,
)
from .distillation.config import (
    DistributionDistillationConfig,
    DistributionKLLossConfig,
    EmbeddingDistillationConfig,
    EmbeddingDistillationLossConfig,
    MarginDistillationConfig,
    MarginMSELossConfig,
)
from .distillation.task import (
    DistributionDistillationTask,
    EmbeddingDistillationTask,
    MarginDistillationTask,
)
from .pairwise.batch import PairwiseBatch
from .pairwise.config import (
    AngleConfig,
    ContrastiveConfig,
    CoSENTConfig,
    CosineRegressionConfig,
    PairwiseConfig,
)
from .pairwise.task import AnglETask, ContrastiveTask, CoSENTTask, CosineRegressionTask
from .retrieval.batch import RetrievalBatch
from .retrieval.config import MNRConfig, RetrievalConfig
from .retrieval.mnr import MNRTask
from .triplet.batch import ExplicitTripletBatch, LabeledExamplesBatch
from .triplet.config import (
    BatchHardSoftMarginLossConfig,
    BatchTripletLossConfig,
    ExplicitTripletConfig,
    ExplicitTripletLossConfig,
    LabeledExamplesConfig,
)
from .triplet.task import BatchTripletTask, ExplicitTripletTask


def _index_definitions(definitions: Iterable[Any], *, label: str) -> Mapping[str, Any]:
    indexed = {}
    for definition in definitions:
        if definition.kind in indexed:
            raise ValueError(f"duplicate {label} kind {definition.kind!r}")
        indexed[definition.kind] = definition
    return MappingProxyType(indexed)


def _parse_registered(
    value: Any,
    *,
    base_type: type,
    definitions: Mapping[str, Any],
    label: str,
) -> Any:
    if isinstance(value, base_type):
        kind = value.kind
        try:
            definition = definitions[kind]
        except KeyError as error:
            raise ValueError(f"{label} kind {kind!r} is not registered") from error
        if not isinstance(value, definition.config_type):
            raise TypeError(
                f"{label} {kind!r} requires {definition.config_type.__name__}, "
                f"received {type(value).__name__}"
            )
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} config must be a mapping or {base_type.__name__}")
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise TypeError(f"{label} config must contain a string kind")
    try:
        definition = definitions[kind]
    except KeyError as error:
        raise ValueError(f"{label} kind {kind!r} is not registered") from error
    return definition.config_type.model_validate(value)


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """One task identity and its model-ready batch contract."""

    kind: str
    config_type: type[TaskConfig]
    batch_type: type

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("task kind must be non-empty")


class TaskRegistry:
    """Closed task definitions with explicit immutable extension."""

    def __init__(self, definitions: Iterable[TaskDefinition]) -> None:
        self._definitions: Mapping[str, TaskDefinition] = _index_definitions(
            definitions,
            label="task",
        )

    @property
    def definitions(self) -> Mapping[str, TaskDefinition]:
        return self._definitions

    def definition(self, kind: str) -> TaskDefinition:
        try:
            return self._definitions[kind]
        except KeyError as error:
            raise KeyError(f"task kind {kind!r} is not registered") from error

    def parse(self, value: Any) -> TaskConfig:
        return _parse_registered(
            value,
            base_type=TaskConfig,
            definitions=self._definitions,
            label="task",
        )

    def extended(self, *definitions: TaskDefinition) -> TaskRegistry:
        return TaskRegistry((*self._definitions.values(), *definitions))


@dataclass(frozen=True, slots=True)
class LossDefinition:
    """One loss identity, runtime builder, and compatibility contract."""

    kind: str
    config_type: type[LossConfig]
    build: Callable[[TaskConfig, LossConfig], Any]
    task_kinds: frozenset[str]
    training_strategies: frozenset[str]

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("loss kind must be non-empty")
        if not self.task_kinds:
            raise ValueError("a loss must support at least one task")
        if not self.training_strategies:
            raise ValueError("a loss must support at least one training strategy")


class LossRegistry:
    """Closed loss definitions with explicit immutable extension."""

    def __init__(self, definitions: Iterable[LossDefinition]) -> None:
        self._definitions: Mapping[str, LossDefinition] = _index_definitions(
            definitions,
            label="loss",
        )

    @property
    def definitions(self) -> Mapping[str, LossDefinition]:
        return self._definitions

    def definition(self, kind: str) -> LossDefinition:
        try:
            return self._definitions[kind]
        except KeyError as error:
            raise KeyError(f"loss kind {kind!r} is not registered") from error

    def parse(self, value: Any) -> LossConfig:
        return _parse_registered(
            value,
            base_type=LossConfig,
            definitions=self._definitions,
            label="loss",
        )

    def build(self, task: TaskConfig, config: LossConfig) -> Any:
        definition = self.definition(config.kind)
        if not isinstance(config, definition.config_type):
            raise TypeError(
                f"loss {config.kind!r} requires {definition.config_type.__name__}, "
                f"received {type(config).__name__}"
            )
        return definition.build(task, config)

    def extended(self, *definitions: LossDefinition) -> LossRegistry:
        return LossRegistry((*self._definitions.values(), *definitions))


def _build_mnr_task(task: TaskConfig, loss: LossConfig) -> MNRTask:
    if not isinstance(task, RetrievalConfig) or not isinstance(loss, MNRConfig):
        raise TypeError("mnr requires MNRConfig")
    return MNRTask(
        scale=loss.scale,
        symmetric=loss.symmetric,
        dimensions=loss.dimensions,
        dimension_weights=loss.dimension_weights,
        negative_scope=loss.negative_scope,
    )


def _pairwise_routes(task: TaskConfig) -> tuple[Route, Route]:
    if not isinstance(task, PairwiseConfig):
        raise TypeError("pairwise losses require PairwiseConfig")
    return task.left_route, task.right_route


def _build_cosine_regression_task(
    task: TaskConfig,
    loss: LossConfig,
) -> CosineRegressionTask:
    if not isinstance(loss, CosineRegressionConfig):
        raise TypeError("cosine_regression requires CosineRegressionConfig")
    left_route, right_route = _pairwise_routes(task)
    return CosineRegressionTask(left_route=left_route, right_route=right_route)


def _build_contrastive_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ContrastiveTask:
    if not isinstance(loss, ContrastiveConfig):
        raise TypeError("contrastive requires ContrastiveConfig")
    left_route, right_route = _pairwise_routes(task)
    return ContrastiveTask(
        distance=loss.distance,
        margin=loss.margin,
        online=loss.mining == "online",
        left_route=left_route,
        right_route=right_route,
    )


def _build_cosent_task(task: TaskConfig, loss: LossConfig) -> CoSENTTask:
    if not isinstance(loss, CoSENTConfig):
        raise TypeError("cosent requires CoSENTConfig")
    left_route, right_route = _pairwise_routes(task)
    return CoSENTTask(
        scale=loss.scale,
        left_route=left_route,
        right_route=right_route,
    )


def _build_angle_task(task: TaskConfig, loss: LossConfig) -> AnglETask:
    if not isinstance(loss, AngleConfig):
        raise TypeError("angle requires AngleConfig")
    left_route, right_route = _pairwise_routes(task)
    return AnglETask(
        scale=loss.scale,
        left_route=left_route,
        right_route=right_route,
    )


def _build_explicit_triplet_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ExplicitTripletTask:
    if not isinstance(task, ExplicitTripletConfig) or not isinstance(
        loss, ExplicitTripletLossConfig
    ):
        raise TypeError(
            "triplet requires ExplicitTripletConfig and ExplicitTripletLossConfig"
        )
    return ExplicitTripletTask(
        distance=loss.distance,
        margin=loss.margin,
        anchor_route=task.anchor_route,
        positive_route=task.positive_route,
        negative_route=task.negative_route,
    )


def _build_batch_triplet_task(
    task: TaskConfig,
    loss: LossConfig,
) -> BatchTripletTask:
    if not isinstance(task, LabeledExamplesConfig) or not isinstance(
        loss, BatchTripletLossConfig
    ):
        raise TypeError(
            "batch_triplet requires LabeledExamplesConfig and BatchTripletLossConfig"
        )
    return BatchTripletTask(
        mining=loss.mining,
        distance=loss.distance,
        margin=loss.margin,
        route=task.route,
    )


def _build_batch_hard_soft_margin_task(
    task: TaskConfig,
    loss: LossConfig,
) -> BatchTripletTask:
    if not isinstance(task, LabeledExamplesConfig) or not isinstance(
        loss, BatchHardSoftMarginLossConfig
    ):
        raise TypeError(
            "batch_hard_soft_margin requires LabeledExamplesConfig and "
            "BatchHardSoftMarginLossConfig"
        )
    return BatchTripletTask(
        mining="hard_soft_margin",
        distance=loss.distance,
        margin=None,
        route=task.route,
    )


def _build_embedding_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> EmbeddingDistillationTask:
    if not isinstance(task, EmbeddingDistillationConfig) or not isinstance(
        loss, EmbeddingDistillationLossConfig
    ):
        raise TypeError(
            "embedding_distillation requires EmbeddingDistillationConfig and "
            "EmbeddingDistillationLossConfig"
        )
    return EmbeddingDistillationTask(distance=loss.distance, routes=task.routes)


def _build_margin_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> MarginDistillationTask:
    if not isinstance(task, MarginDistillationConfig) or not isinstance(
        loss, MarginMSELossConfig
    ):
        raise TypeError(
            "margin_mse requires MarginDistillationConfig and MarginMSELossConfig"
        )
    return MarginDistillationTask(
        similarity=loss.similarity,
        query_route=task.query_route,
        document_route=task.document_route,
    )


def _build_distribution_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> DistributionDistillationTask:
    if not isinstance(task, DistributionDistillationConfig) or not isinstance(
        loss, DistributionKLLossConfig
    ):
        raise TypeError(
            "distribution_kl requires DistributionDistillationConfig and "
            "DistributionKLLossConfig"
        )
    return DistributionDistillationTask(
        similarity=loss.similarity,
        temperature=loss.temperature,
        query_route=task.query_route,
        candidate_route=task.candidate_route,
    )


BUILTIN_TASKS = TaskRegistry(
    (
        TaskDefinition(
            kind="retrieval",
            config_type=RetrievalConfig,
            batch_type=RetrievalBatch,
        ),
        TaskDefinition(
            kind="pairwise",
            config_type=PairwiseConfig,
            batch_type=PairwiseBatch,
        ),
        TaskDefinition(
            kind="explicit_triplet",
            config_type=ExplicitTripletConfig,
            batch_type=ExplicitTripletBatch,
        ),
        TaskDefinition(
            kind="labeled_examples",
            config_type=LabeledExamplesConfig,
            batch_type=LabeledExamplesBatch,
        ),
        TaskDefinition(
            kind="embedding_distillation",
            config_type=EmbeddingDistillationConfig,
            batch_type=EmbeddingDistillationBatch,
        ),
        TaskDefinition(
            kind="margin_distillation",
            config_type=MarginDistillationConfig,
            batch_type=MarginDistillationBatch,
        ),
        TaskDefinition(
            kind="distribution_distillation",
            config_type=DistributionDistillationConfig,
            batch_type=DistributionDistillationBatch,
        ),
    )
)
BUILTIN_LOSSES = LossRegistry(
    (
        LossDefinition(
            kind="mnr",
            config_type=MNRConfig,
            build=_build_mnr_task,
            task_kinds=frozenset({"retrieval"}),
            training_strategies=frozenset({"direct", "grad_cache"}),
        ),
        LossDefinition(
            kind="cosine_regression",
            config_type=CosineRegressionConfig,
            build=_build_cosine_regression_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="contrastive",
            config_type=ContrastiveConfig,
            build=_build_contrastive_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="cosent",
            config_type=CoSENTConfig,
            build=_build_cosent_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="angle",
            config_type=AngleConfig,
            build=_build_angle_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="triplet",
            config_type=ExplicitTripletLossConfig,
            build=_build_explicit_triplet_task,
            task_kinds=frozenset({"explicit_triplet"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="batch_triplet",
            config_type=BatchTripletLossConfig,
            build=_build_batch_triplet_task,
            task_kinds=frozenset({"labeled_examples"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="batch_hard_soft_margin",
            config_type=BatchHardSoftMarginLossConfig,
            build=_build_batch_hard_soft_margin_task,
            task_kinds=frozenset({"labeled_examples"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="embedding_distillation",
            config_type=EmbeddingDistillationLossConfig,
            build=_build_embedding_distillation_task,
            task_kinds=frozenset({"embedding_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="margin_mse",
            config_type=MarginMSELossConfig,
            build=_build_margin_distillation_task,
            task_kinds=frozenset({"margin_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="distribution_kl",
            config_type=DistributionKLLossConfig,
            build=_build_distribution_distillation_task,
            task_kinds=frozenset({"distribution_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
    )
)


def build_task(
    task: TaskConfig,
    loss: LossConfig,
    *,
    task_registry: TaskRegistry | None = None,
    loss_registry: LossRegistry | None = None,
) -> Any:
    """Build a runtime task from compatible scientific task and loss configs."""

    tasks = BUILTIN_TASKS if task_registry is None else task_registry
    losses = BUILTIN_LOSSES if loss_registry is None else loss_registry
    task = tasks.parse(task)
    loss = losses.parse(loss)
    definition = losses.definition(loss.kind)
    if task.kind not in definition.task_kinds:
        raise ValueError(f"loss {loss.kind!r} does not support task {task.kind!r}")
    return losses.build(task, loss)


__all__ = [
    "BUILTIN_LOSSES",
    "BUILTIN_TASKS",
    "LossDefinition",
    "LossRegistry",
    "TaskDefinition",
    "TaskRegistry",
    "build_task",
]
