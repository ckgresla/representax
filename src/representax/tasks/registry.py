"""Immutable registries for structured task and loss configurations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .config import LossConfig, TaskConfig
from .retrieval.batch import RetrievalBatch
from .retrieval.config import MNRConfig, RetrievalConfig
from .retrieval.mnr import MNRTask


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
    build: Callable[[LossConfig], Any]
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

    def build(self, config: LossConfig) -> Any:
        definition = self.definition(config.kind)
        if not isinstance(config, definition.config_type):
            raise TypeError(
                f"loss {config.kind!r} requires {definition.config_type.__name__}, "
                f"received {type(config).__name__}"
            )
        return definition.build(config)

    def extended(self, *definitions: LossDefinition) -> LossRegistry:
        return LossRegistry((*self._definitions.values(), *definitions))


def _build_mnr_task(config: LossConfig) -> MNRTask:
    if not isinstance(config, MNRConfig):
        raise TypeError("mnr requires MNRConfig")
    return MNRTask(
        scale=config.scale,
        symmetric=config.symmetric,
        dimensions=config.dimensions,
        dimension_weights=config.dimension_weights,
        negative_scope=config.negative_scope,
    )


BUILTIN_TASKS = TaskRegistry(
    (
        TaskDefinition(
            kind="retrieval",
            config_type=RetrievalConfig,
            batch_type=RetrievalBatch,
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
    return losses.build(loss)


__all__ = [
    "BUILTIN_LOSSES",
    "BUILTIN_TASKS",
    "LossDefinition",
    "LossRegistry",
    "TaskDefinition",
    "TaskRegistry",
    "build_task",
]
