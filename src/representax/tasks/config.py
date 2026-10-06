"""Base configuration identity shared by registered task families."""

from __future__ import annotations

from representax._config import FrozenConfig, NonEmptyString


class TaskConfig(FrozenConfig):
    """Serializable base identity resolved by a task registry."""

    kind: NonEmptyString


class LossConfig(FrozenConfig):
    """Serializable loss identity resolved by a loss registry."""

    kind: NonEmptyString


class LossModifierConfig(FrozenConfig):
    """Serializable loss-composition identity resolved by a modifier registry."""

    kind: NonEmptyString


__all__ = ["LossConfig", "LossModifierConfig", "TaskConfig"]
