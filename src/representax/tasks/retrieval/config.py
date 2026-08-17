"""Serializable scientific configuration for MNR retrieval tasks."""

from __future__ import annotations

from typing import Literal

from representax._config import FinitePositiveFloat
from representax.tasks.config import LossConfig, TaskConfig


class RetrievalConfig(TaskConfig):
    """Query/document retrieval task and batch semantics."""

    kind: Literal["retrieval"] = "retrieval"


class MNRConfig(LossConfig):
    """Scientific definition of a multiple-negatives ranking objective."""

    kind: Literal["mnr"] = "mnr"
    scale: FinitePositiveFloat = 20.0
    symmetric: bool = False
    negative_scope: Literal["local", "global"] = "global"


__all__ = ["MNRConfig", "RetrievalConfig"]
