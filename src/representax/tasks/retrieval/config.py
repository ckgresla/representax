"""Serializable scientific configuration for MNR retrieval tasks."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import PositiveInt, model_validator

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
    dimensions: tuple[PositiveInt, ...] | None = None
    dimension_weights: tuple[FinitePositiveFloat, ...] | None = None
    negative_scope: Literal["local", "global"] = "global"

    @model_validator(mode="after")
    def validate_matryoshka(self) -> Self:
        if self.dimensions is not None:
            if tuple(sorted(set(self.dimensions))) != self.dimensions:
                raise ValueError("Matryoshka dimensions must be sorted and unique")
            if self.dimension_weights is not None and len(
                self.dimension_weights
            ) != len(self.dimensions):
                raise ValueError("dimension weights must match dimensions")
        elif self.dimension_weights is not None:
            raise ValueError("dimension weights require Matryoshka dimensions")
        return self


__all__ = ["MNRConfig", "RetrievalConfig"]
