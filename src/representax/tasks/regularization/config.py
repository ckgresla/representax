"""Serializable global orthogonal regularization configuration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import NonNegativeFloat, model_validator

from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig


class RegularizationConfig(TaskConfig):
    kind: Literal["representation_regularization"] = "representation_regularization"
    routes: tuple[Route, ...] = (Route.GENERIC,)


class GORConfig(LossConfig):
    kind: Literal["global_orthogonal_regularization"] = (
        "global_orthogonal_regularization"
    )
    similarity: Literal["cosine", "dot"] = "cosine"
    mean_weight: NonNegativeFloat = 1.0
    second_moment_weight: NonNegativeFloat = 1.0
    aggregation: Literal["mean", "sum"] = "mean"

    @model_validator(mode="after")
    def validate_weights(self) -> Self:
        if self.mean_weight == 0 and self.second_moment_weight == 0:
            raise ValueError("at least one GOR weight must be positive")
        return self


__all__ = ["GORConfig", "RegularizationConfig"]
