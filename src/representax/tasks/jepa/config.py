"""Serializable LeJEPA task and loss configuration."""

from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt

from representax.tasks.config import LossConfig, TaskConfig


class JEPAConfig(TaskConfig):
    kind: Literal["jepa"] = "jepa"


class LeJEPAConfig(LossConfig):
    kind: Literal["lejepa"] = "lejepa"
    regularization_weight: float = Field(default=0.02, ge=0.0, le=1.0)
    knots: int = Field(default=17, ge=2)
    slices: PositiveInt = 256
    max_frequency: PositiveFloat = 3.0


__all__ = ["JEPAConfig", "LeJEPAConfig"]
