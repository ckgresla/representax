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


class VJEPA2_1TaskConfig(TaskConfig):
    kind: Literal["vjepa2_1"] = "vjepa2_1"


class VJEPA2_1DenseConfig(LossConfig):
    kind: Literal["vjepa2_1_dense"] = "vjepa2_1_dense"
    context_weight: float = Field(default=0.5, ge=0.0)
    ema_start: float = Field(default=0.99925, ge=0.0, le=1.0)
    ema_end: float = Field(default=0.99925, ge=0.0, le=1.0)
    ema_steps: PositiveInt = 1
    offset_context_loss: bool = False


__all__ = [
    "JEPAConfig",
    "LeJEPAConfig",
    "VJEPA2_1TaskConfig",
    "VJEPA2_1DenseConfig",
]
