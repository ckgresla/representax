"""Serializable scientific configuration for representation-loss modifiers."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, PositiveInt, model_validator

from representax._config import FinitePositiveFloat
from representax.tasks.config import LossModifierConfig

FiniteNonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class MatryoshkaModifierConfig(LossModifierConfig):
    """Evaluate one representation loss over normalized prefix dimensions."""

    kind: Literal["matryoshka"] = "matryoshka"
    dimensions: tuple[PositiveInt, ...]
    weights: tuple[FinitePositiveFloat, ...] | None = None
    dimensions_per_step: int = -1

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if not self.dimensions:
            raise ValueError("Matryoshka dimensions must be non-empty")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Matryoshka dimensions must be unique")
        if self.weights is not None and len(self.weights) != len(self.dimensions):
            raise ValueError("Matryoshka weights must match dimensions")
        if self.dimensions_per_step == 0 or self.dimensions_per_step < -1:
            raise ValueError("dimensions_per_step must be -1 or positive")
        return self


class AdaptiveLayerModifierConfig(LossModifierConfig):
    """Train selected prior encoder depths against the final-layer objective."""

    kind: Literal["adaptive_layer"] = "adaptive_layer"
    layers_per_step: int = 1
    final_layer_weight: FiniteNonNegativeFloat = 1.0
    prior_layer_weight: FiniteNonNegativeFloat = 1.0
    kl_divergence_weight: FiniteNonNegativeFloat = 1.0
    kl_temperature: FiniteNonNegativeFloat = 0.3

    @model_validator(mode="after")
    def validate_layers_per_step(self) -> Self:
        if self.layers_per_step == 0 or self.layers_per_step < -1:
            raise ValueError("layers_per_step must be -1 or positive")
        return self


class Matryoshka2dModifierConfig(LossModifierConfig):
    """Joint dimension-prefix and adaptive-layer objective composition."""

    kind: Literal["matryoshka_2d"] = "matryoshka_2d"
    dimensions: tuple[PositiveInt, ...]
    weights: tuple[FinitePositiveFloat, ...] | None = None
    dimensions_per_step: int = 1
    layers_per_step: int = 1
    final_layer_weight: FiniteNonNegativeFloat = 1.0
    prior_layer_weight: FiniteNonNegativeFloat = 1.0
    kl_divergence_weight: FiniteNonNegativeFloat = 1.0
    kl_temperature: FiniteNonNegativeFloat = 0.3

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        if not self.dimensions:
            raise ValueError("Matryoshka dimensions must be non-empty")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("Matryoshka dimensions must be unique")
        if self.weights is not None and len(self.weights) != len(self.dimensions):
            raise ValueError("Matryoshka weights must match dimensions")
        for name, value in (
            ("dimensions_per_step", self.dimensions_per_step),
            ("layers_per_step", self.layers_per_step),
        ):
            if value == 0 or value < -1:
                raise ValueError(f"{name} must be -1 or positive")
        return self


__all__ = [
    "AdaptiveLayerModifierConfig",
    "Matryoshka2dModifierConfig",
    "MatryoshkaModifierConfig",
]
