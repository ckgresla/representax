"""Token-level representation contracts for late-interaction models."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.precision import (
    activation_inputs,
    active_model_for_compute,
    objective_output,
)

from .model import EncoderMetadata, Route


class LateInteractionRepresentation(eqx.Module):
    """Normalized token vectors and their fixed-shape validity mask."""

    values: Float[Array, "batch token dimension"]
    valid: Bool[Array, "batch token"]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(
                "late-interaction values must have shape [batch, token, dimension]"
            )
        if self.valid.shape != self.values.shape[:2]:
            raise ValueError("late-interaction validity must match [batch, token]")
        if not jnp.issubdtype(self.values.dtype, jnp.floating):
            raise TypeError("late-interaction values must be floating point")
        if self.valid.dtype != jnp.bool_:
            raise TypeError("late-interaction validity must be boolean")


@runtime_checkable
class LateInteractionEncoder(Protocol):
    """A model that retains one contextual representation per valid token."""

    metadata: EncoderMetadata

    def encode_late_interaction(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> LateInteractionRepresentation: ...


def encode_late_interaction(
    model: LateInteractionEncoder,
    inputs: Any,
    *,
    route: Route = Route.GENERIC,
    key: PRNGKeyArray | None = None,
) -> LateInteractionRepresentation:
    """Encode and enforce the shared normalized token-representation contract."""

    if not isinstance(model, eqx.Module):
        raise TypeError("late-interaction encoders must be Equinox module trees")
    metadata = getattr(model, "metadata", None)
    if not isinstance(metadata, EncoderMetadata):
        raise TypeError("late-interaction encoders must expose EncoderMetadata")
    route = Route(route)
    if route not in metadata.routes:
        raise ValueError(f"{metadata.model_id} does not support route {route.value!r}")
    operation = getattr(
        active_model_for_compute(model), "encode_late_interaction", None
    )
    if not callable(operation):
        raise TypeError(
            "late-interaction encoders must implement encode_late_interaction"
        )
    representation = operation(activation_inputs(inputs), route=route, key=key)
    if not isinstance(representation, LateInteractionRepresentation):
        raise TypeError(
            "encode_late_interaction must return LateInteractionRepresentation"
        )
    if representation.values.shape[-1] != metadata.output_dimension:
        raise ValueError(
            f"{metadata.model_id} declares output_dimension="
            f"{metadata.output_dimension} "
            f"but returned {representation.values.shape[-1]}"
        )

    values = objective_output(representation.values).astype(jnp.float32)
    norm = jnp.linalg.norm(values, axis=-1, keepdims=True)
    values = values / jnp.maximum(norm, jnp.asarray(1e-12, dtype=values.dtype))
    values = jnp.where(representation.valid[..., None], values, 0.0)
    return LateInteractionRepresentation(values=values, valid=representation.valid)


__all__ = [
    "LateInteractionEncoder",
    "LateInteractionRepresentation",
    "encode_late_interaction",
]
