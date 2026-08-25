"""Model-neutral contracts for joint-input scorers and rerankers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.precision import (
    activation_inputs,
    active_model_for_compute,
    objective_output,
)


@runtime_checkable
class Scorer(Protocol):
    """A native model that jointly maps prepared inputs to raw score logits."""

    def logits(
        self,
        inputs: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, ...]: ...


def score_logits(
    model: Scorer,
    inputs: Any,
    *,
    key: PRNGKeyArray | None = None,
) -> Float[Array, ...]:
    """Evaluate raw logits and enforce the shared scorer output contract."""

    if not isinstance(model, eqx.Module):
        raise TypeError("scorers must be Equinox module trees")
    logits = getattr(active_model_for_compute(model), "logits", None)
    if not callable(logits):
        raise TypeError("scorers must implement logits")
    result = jnp.asarray(logits(activation_inputs(inputs), key=key))
    if result.ndim not in {1, 2}:
        raise ValueError("scorer logits must have shape [batch] or [batch, output]")
    if not jnp.issubdtype(result.dtype, jnp.floating):
        raise TypeError("scorer logits must have a floating dtype")
    return objective_output(result)


__all__ = ["Scorer", "score_logits"]
