"""Task boundary shared by retrieval and future representation objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

ModelT = TypeVar("ModelT")


class LossOutput(eqx.Module):
    """Scalar optimization target plus stable, named metric leaves."""

    loss: Float[Array, ""]
    metrics: Mapping[str, Array]


@runtime_checkable
class Task(Protocol, Generic[ModelT]):
    """A task interprets examples and turns model outputs into a loss."""

    def loss(
        self,
        model: ModelT,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput: ...


def evaluate_loss(
    task: Task[ModelT],
    model: ModelT,
    batch: Any,
    *,
    key: PRNGKeyArray | None = None,
) -> LossOutput:
    """Evaluate a task while validating the optimization boundary."""

    output = task.loss(model, batch, key=key)
    loss = jnp.asarray(output.loss)
    if loss.shape != ():
        raise ValueError("task loss must be scalar")
    if not jnp.issubdtype(loss.dtype, jnp.floating):
        raise TypeError("task loss must have a floating dtype")
    return output
