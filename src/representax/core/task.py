"""Task boundary shared by retrieval and future representation objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.precision import accumulated_values, loss_value

ModelT = TypeVar("ModelT")

# Representation functions may return one dense array or a structured PyTree
# such as token values plus their validity mask. Every array leaf remains
# leading-batch-major so execution transforms can chunk and replay it generically.
EncodeFunction = Callable[..., Any]


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


@runtime_checkable
class PostUpdateTask(Task[ModelT], Protocol, Generic[ModelT]):
    """A task with a deterministic model-state transition after optimization.

    This is the narrow stateful seam used by objectives such as V-JEPA, whose
    stop-gradient target encoder is an exponential moving average of the newly
    optimized online encoder. The ordinary optimizer remains responsible only
    for selected trainable leaves.
    """

    def post_update_model(
        self,
        previous_model: ModelT,
        optimized_model: ModelT,
        *,
        step: Array,
    ) -> ModelT: ...


@runtime_checkable
class RepresentationTask(Task[ModelT], Protocol, Generic[ModelT]):
    """A task whose model work can be separated from its representation loss.

    This seam lets loss modifiers reuse a single encoder pass. ``encode_fn`` is
    normally :func:`representax.core.encode`; adaptive-layer training supplies
    the corresponding layerwise encoder operation instead.
    """

    def representations(
        self,
        model: ModelT,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction,
    ) -> Any: ...

    def loss_from_representations(
        self,
        representations: Any,
        batch: Any,
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
    return LossOutput(
        loss=loss_value(loss),
        metrics=accumulated_values(output.metrics),
    )
