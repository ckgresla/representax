"""Bounded-memory execution for mega-batch hard-negative mining."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import Encoder, LossOutput
from representax.tasks.mega_batch import MegaBatch, MegaBatchMarginTask

from .execution import _LOCAL_EXECUTION_CONTEXT, ExecutionContext
from .grad_cache import _leading_batch_size, _rematerialized_encode


def _normalize(values: Array) -> Array:
    return values / jnp.maximum(
        jnp.linalg.norm(values, axis=-1, keepdims=True),
        jnp.asarray(1e-12, values.dtype),
    )


def _hard_negative_indices(
    anchors: Array,
    candidates: Array,
    valid: Array,
    *,
    row_chunk_size: int,
) -> Array:
    batch_size = anchors.shape[0]
    chunk_count = (batch_size + row_chunk_size - 1) // row_chunk_size
    padded_size = chunk_count * row_chunk_size
    padding = padded_size - batch_size
    padded = jnp.pad(anchors, ((0, padding), (0, 0))).reshape(
        chunk_count,
        row_chunk_size,
        anchors.shape[1],
    )
    row_indices = jnp.pad(jnp.arange(batch_size), (0, padding)).reshape(
        chunk_count,
        row_chunk_size,
    )

    def body(_, values):
        rows, indices = values
        scores = _normalize(rows) @ _normalize(candidates).T
        allowed = valid[None, :] & (jnp.arange(batch_size)[None, :] != indices[:, None])
        return None, jnp.argmax(jnp.where(allowed, scores, -jnp.inf), axis=1)

    _, selected = jax.lax.scan(body, None, (padded, row_indices))
    return selected.reshape(-1)[:batch_size]


@dataclass(frozen=True)
class MegaBatchMining:
    """Mine against frozen candidates, then replay differentiable triplets."""

    micro_batch_size: int
    loss_row_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.micro_batch_size <= 0:
            raise ValueError("mega-batch micro_batch_size must be positive")
        if self.loss_row_chunk_size is not None and self.loss_row_chunk_size <= 0:
            raise ValueError("mega-batch loss_row_chunk_size must be positive")

    @property
    def resolved_loss_row_chunk_size(self) -> int:
        return self.loss_row_chunk_size or self.micro_batch_size

    def validate(self, task: Any) -> None:
        if not isinstance(task, MegaBatchMarginTask):
            raise TypeError("MegaBatchMining requires MegaBatchMarginTask")

    def evaluate(
        self,
        task: Any,
        model: eqx.Module,
        batch: Any,
        *,
        key: PRNGKeyArray | None,
        context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    ) -> LossOutput:
        if not isinstance(task, MegaBatchMarginTask) or not isinstance(
            batch, MegaBatch
        ):
            raise TypeError("MegaBatchMining requires its task and batch")
        if context.data_axis_name is not None:
            raise NotImplementedError(
                "distributed mega-batch mining is not implemented"
            )
        encoder = cast(Encoder, model)
        batch_size = _leading_batch_size(batch.anchor, role="anchor")
        if key is None:
            anchor_key = positive_key = negative_key = None
        else:
            anchor_key, positive_key, negative_key = jax.random.split(key, 3)
        anchor = _rematerialized_encode(
            encoder,
            batch.anchor,
            route=task.anchor_route,
            batch_size=batch_size,
            chunk_size=self.micro_batch_size,
            key=anchor_key,
        )
        mining_positive = jax.lax.stop_gradient(
            _rematerialized_encode(
                encoder,
                batch.positive,
                route=task.positive_route,
                batch_size=batch_size,
                chunk_size=self.micro_batch_size,
                key=None,
            )
        )
        negative_indices = _hard_negative_indices(
            jax.lax.stop_gradient(anchor),
            mining_positive,
            batch.valid,
            row_chunk_size=self.resolved_loss_row_chunk_size,
        )
        negative_payload = jax.tree.map(
            lambda value: value[negative_indices],
            batch.positive,
        )
        positive = _rematerialized_encode(
            encoder,
            batch.positive,
            route=task.positive_route,
            batch_size=batch_size,
            chunk_size=self.micro_batch_size,
            key=positive_key,
        )
        negative = _rematerialized_encode(
            encoder,
            negative_payload,
            route=task.positive_route,
            batch_size=batch_size,
            chunk_size=self.micro_batch_size,
            key=negative_key,
        )
        return task.loss_from_selected_embeddings(anchor, positive, negative, batch)
