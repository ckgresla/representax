"""Fixed-shape sparse targets for encoder masked language modeling."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Bool, Int

from representax.tasks._batch import payload_row_count


class MaskedLanguageModelBatch(eqx.Module):
    """Corrupted inputs and selected token positions; -100 labels are ignored.

    Positions must index the input sequence. Unused target slots use position
    zero and label -100. Padding and special tokens must not be supervised.
    """

    inputs: Any
    positions: Int[Array, "batch target"]
    labels: Int[Array, "batch target"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or not jnp.issubdtype(self.labels.dtype, jnp.integer):
            raise TypeError("MLM labels must be a two-dimensional integer array")
        if self.positions.shape != self.labels.shape or not jnp.issubdtype(
            self.positions.dtype, jnp.integer
        ):
            raise TypeError("MLM positions must be matching integer indices")
        if self.valid.shape != self.labels.shape[:1] or self.valid.dtype != jnp.bool_:
            raise TypeError("MLM valid must be a matching boolean row vector")
        if payload_row_count(self.inputs, name="inputs") != self.labels.shape[0]:
            raise ValueError("MLM inputs must contain one payload per target row")


def masked_language_model_batch(
    *,
    inputs: Any,
    positions: ArrayLike,
    labels: ArrayLike,
    valid: ArrayLike | None = None,
) -> MaskedLanguageModelBatch:
    labels = jnp.asarray(labels, dtype=jnp.int32)
    return MaskedLanguageModelBatch(
        inputs=inputs,
        positions=jnp.asarray(positions, dtype=jnp.int32),
        labels=labels,
        valid=jnp.ones(labels.shape[:1], dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_),
    )
