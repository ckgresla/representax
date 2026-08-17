"""Fixed-shape pair-classification inputs."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from representax.tasks._batch import payload_row_count


class PairClassificationBatch(eqx.Module):
    """Two aligned payloads with one integer class per row."""

    left: Any
    right: Any
    labels: Int[Array, " pair"]
    valid: Bool[Array, " pair"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.integer):
            raise TypeError("classification labels must be an integer vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("classification valid must be a matching boolean vector")
        for name, payload in (("left", self.left), ("right", self.right)):
            if payload_row_count(payload, name=name) != self.labels.shape[0]:
                raise ValueError(f"{name} must contain one payload per label")


def pair_classification_batch(
    *,
    left: Any,
    right: Any,
    labels: Int[Array, " pair"],
    valid: Bool[Array, " pair"] | None = None,
) -> PairClassificationBatch:
    """Build a pair-classification batch with all rows valid by default."""

    resolved_labels = jnp.asarray(labels, dtype=jnp.int32)
    if valid is None:
        valid = jnp.ones(resolved_labels.shape, dtype=jnp.bool_)
    return PairClassificationBatch(
        left=left,
        right=right,
        labels=resolved_labels,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
