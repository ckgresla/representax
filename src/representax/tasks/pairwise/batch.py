"""Fixed-shape labeled-pair inputs for supervised representation learning."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from representax.tasks._batch import payload_row_count


class PairwiseBatch(eqx.Module):
    """Two aligned model-native payloads with one label per pair."""

    left: Any
    right: Any
    labels: Float[Array, " pair"]
    valid: Bool[Array, " pair"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.floating):
            raise TypeError("labels must be a floating-point vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector matching labels")
        pair_count = self.labels.shape[0]
        for name, payload in (("left", self.left), ("right", self.right)):
            if payload_row_count(payload, name=name) != pair_count:
                raise ValueError(f"{name} payload must contain one row per pair")


def pairwise_batch(
    *,
    left: Any,
    right: Any,
    labels: Float[Array, " pair"],
    valid: Bool[Array, " pair"] | None = None,
) -> PairwiseBatch:
    """Build a labeled-pair batch with all rows valid by default."""

    resolved_labels = jnp.asarray(labels, dtype=jnp.float32)
    if valid is None:
        valid = jnp.ones(resolved_labels.shape, dtype=jnp.bool_)
    return PairwiseBatch(
        left=left,
        right=right,
        labels=resolved_labels,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


__all__ = ["PairwiseBatch", "pairwise_batch"]
