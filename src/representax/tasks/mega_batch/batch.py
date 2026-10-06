"""Aligned anchor/positive inputs for mega-batch mining."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool

from representax.tasks._batch import payload_row_count


class MegaBatch(eqx.Module):
    anchor: Any
    positive: Any
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.valid.ndim != 1 or self.valid.dtype != jnp.bool_:
            raise TypeError("mega-batch valid must be a boolean vector")
        row_count = self.valid.shape[0]
        if payload_row_count(self.anchor, name="anchor") != row_count:
            raise ValueError("mega-batch anchors must match valid rows")
        if payload_row_count(self.positive, name="positive") != row_count:
            raise ValueError("mega-batch positives must match valid rows")


def mega_batch(
    *,
    anchor: Any,
    positive: Any,
    valid: Bool[Array, " batch"] | None = None,
) -> MegaBatch:
    row_count = payload_row_count(anchor, name="anchor")
    if valid is None:
        valid = jnp.ones((row_count,), dtype=jnp.bool_)
    return MegaBatch(
        anchor=anchor,
        positive=positive,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
