"""Damaged encoder inputs and original decoder token targets."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int

from representax.tasks._batch import payload_row_count


class DenoisingBatch(eqx.Module):
    damaged: Any
    target_input_ids: Int[Array, "batch sequence"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.target_input_ids.ndim != 2 or not jnp.issubdtype(
            self.target_input_ids.dtype, jnp.integer
        ):
            raise TypeError("denoising targets must be an integer token matrix")
        if self.target_input_ids.shape[1] < 2:
            raise ValueError("denoising targets require at least two tokens")
        if self.valid.shape != self.target_input_ids.shape[:1]:
            raise ValueError("denoising valid must match target rows")
        if self.valid.dtype != jnp.bool_:
            raise TypeError("denoising valid must be boolean")
        if payload_row_count(self.damaged, name="damaged") != self.valid.shape[0]:
            raise ValueError("damaged inputs must match target rows")


def denoising_batch(
    *,
    damaged: Any,
    target_input_ids: Int[Array, "batch sequence"],
    valid: Bool[Array, " batch"] | None = None,
) -> DenoisingBatch:
    targets = jnp.asarray(target_input_ids, dtype=jnp.int32)
    if valid is None:
        valid = jnp.ones(targets.shape[:1], dtype=jnp.bool_)
    return DenoisingBatch(
        damaged=damaged,
        target_input_ids=targets,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
