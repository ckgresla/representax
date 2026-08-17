"""Arbitrary aligned input columns for representation regularization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool

from representax.tasks._batch import payload_row_count


class RegularizationBatch(eqx.Module):
    inputs: tuple[Any, ...]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("regularization requires at least one input column")
        if self.valid.ndim != 1 or self.valid.dtype != jnp.bool_:
            raise TypeError("regularization valid must be a boolean vector")
        for index, payload in enumerate(self.inputs):
            if (
                payload_row_count(payload, name=f"inputs[{index}]")
                != self.valid.shape[0]
            ):
                raise ValueError("every regularization input must match valid rows")


def regularization_batch(
    inputs: Sequence[Any],
    *,
    valid: Bool[Array, " batch"] | None = None,
) -> RegularizationBatch:
    resolved = tuple(inputs)
    if not resolved:
        raise ValueError("regularization requires at least one input column")
    row_count = payload_row_count(resolved[0], name="inputs[0]")
    if valid is None:
        valid = jnp.ones((row_count,), dtype=jnp.bool_)
    return RegularizationBatch(
        inputs=resolved,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
