"""Fixed-shape student payloads and offline guide representations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from representax.tasks._batch import payload_row_count


class GISTBatch(eqx.Module):
    """Aligned anchor/positive rows, optional negatives, and guide embeddings."""

    anchor: Any
    positive: Any
    negatives: tuple[Any, ...]
    guide_anchor: Float[Array, "batch guide"]
    guide_positive: Float[Array, "batch guide"]
    guide_negatives: tuple[Float[Array, "batch guide"], ...]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        row_count = self.guide_anchor.shape[0]
        guide_values = (
            self.guide_anchor,
            self.guide_positive,
            *self.guide_negatives,
        )
        if any(
            value.ndim != 2 or not jnp.issubdtype(value.dtype, jnp.floating)
            for value in guide_values
        ):
            raise TypeError("GIST guide embeddings must be floating matrices")
        if any(value.shape[0] != row_count for value in guide_values):
            raise ValueError("GIST guide embeddings must share their batch axis")
        guide_dimension = self.guide_anchor.shape[-1]
        if any(value.shape[-1] != guide_dimension for value in guide_values):
            raise ValueError("GIST guide embeddings must share their dimension")
        if len(self.negatives) != len(self.guide_negatives):
            raise ValueError("every GIST negative requires matching guide embeddings")
        if self.valid.shape != (row_count,) or self.valid.dtype != jnp.bool_:
            raise TypeError("GIST valid must be a boolean vector matching batch rows")
        for name, payload in (
            ("anchor", self.anchor),
            ("positive", self.positive),
            *(
                (f"negatives[{index}]", value)
                for index, value in enumerate(self.negatives)
            ),
        ):
            if payload_row_count(payload, name=name) != row_count:
                raise ValueError(f"{name} must contain one payload per batch row")


def gist_batch(
    *,
    anchor: Any,
    positive: Any,
    guide_anchor: Float[Array, "batch guide"],
    guide_positive: Float[Array, "batch guide"],
    negatives: Sequence[Any] = (),
    guide_negatives: Sequence[Float[Array, "batch guide"]] = (),
    valid: Bool[Array, " batch"] | None = None,
) -> GISTBatch:
    """Build a GIST batch without requiring a live guide model in training."""

    anchor_guide = jnp.asarray(guide_anchor, dtype=jnp.float32)
    row_count = anchor_guide.shape[0]
    if valid is None:
        valid = jnp.ones((row_count,), dtype=jnp.bool_)
    return GISTBatch(
        anchor=anchor,
        positive=positive,
        negatives=tuple(negatives),
        guide_anchor=anchor_guide,
        guide_positive=jnp.asarray(guide_positive, dtype=jnp.float32),
        guide_negatives=tuple(
            jnp.asarray(value, dtype=jnp.float32) for value in guide_negatives
        ),
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
