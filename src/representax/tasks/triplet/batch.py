"""Fixed-shape inputs for explicit and label-mined triplet learning."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def _payload_row_count(payload: Any, *, name: str) -> int:
    leaves = [value for value in jax.tree.leaves(payload) if eqx.is_array(value)]
    if not leaves:
        raise ValueError(f"{name} payload must contain arrays")
    if any(value.ndim == 0 for value in leaves):
        raise ValueError(f"{name} payload arrays must have a batch dimension")
    row_count = leaves[0].shape[0]
    if any(value.shape[0] != row_count for value in leaves):
        raise ValueError(f"{name} payload arrays must have the same row count")
    return row_count


class ExplicitTripletBatch(eqx.Module):
    """Aligned anchor, positive, and negative model-native payloads."""

    anchor: Any
    positive: Any
    negative: Any
    valid: Bool[Array, " triplet"]

    def __post_init__(self) -> None:
        if self.valid.ndim != 1 or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector")
        triplet_count = self.valid.shape[0]
        for name, payload in (
            ("anchor", self.anchor),
            ("positive", self.positive),
            ("negative", self.negative),
        ):
            if _payload_row_count(payload, name=name) != triplet_count:
                raise ValueError(f"{name} payload must contain one row per triplet")


def explicit_triplet_batch(
    *,
    anchor: Any,
    positive: Any,
    negative: Any,
    valid: Bool[Array, " triplet"] | None = None,
) -> ExplicitTripletBatch:
    """Build an explicit-triplet batch with all rows valid by default."""

    triplet_count = _payload_row_count(anchor, name="anchor")
    if valid is None:
        valid = jnp.ones((triplet_count,), dtype=jnp.bool_)
    return ExplicitTripletBatch(
        anchor=anchor,
        positive=positive,
        negative=negative,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


class LabeledExamplesBatch(eqx.Module):
    """One model-native payload with an integer class label per example."""

    examples: Any
    labels: Int[Array, " example"]
    valid: Bool[Array, " example"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.integer):
            raise TypeError("labels must be an integer vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector matching labels")
        if _payload_row_count(self.examples, name="examples") != self.labels.shape[0]:
            raise ValueError("examples payload must contain one row per label")


def labeled_examples_batch(
    *,
    examples: Any,
    labels: Int[Array, " example"],
    valid: Bool[Array, " example"] | None = None,
) -> LabeledExamplesBatch:
    """Build a class-labeled batch with all rows valid by default."""

    resolved_labels = jnp.asarray(labels)
    if not jnp.issubdtype(resolved_labels.dtype, jnp.integer):
        raise TypeError("labels must have an integer dtype")
    if valid is None:
        valid = jnp.ones(resolved_labels.shape, dtype=jnp.bool_)
    return LabeledExamplesBatch(
        examples=examples,
        labels=resolved_labels,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


__all__ = [
    "ExplicitTripletBatch",
    "LabeledExamplesBatch",
    "explicit_triplet_batch",
    "labeled_examples_batch",
]
