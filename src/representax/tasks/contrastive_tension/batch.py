"""Fixed-shape batches for contrastive-tension pretraining."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from representax.tasks._batch import payload_row_count


class ContrastiveTensionBatch(eqx.Module):
    first: Any
    second: Any
    labels: Float[Array, " pair"]
    valid: Bool[Array, " pair"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.floating):
            raise TypeError("contrastive-tension labels must be a floating vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("contrastive-tension valid must match labels")
        for name, payload in (("first", self.first), ("second", self.second)):
            if payload_row_count(payload, name=name) != self.labels.shape[0]:
                raise ValueError(f"{name} must contain one payload per label")


class ContrastiveTensionExamples(eqx.Module):
    examples: Any
    valid: Bool[Array, " example"]

    def __post_init__(self) -> None:
        if self.valid.ndim != 1 or self.valid.dtype != jnp.bool_:
            raise TypeError("contrastive-tension valid must be a boolean vector")
        if payload_row_count(self.examples, name="examples") != self.valid.shape[0]:
            raise ValueError("examples must contain one payload per valid row")


def contrastive_tension_batch(
    *,
    first: Any,
    second: Any,
    labels: Float[Array, " pair"],
    valid: Bool[Array, " pair"] | None = None,
) -> ContrastiveTensionBatch:
    resolved_labels = jnp.asarray(labels, dtype=jnp.float32)
    if valid is None:
        valid = jnp.ones(resolved_labels.shape, dtype=jnp.bool_)
    return ContrastiveTensionBatch(
        first=first,
        second=second,
        labels=resolved_labels,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


def contrastive_tension_examples(
    examples: Any,
    *,
    valid: Bool[Array, " example"] | None = None,
) -> ContrastiveTensionExamples:
    row_count = payload_row_count(examples, name="examples")
    if valid is None:
        valid = jnp.ones((row_count,), dtype=jnp.bool_)
    return ContrastiveTensionExamples(
        examples=examples,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )
