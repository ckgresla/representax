"""Fixed-shape labeled-pair inputs for supervised representation learning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

if TYPE_CHECKING:
    from representax.models.processing import Processor
from representax.tasks._batch import payload_row_count


class PairwiseCollator:
    """Build labeled-pair batches with the loaded model's processor."""

    def __init__(
        self,
        *,
        processor: Processor,
        left_field: str = "sentence1",
        right_field: str = "sentence2",
        label_field: str = "score",
        pad_to_size: int | None = None,
    ) -> None:
        if pad_to_size is not None and pad_to_size <= 0:
            raise ValueError("pad_to_size must be positive or None")
        self.processor = processor
        self.left_field = left_field
        self.right_field = right_field
        self.label_field = label_field
        self.pad_to_size = pad_to_size

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-pairwise-collator-v1",
            "processor": self.processor.data_contract(),
            "left_field": self.left_field,
            "right_field": self.right_field,
            "label_field": self.label_field,
            "pad_to_size": self.pad_to_size,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> PairwiseBatch:
        try:
            left = tuple(str(example[self.left_field]) for example in examples)
            right = tuple(str(example[self.right_field]) for example in examples)
            labels = tuple(float(example[self.label_field]) for example in examples)
        except KeyError as error:
            raise KeyError(
                f"pairwise record is missing field {error.args[0]!r}"
            ) from error
        valid = [True] * len(labels)
        if self.pad_to_size is not None:
            if len(labels) > self.pad_to_size:
                raise ValueError("pairwise batch exceeds pad_to_size")
            padding = self.pad_to_size - len(labels)
            left = (*left, *("" for _ in range(padding)))
            right = (*right, *("" for _ in range(padding)))
            labels = (*labels, *(0.0 for _ in range(padding)))
            valid.extend(False for _ in range(padding))
        return pairwise_batch(
            left=self.processor(left),
            right=self.processor(right),
            labels=jnp.asarray(labels, dtype=jnp.float32),
            valid=jnp.asarray(valid),
        )


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


__all__ = ["PairwiseBatch", "PairwiseCollator", "pairwise_batch"]
