"""Fixed-shape batches and host collators for reward modeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from representax.tasks._batch import payload_row_count

if TYPE_CHECKING:
    from representax.models.processing import Processor


def _reshape_candidates(payload: Any, rows: int, candidates: int) -> Any:
    def reshape(value: Any) -> Any:
        if not isinstance(value, (jax.Array, np.ndarray)):
            return value
        if value.shape[0] != rows * candidates:
            raise ValueError("processed rewards must contain every candidate")
        return value.reshape((rows, candidates, *value.shape[1:]))

    return jax.tree.map(reshape, payload)


class PairwiseRewardBatch(eqx.Module):
    """Chosen/rejected inputs, optional preference margins, and valid rows."""

    chosen: Any
    rejected: Any
    margins: Float[Array, " batch"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.margins.ndim != 1 or not jnp.issubdtype(
            self.margins.dtype, jnp.floating
        ):
            raise TypeError("reward margins must be a floating-point vector")
        if self.valid.shape != self.margins.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("reward validity must be a matching boolean vector")
        for name, payload in (("chosen", self.chosen), ("rejected", self.rejected)):
            if payload_row_count(payload, name=name) != self.margins.shape[0]:
                raise ValueError(f"{name} must contain one row per preference")


class ListwiseRewardBatch(eqx.Module):
    """Candidate inputs and graded preferences, grouped by prompt."""

    candidates: Any
    preferences: Float[Array, "batch candidate"]
    valid: Bool[Array, "batch candidate"]

    def __post_init__(self) -> None:
        if self.preferences.ndim != 2 or not jnp.issubdtype(
            self.preferences.dtype, jnp.floating
        ):
            raise TypeError("listwise preferences must be a floating-point matrix")
        if self.valid.shape != self.preferences.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("listwise validity must be a matching boolean matrix")
        leaves = [
            leaf
            for leaf in jax.tree.leaves(self.candidates)
            if isinstance(leaf, (jax.Array, np.ndarray))
        ]
        if not leaves or any(
            leaf.shape[:2] != self.preferences.shape for leaf in leaves
        ):
            raise ValueError("candidate leaves must begin with [batch, candidate]")


class PointwiseRewardBatch(eqx.Module):
    """One independently scored input and scalar target per row."""

    inputs: Any
    labels: Float[Array, " batch"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.floating):
            raise TypeError("pointwise reward labels must be a floating vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("pointwise validity must be a matching boolean vector")
        if payload_row_count(self.inputs, name="inputs") != self.labels.shape[0]:
            raise ValueError("pointwise inputs must contain one row per target")


class ProcessRewardBatch(eqx.Module):
    """Inputs with fixed step/token targets and a validity mask."""

    inputs: Any
    labels: Float[Array, "batch step"]
    valid: Bool[Array, "batch step"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 2 or not jnp.issubdtype(self.labels.dtype, jnp.floating):
            raise TypeError("process reward labels must have [batch, step] shape")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("process validity must be a matching boolean matrix")
        if payload_row_count(self.inputs, name="inputs") != self.labels.shape[0]:
            raise ValueError("process inputs must contain one row per trajectory")


class PairwiseRewardCollator:
    """Process chosen and rejected artifacts without framework-owned formatting."""

    def __init__(
        self,
        *,
        processor: Processor,
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        margin_field: str = "margin",
    ) -> None:
        self.processor = processor
        self.chosen_field = chosen_field
        self.rejected_field = rejected_field
        self.margin_field = margin_field

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-pairwise-reward-collator-v1",
            "processor": self.processor.data_contract(),
            "chosen_field": self.chosen_field,
            "rejected_field": self.rejected_field,
            "margin_field": self.margin_field,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> PairwiseRewardBatch:
        margins = np.asarray(
            [float(example.get(self.margin_field, 0.0)) for example in examples],
            dtype=np.float32,
        )
        return PairwiseRewardBatch(
            chosen=self.processor([example[self.chosen_field] for example in examples]),
            rejected=self.processor(
                [example[self.rejected_field] for example in examples]
            ),
            margins=jnp.asarray(margins),
            valid=jnp.ones(margins.shape, dtype=jnp.bool_),
        )


class ListwiseRewardCollator:
    """Process finite candidate lists and graded preference targets."""

    def __init__(
        self,
        *,
        processor: Processor,
        candidates_per_prompt: int,
        candidates_field: str = "candidates",
        preferences_field: str = "preferences",
    ) -> None:
        if candidates_per_prompt < 2:
            raise ValueError("listwise rewards require at least two candidate slots")
        self.processor = processor
        self.candidates_per_prompt = candidates_per_prompt
        self.candidates_field = candidates_field
        self.preferences_field = preferences_field

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> ListwiseRewardBatch:
        flat: list[Any] = []
        preferences = np.zeros(
            (len(examples), self.candidates_per_prompt), dtype=np.float32
        )
        valid = np.zeros(preferences.shape, dtype=np.bool_)
        for row, example in enumerate(examples):
            candidates = list(example[self.candidates_field])
            values = list(example[self.preferences_field])
            if len(candidates) != len(values):
                raise ValueError("reward candidates and preferences must align")
            if not 2 <= len(candidates) <= self.candidates_per_prompt:
                raise ValueError("reward list is outside its finite shape bucket")
            preferences[row, : len(values)] = values
            valid[row, : len(values)] = True
            flat.extend(candidates)
            flat.extend("" for _ in range(self.candidates_per_prompt - len(candidates)))
        return ListwiseRewardBatch(
            candidates=_reshape_candidates(
                self.processor(flat), len(examples), self.candidates_per_prompt
            ),
            preferences=jnp.asarray(preferences),
            valid=jnp.asarray(valid),
        )


class PointwiseRewardCollator:
    def __init__(
        self,
        *,
        processor: Processor,
        input_field: str = "input",
        label_field: str = "label",
    ) -> None:
        self.processor = processor
        self.input_field = input_field
        self.label_field = label_field

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> PointwiseRewardBatch:
        labels = np.asarray(
            [example[self.label_field] for example in examples], dtype=np.float32
        )
        return PointwiseRewardBatch(
            inputs=self.processor([example[self.input_field] for example in examples]),
            labels=jnp.asarray(labels),
            valid=jnp.ones(labels.shape, dtype=jnp.bool_),
        )


class ProcessRewardCollator:
    def __init__(
        self,
        *,
        processor: Processor,
        steps_per_trajectory: int,
        input_field: str = "input",
        labels_field: str = "labels",
    ) -> None:
        if steps_per_trajectory <= 0:
            raise ValueError("steps_per_trajectory must be positive")
        self.processor = processor
        self.steps_per_trajectory = steps_per_trajectory
        self.input_field = input_field
        self.labels_field = labels_field

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> ProcessRewardBatch:
        labels = np.zeros((len(examples), self.steps_per_trajectory), dtype=np.float32)
        valid = np.zeros(labels.shape, dtype=np.bool_)
        for row, example in enumerate(examples):
            values = list(example[self.labels_field])
            if not 1 <= len(values) <= self.steps_per_trajectory:
                raise ValueError("process labels are outside the finite step bucket")
            labels[row, : len(values)] = values
            valid[row, : len(values)] = True
        return ProcessRewardBatch(
            inputs=self.processor([example[self.input_field] for example in examples]),
            labels=jnp.asarray(labels),
            valid=jnp.asarray(valid),
        )


__all__ = [
    "ListwiseRewardBatch",
    "ListwiseRewardCollator",
    "PairwiseRewardBatch",
    "PairwiseRewardCollator",
    "PointwiseRewardBatch",
    "PointwiseRewardCollator",
    "ProcessRewardBatch",
    "ProcessRewardCollator",
]
