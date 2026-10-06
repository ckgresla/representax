"""Fixed-shape teacher targets for representation distillation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from representax.tasks._batch import payload_row_count


def _valid_rows(
    row_count: int,
    valid: Bool[Array, " batch"] | None,
) -> Bool[Array, " batch"]:
    if valid is None:
        return jnp.ones((row_count,), dtype=jnp.bool_)
    return jnp.asarray(valid, dtype=jnp.bool_)


class EmbeddingDistillationBatch(eqx.Module):
    """One or more student inputs with a teacher embedding for each column."""

    inputs: tuple[Any, ...]
    teacher_embeddings: Float[Array, "column batch teacher"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError(
                "embedding distillation requires at least one input column"
            )
        if self.teacher_embeddings.ndim != 3 or not jnp.issubdtype(
            self.teacher_embeddings.dtype, jnp.floating
        ):
            raise TypeError(
                "teacher_embeddings must have shape [column, batch, dimension]"
            )
        column_count, row_count, teacher_dimension = self.teacher_embeddings.shape
        if column_count != len(self.inputs) or teacher_dimension == 0:
            raise ValueError(
                "teacher embeddings must match input columns and be non-empty"
            )
        if self.valid.shape != (row_count,) or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector matching batch rows")
        for index, payload in enumerate(self.inputs):
            if payload_row_count(payload, name=f"inputs[{index}]") != row_count:
                raise ValueError(
                    "every input column must contain one payload per batch row"
                )


def embedding_distillation_batch(
    *,
    inputs: Sequence[Any],
    teacher_embeddings: Float[Array, "batch teacher"]
    | Float[Array, "batch column teacher"],
    valid: Bool[Array, " batch"] | None = None,
) -> EmbeddingDistillationBatch:
    """Normalize broadcast or per-column teacher embeddings to column-major form."""

    resolved_inputs = tuple(inputs)
    if not resolved_inputs:
        raise ValueError("embedding distillation requires at least one input column")
    row_count = payload_row_count(resolved_inputs[0], name="inputs[0]")
    targets = jnp.asarray(teacher_embeddings, dtype=jnp.float32)
    if targets.ndim == 2:
        if targets.shape[0] != row_count:
            raise ValueError("teacher embeddings must match batch rows")
        targets = jnp.broadcast_to(
            targets[None, :, :],
            (len(resolved_inputs), *targets.shape),
        )
    elif targets.ndim == 3:
        if targets.shape[:2] != (row_count, len(resolved_inputs)):
            raise ValueError(
                "per-column teacher embeddings must have shape "
                "[batch, column, dimension]"
            )
        targets = jnp.transpose(targets, (1, 0, 2))
    else:
        raise ValueError("teacher embeddings must be a matrix or rank-three tensor")
    return EmbeddingDistillationBatch(
        inputs=resolved_inputs,
        teacher_embeddings=targets,
        valid=_valid_rows(row_count, valid),
    )


class MarginDistillationBatch(eqx.Module):
    """Query, positive, negatives, and teacher positive-minus-negative margins."""

    query: Any
    positive: Any
    negatives: tuple[Any, ...]
    teacher_margins: Float[Array, "batch negative"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if not self.negatives:
            raise ValueError("margin distillation requires at least one negative")
        if self.teacher_margins.ndim != 2 or not jnp.issubdtype(
            self.teacher_margins.dtype, jnp.floating
        ):
            raise TypeError("teacher_margins must be a floating-point matrix")
        row_count, negative_count = self.teacher_margins.shape
        if negative_count != len(self.negatives):
            raise ValueError("teacher margins must contain one column per negative")
        if self.valid.shape != (row_count,) or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector matching batch rows")
        payloads = (
            ("query", self.query),
            ("positive", self.positive),
            *(
                (f"negatives[{index}]", payload)
                for index, payload in enumerate(self.negatives)
            ),
        )
        for name, payload in payloads:
            if payload_row_count(payload, name=name) != row_count:
                raise ValueError(f"{name} must contain one payload per batch row")


def margin_distillation_batch(
    *,
    query: Any,
    positive: Any,
    negatives: Sequence[Any],
    teacher_margins: Float[Array, "batch negative"] | None = None,
    teacher_scores: Float[Array, "batch candidate"] | None = None,
    valid: Bool[Array, " batch"] | None = None,
) -> MarginDistillationBatch:
    """Build canonical margins from direct margins or positive-and-negative scores."""

    resolved_negatives = tuple(negatives)
    if bool(teacher_margins is None) == bool(teacher_scores is None):
        raise ValueError("provide exactly one of teacher_margins or teacher_scores")
    row_count = payload_row_count(query, name="query")
    if teacher_scores is not None:
        scores = jnp.asarray(teacher_scores, dtype=jnp.float32)
        if scores.shape != (row_count, len(resolved_negatives) + 1):
            raise ValueError(
                "teacher scores must contain positive then negative columns"
            )
        margins = scores[:, :1] - scores[:, 1:]
    else:
        margins = jnp.asarray(teacher_margins, dtype=jnp.float32)
        if margins.ndim == 1:
            margins = margins[:, None]
        if margins.shape != (row_count, len(resolved_negatives)):
            raise ValueError("teacher margins must contain one column per negative")
    return MarginDistillationBatch(
        query=query,
        positive=positive,
        negatives=resolved_negatives,
        teacher_margins=margins,
        valid=_valid_rows(row_count, valid),
    )


class DistributionDistillationBatch(eqx.Module):
    """Query-candidate payloads and teacher scores defining a distribution."""

    query: Any
    candidates: tuple[Any, ...]
    teacher_scores: Float[Array, "batch candidate"]
    valid: Bool[Array, " batch"]

    def __post_init__(self) -> None:
        if len(self.candidates) < 2:
            raise ValueError(
                "distribution distillation requires at least two candidates"
            )
        if self.teacher_scores.ndim != 2 or not jnp.issubdtype(
            self.teacher_scores.dtype, jnp.floating
        ):
            raise TypeError("teacher_scores must be a floating-point matrix")
        row_count, candidate_count = self.teacher_scores.shape
        if candidate_count != len(self.candidates):
            raise ValueError("teacher scores must contain one column per candidate")
        if self.valid.shape != (row_count,) or self.valid.dtype != jnp.bool_:
            raise TypeError("valid must be a boolean vector matching batch rows")
        if payload_row_count(self.query, name="query") != row_count:
            raise ValueError("query must contain one payload per batch row")
        for index, payload in enumerate(self.candidates):
            if payload_row_count(payload, name=f"candidates[{index}]") != row_count:
                raise ValueError(
                    "every candidate must contain one payload per batch row"
                )


def distribution_distillation_batch(
    *,
    query: Any,
    candidates: Sequence[Any],
    teacher_scores: Float[Array, "batch candidate"],
    valid: Bool[Array, " batch"] | None = None,
) -> DistributionDistillationBatch:
    """Build a teacher-score distribution batch."""

    row_count = payload_row_count(query, name="query")
    return DistributionDistillationBatch(
        query=query,
        candidates=tuple(candidates),
        teacher_scores=jnp.asarray(teacher_scores, dtype=jnp.float32),
        valid=_valid_rows(row_count, valid),
    )


__all__ = [
    "DistributionDistillationBatch",
    "EmbeddingDistillationBatch",
    "MarginDistillationBatch",
    "distribution_distillation_batch",
    "embedding_distillation_batch",
    "margin_distillation_batch",
]
