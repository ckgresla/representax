"""Shared batches and host reductions for frozen representation evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import Encoder, Route, encode
from representax.tasks._batch import payload_row_count


class EvaluationSplit(IntEnum):
    """Stable split identifiers carried through compiled embedding extraction."""

    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class LabeledEvaluationBatch(eqx.Module):
    """Model-native examples with labels and explicit train/validation/test splits."""

    examples: Any
    labels: Int[Array, " example"]
    splits: Int[Array, " example"]
    valid: Bool[Array, " example"]

    def __post_init__(self) -> None:
        if self.labels.ndim != 1 or not jnp.issubdtype(self.labels.dtype, jnp.integer):
            raise TypeError("evaluation labels must be an integer vector")
        if self.splits.shape != self.labels.shape or not jnp.issubdtype(
            self.splits.dtype, jnp.integer
        ):
            raise TypeError("evaluation splits must be a matching integer vector")
        if self.valid.shape != self.labels.shape or self.valid.dtype != jnp.bool_:
            raise TypeError("evaluation valid must be a matching boolean vector")
        if payload_row_count(self.examples, name="examples") != self.labels.shape[0]:
            raise ValueError("examples must contain one row per evaluation label")


def labeled_evaluation_batch(
    *,
    examples: Any,
    labels: Int[Array, " example"],
    split: EvaluationSplit | Int[Array, " example"],
    valid: Bool[Array, " example"] | None = None,
) -> LabeledEvaluationBatch:
    """Build a fixed-shape labeled batch for a frozen-representation protocol."""

    resolved_labels = jnp.asarray(labels, dtype=jnp.int32)
    if isinstance(split, EvaluationSplit):
        splits = jnp.full(resolved_labels.shape, int(split), dtype=jnp.int32)
    else:
        splits = jnp.asarray(split, dtype=jnp.int32)
    if valid is None:
        valid = jnp.ones(resolved_labels.shape, dtype=jnp.bool_)
    return LabeledEvaluationBatch(
        examples=examples,
        labels=resolved_labels,
        splits=splits,
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


class LabeledEmbeddingOutput(eqx.Module):
    embeddings: Float[Array, "example representation"]
    labels: Int[Array, " example"]
    splits: Int[Array, " example"]
    valid: Bool[Array, " example"]


@dataclass(frozen=True, slots=True)
class LabeledEmbeddingAccumulator:
    embeddings: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    splits: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def evaluate_labeled_embeddings(
    model: eqx.Module,
    batch: LabeledEvaluationBatch,
    *,
    route: Route,
    key: PRNGKeyArray | None,
) -> LabeledEmbeddingOutput:
    if not isinstance(batch, LabeledEvaluationBatch):
        raise TypeError("frozen representation evaluation requires a labeled batch")
    if not isinstance(model, Encoder):
        raise TypeError("frozen representation evaluation requires an Encoder")
    return LabeledEmbeddingOutput(
        embeddings=encode(model, batch.examples, route=route, key=key),
        labels=batch.labels,
        splits=batch.splits,
        valid=batch.valid,
    )


def accumulate_labeled_embeddings(
    accumulator: LabeledEmbeddingAccumulator,
    output: LabeledEmbeddingOutput,
) -> LabeledEmbeddingAccumulator:
    return LabeledEmbeddingAccumulator(
        embeddings=(*accumulator.embeddings, np.asarray(output.embeddings)),
        labels=(*accumulator.labels, np.asarray(output.labels)),
        splits=(*accumulator.splits, np.asarray(output.splits)),
        valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
    )


def materialize_labeled_embeddings(
    accumulator: LabeledEmbeddingAccumulator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not accumulator.embeddings:
        raise ValueError("representation evaluation received no batches")
    valid = np.concatenate(accumulator.valid)
    embeddings = np.concatenate(accumulator.embeddings)[valid]
    labels = np.concatenate(accumulator.labels)[valid].astype(np.int64, copy=False)
    splits = np.concatenate(accumulator.splits)[valid].astype(np.int64, copy=False)
    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("representation evaluation requires non-empty 2D embeddings")
    if not np.all(np.isfinite(embeddings)):
        raise ValueError("representation embeddings must be finite")
    return embeddings, labels, splits


def normalize_embeddings(
    embeddings: np.ndarray,
    normalization: Literal["none", "l2"],
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if normalization == "none":
        return values
    if normalization != "l2":
        raise ValueError(f"unsupported embedding normalization {normalization!r}")
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, np.finfo(np.float64).eps)


def linear_probe_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    *,
    inverse_regularization: Sequence[float],
    normalization: Literal["none", "l2"] = "l2",
    max_iterations: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Select and fit a deterministic multinomial logistic-regression probe."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as error:  # pragma: no cover - exercised by package smoke
        raise ImportError(
            "classification probes require the mandatory scikit-learn dependency; "
            "reinstall Representax"
        ) from error

    values = normalize_embeddings(embeddings, normalization)
    labels = np.asarray(labels, dtype=np.int64)
    splits = np.asarray(splits, dtype=np.int64)
    masks = {
        split: splits == int(split)
        for split in (
            EvaluationSplit.TRAIN,
            EvaluationSplit.VALIDATION,
            EvaluationSplit.TEST,
        )
    }
    if any(not np.any(mask) for mask in masks.values()):
        raise ValueError("linear probe requires non-empty train, validation, and test")
    candidates = tuple(float(value) for value in inverse_regularization)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("inverse_regularization must contain positive values")

    best_c = candidates[0]
    best_validation = -np.inf
    for value in candidates:
        probe = LogisticRegression(
            C=value,
            max_iter=max_iterations,
            random_state=seed,
            solver="lbfgs",
        )
        probe.fit(
            values[masks[EvaluationSplit.TRAIN]],
            labels[masks[EvaluationSplit.TRAIN]],
        )
        validation = float(
            probe.score(
                values[masks[EvaluationSplit.VALIDATION]],
                labels[masks[EvaluationSplit.VALIDATION]],
            )
        )
        if validation > best_validation:
            best_validation = validation
            best_c = value

    fit_mask = masks[EvaluationSplit.TRAIN] | masks[EvaluationSplit.VALIDATION]
    probe = LogisticRegression(
        C=best_c,
        max_iter=max_iterations,
        random_state=seed,
        solver="lbfgs",
    )
    probe.fit(values[fit_mask], labels[fit_mask])
    test_labels = labels[masks[EvaluationSplit.TEST]]
    predictions = probe.predict(values[masks[EvaluationSplit.TEST]])
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "f1_macro": float(f1_score(test_labels, predictions, average="macro")),
        "validation_accuracy": best_validation,
        "selected_inverse_regularization": best_c,
    }


__all__ = [
    "EvaluationSplit",
    "LabeledEmbeddingAccumulator",
    "LabeledEmbeddingOutput",
    "LabeledEvaluationBatch",
    "accumulate_labeled_embeddings",
    "evaluate_labeled_embeddings",
    "labeled_evaluation_batch",
    "linear_probe_metrics",
    "materialize_labeled_embeddings",
    "normalize_embeddings",
]
