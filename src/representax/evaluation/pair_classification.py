"""Embedding-distance decisions for labeled pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Encoder, Route, encode
from representax.tasks.classification import PairClassificationBatch

from .similarity import (
    SIMILARITY_FUNCTIONS,
    SimilarityFunction,
    _pairwise_scores,
)


class PairClassificationOutput(eqx.Module):
    left: Float[Array, "pair representation"]
    right: Float[Array, "pair representation"]
    labels: Bool[Array, " pair"]
    valid: Bool[Array, " pair"]


@dataclass(frozen=True, slots=True)
class _PairClassificationAccumulator:
    left: tuple[np.ndarray, ...] = ()
    right: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def _binary_decision_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or not len(labels):
        raise ValueError("pair scores and labels must be matching non-empty vectors")
    if len(np.unique(labels)) != 2:
        raise ValueError("pair classification requires positive and negative examples")
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    ranked = labels[order]
    positives = int(np.sum(ranked))
    cumulative_positive = np.cumsum(ranked)
    cumulative_total = np.arange(1, len(labels) + 1)
    group_end = np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    precision_curve = cumulative_positive[group_end] / cumulative_total[group_end]
    recall_curve = cumulative_positive[group_end] / positives
    average_precision = float(
        np.sum(np.diff(np.r_[0.0, recall_curve]) * precision_curve)
    )

    best_accuracy = 0.0
    best_accuracy_threshold = -1.0
    best_f1 = 0.0
    best_f1_threshold = 0.0
    best_precision = 0.0
    best_recall = 0.0
    positive_so_far = 0
    remaining_negatives = int(np.sum(~ranked))
    for index in range(len(ranked) - 1):
        if ranked[index]:
            positive_so_far += 1
        else:
            remaining_negatives -= 1
        threshold = float((ranked_scores[index] + ranked_scores[index + 1]) / 2)
        accuracy = (positive_so_far + remaining_negatives) / len(labels)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_accuracy_threshold = threshold
        if positive_so_far:
            precision = positive_so_far / (index + 1)
            recall = positive_so_far / positives
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_f1_threshold = threshold
                best_precision = float(precision)
                best_recall = float(recall)

    predicted = scores >= best_f1_threshold
    true_positive = int(np.sum(predicted & labels))
    true_negative = int(np.sum(~predicted & ~labels))
    false_positive = int(np.sum(predicted & ~labels))
    false_negative = int(np.sum(~predicted & labels))
    denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        0.0
        if denominator == 0
        else (true_positive * true_negative - false_positive * false_negative)
        / denominator
    )
    return {
        "accuracy": best_accuracy,
        "accuracy_threshold": best_accuracy_threshold,
        "f1": best_f1,
        "f1_threshold": best_f1_threshold,
        "precision": best_precision,
        "recall": best_recall,
        "average_precision": average_precision,
        "matthews_correlation": float(mcc),
    }


def pair_classification_metrics(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    similarity_functions: Sequence[SimilarityFunction] = SIMILARITY_FUNCTIONS,
) -> dict[str, float]:
    """Select thresholds independently for each declared embedding similarity."""

    left = np.asarray(left)
    right = np.asarray(right)
    labels = np.asarray(labels)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("paired embeddings must share shape [pair, representation]")
    if labels.shape != (left.shape[0],):
        raise ValueError("pair labels must have one value per embedding pair")
    functions = tuple(similarity_functions)
    if not functions or len(set(functions)) != len(functions):
        raise ValueError("similarity functions must be non-empty and unique")
    invalid = tuple(value for value in functions if value not in SIMILARITY_FUNCTIONS)
    if invalid:
        raise ValueError(f"unsupported similarity functions: {invalid}")
    metrics: dict[str, float] = {}
    for function in functions:
        values = _binary_decision_metrics(
            _pairwise_scores(left, right, function), labels
        )
        metrics.update({f"{function}_{name}": value for name, value in values.items()})
    for metric in ("accuracy", "f1", "average_precision"):
        metrics[f"{metric}_max"] = max(
            metrics[f"{function}_{metric}"] for function in functions
        )
    return metrics


@dataclass(frozen=True, slots=True)
class PairClassificationEvaluator:
    """Evaluate binary pair decisions from frozen encoder representations."""

    name: str = "pair_classification"
    similarity_functions: tuple[SimilarityFunction, ...] = SIMILARITY_FUNCTIONS
    left_route: Route = Route.GENERIC
    right_route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.similarity_functions:
            raise ValueError("at least one similarity function is required")
        if len(set(self.similarity_functions)) != len(self.similarity_functions):
            raise ValueError("similarity functions must be unique")
        invalid = tuple(
            value
            for value in self.similarity_functions
            if value not in SIMILARITY_FUNCTIONS
        )
        if invalid:
            raise ValueError(f"unsupported similarity functions: {invalid}")

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/average_precision_max"

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: PairClassificationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> PairClassificationOutput:
        if not isinstance(batch, PairClassificationBatch):
            raise TypeError("pair decisions require a PairClassificationBatch")
        if not isinstance(model, Encoder):
            raise TypeError("pair decisions require an Encoder")
        left_key, right_key = (None, None) if key is None else jax.random.split(key)
        return PairClassificationOutput(
            left=encode(model, batch.left, route=self.left_route, key=left_key),
            right=encode(model, batch.right, route=self.right_route, key=right_key),
            labels=batch.labels.astype(bool),
            valid=batch.valid,
        )

    def initialize(self) -> _PairClassificationAccumulator:
        return _PairClassificationAccumulator()

    def accumulate(
        self,
        accumulator: _PairClassificationAccumulator,
        output: PairClassificationOutput,
    ) -> _PairClassificationAccumulator:
        return _PairClassificationAccumulator(
            left=(*accumulator.left, np.asarray(output.left)),
            right=(*accumulator.right, np.asarray(output.right)),
            labels=(*accumulator.labels, np.asarray(output.labels, dtype=bool)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(
        self, accumulator: _PairClassificationAccumulator
    ) -> Mapping[str, float]:
        if not accumulator.left:
            raise ValueError("pair-classification evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        metrics = pair_classification_metrics(
            np.concatenate(accumulator.left)[valid],
            np.concatenate(accumulator.right)[valid],
            np.concatenate(accumulator.labels)[valid],
            similarity_functions=self.similarity_functions,
        )
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = [
    "PairClassificationEvaluator",
    "PairClassificationOutput",
    "pair_classification_metrics",
]
