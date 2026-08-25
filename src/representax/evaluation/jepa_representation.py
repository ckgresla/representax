"""Downstream transfer and collapse diagnostics for JEPA representations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import numpy as np
from jaxtyping import PRNGKeyArray

from representax.core import Route

from .representation import (
    EvaluationSplit,
    LabeledEmbeddingAccumulator,
    LabeledEmbeddingOutput,
    LabeledEvaluationBatch,
    accumulate_labeled_embeddings,
    evaluate_labeled_embeddings,
    linear_probe_metrics,
    materialize_labeled_embeddings,
    normalize_embeddings,
)


def representation_geometry_metrics(embeddings: np.ndarray) -> dict[str, float]:
    """Return scale, rank, and conditioning diagnostics for one representation."""

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("geometry diagnostics require finite 2D embeddings")
    centered = values - np.mean(values, axis=0, keepdims=True)
    feature_std = np.std(values, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    probability = singular / max(float(np.sum(singular)), np.finfo(float).eps)
    effective_rank = float(
        np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
    )
    nonzero = singular[singular > np.finfo(np.float64).eps * singular[0]]
    condition = float("inf") if not len(nonzero) else float(nonzero[0] / nonzero[-1])
    return {
        "feature_std_mean": float(np.mean(feature_std)),
        "feature_std_min": float(np.min(feature_std)),
        "effective_rank": effective_rank,
        "condition_number": condition,
    }


def knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    *,
    neighbors: int,
) -> float:
    """Evaluate cosine k-NN from train/validation references onto test examples."""

    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    values = normalize_embeddings(embeddings, "l2")
    reference = splits != int(EvaluationSplit.TEST)
    test = splits == int(EvaluationSplit.TEST)
    if not np.any(reference) or not np.any(test) or neighbors > int(np.sum(reference)):
        raise ValueError("k-NN requires reference/test rows and enough neighbors")
    similarities = values[test] @ values[reference].T
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbors]
    reference_labels = labels[reference]
    predictions = []
    for row in nearest:
        values_for_row, counts = np.unique(reference_labels[row], return_counts=True)
        predictions.append(values_for_row[np.argmax(counts)])
    return float(np.mean(np.asarray(predictions) == labels[test]))


@dataclass(frozen=True, slots=True)
class JEPARepresentationEvaluator:
    """Measure frozen transfer, nearest-neighbor quality, and representation health."""

    name: str = "jepa_representation"
    inverse_regularization: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    normalization: Literal["none", "l2"] = "l2"
    max_iterations: int = 1000
    neighbors: int = 20
    seed: int = 0
    route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.inverse_regularization or any(
            value <= 0 for value in self.inverse_regularization
        ):
            raise ValueError("inverse_regularization must contain positive values")
        if self.max_iterations <= 0 or self.neighbors <= 0:
            raise ValueError("probe iterations and neighbors must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/linear_probe_accuracy"

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: LabeledEvaluationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LabeledEmbeddingOutput:
        return evaluate_labeled_embeddings(model, batch, route=self.route, key=key)

    def initialize(self) -> LabeledEmbeddingAccumulator:
        return LabeledEmbeddingAccumulator()

    def accumulate(
        self,
        accumulator: LabeledEmbeddingAccumulator,
        output: LabeledEmbeddingOutput,
    ) -> LabeledEmbeddingAccumulator:
        return accumulate_labeled_embeddings(accumulator, output)

    def finalize(self, accumulator: LabeledEmbeddingAccumulator) -> Mapping[str, float]:
        embeddings, labels, splits = materialize_labeled_embeddings(accumulator)
        probe = linear_probe_metrics(
            embeddings,
            labels,
            splits,
            inverse_regularization=self.inverse_regularization,
            normalization=self.normalization,
            max_iterations=self.max_iterations,
            seed=self.seed,
        )
        metrics = {
            "linear_probe_accuracy": probe["accuracy"],
            "linear_probe_f1_macro": probe["f1_macro"],
            "linear_probe_validation_accuracy": probe["validation_accuracy"],
            "linear_probe_selected_inverse_regularization": probe[
                "selected_inverse_regularization"
            ],
            "knn_accuracy": knn_accuracy(
                embeddings, labels, splits, neighbors=self.neighbors
            ),
            **representation_geometry_metrics(embeddings),
        }
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = [
    "JEPARepresentationEvaluator",
    "knn_accuracy",
    "representation_geometry_metrics",
]
