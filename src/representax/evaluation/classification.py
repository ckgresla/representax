"""Classification evaluation for native scorer and pair-classifier batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import score_logits
from representax.tasks.classification import PairClassificationBatch
from representax.tasks.classification.losses import pair_features
from representax.tasks.classification.task import SoftmaxClassificationTask
from representax.tasks.cross_encoder import PointwiseBatch


class ClassificationBatchOutput(eqx.Module):
    logits: Float[Array, "batch output"]
    labels: Int[Array, " batch"]
    valid: Bool[Array, " batch"]


@dataclass(frozen=True, slots=True)
class _ClassificationAccumulator:
    logits: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Return accuracy plus macro precision, recall, and F1."""

    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    binary_scores = None
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        binary_scores = logits.reshape(-1)
        predictions = (binary_scores >= 0).astype(np.int64)
        class_count = 2
    elif logits.ndim == 2:
        predictions = np.argmax(logits, axis=-1)
        class_count = logits.shape[1]
    else:
        raise ValueError("classification logits must be a vector or matrix")
    if labels.shape != predictions.shape or not len(labels):
        raise ValueError("classification labels must match non-empty predictions")
    precisions = []
    recalls = []
    f1s = []
    for label in range(class_count):
        true_positive = np.sum((predictions == label) & (labels == label))
        false_positive = np.sum((predictions == label) & (labels != label))
        false_negative = np.sum((predictions != label) & (labels == label))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1s.append(float(f1))
    metrics = {
        "accuracy": float(np.mean(predictions == labels)),
        "precision_macro": float(np.mean(precisions)),
        "recall_macro": float(np.mean(recalls)),
        "f1_macro": float(np.mean(f1s)),
    }
    if binary_scores is not None:
        order = np.argsort(-binary_scores, kind="stable")
        ranked_labels = labels[order] == 1
        positives = max(int(np.sum(ranked_labels)), 1)
        cumulative = np.cumsum(ranked_labels)
        metrics["average_precision"] = float(
            np.sum((cumulative / np.arange(1, len(labels) + 1)) * ranked_labels)
            / positives
        )
        best_f1 = 0.0
        best_accuracy = 0.0
        for threshold in np.concatenate(
            ((np.asarray((np.inf,))), np.unique(binary_scores)[::-1])
        ):
            threshold_predictions = binary_scores >= threshold
            true_positive = np.sum(threshold_predictions & (labels == 1))
            false_positive = np.sum(threshold_predictions & (labels != 1))
            false_negative = np.sum(~threshold_predictions & (labels == 1))
            precision = true_positive / max(true_positive + false_positive, 1)
            recall = true_positive / max(true_positive + false_negative, 1)
            f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
            best_f1 = max(best_f1, float(f1))
            best_accuracy = max(
                best_accuracy, float(np.mean(threshold_predictions == labels))
            )
        metrics["f1_best"] = best_f1
        metrics["accuracy_best"] = best_accuracy
    return metrics


@dataclass(frozen=True, slots=True)
class ClassificationEvaluator:
    """Evaluate pointwise scorers or the native pair-classification task."""

    name: str = "classification"
    task: SoftmaxClassificationTask | None = None

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/accuracy"

    def evaluate_batch(
        self,
        model: Any,
        batch: PointwiseBatch | PairClassificationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> ClassificationBatchOutput:
        if isinstance(batch, PointwiseBatch):
            logits = score_logits(model, batch.inputs, key=key)
            labels = batch.labels
            valid = batch.valid
        elif isinstance(batch, PairClassificationBatch) and self.task is not None:
            left, right = self.task.representations(model, batch, key=key)
            features = pair_features(
                left,
                right,
                concatenate_representations=self.task.concatenate_representations,
                concatenate_difference=self.task.concatenate_difference,
                concatenate_product=self.task.concatenate_product,
            )
            logits = model.classify(features)
            labels = batch.labels
            valid = batch.valid
        else:
            raise TypeError(
                "classification requires PointwiseBatch or a configured "
                "SoftmaxClassificationTask"
            )
        values = jnp.asarray(logits)
        if values.ndim == 1:
            values = values[:, None]
        return ClassificationBatchOutput(
            logits=values,
            labels=jnp.asarray(labels, dtype=jnp.int32),
            valid=valid,
        )

    def initialize(self) -> _ClassificationAccumulator:
        return _ClassificationAccumulator()

    def accumulate(
        self,
        accumulator: _ClassificationAccumulator,
        output: ClassificationBatchOutput,
    ) -> _ClassificationAccumulator:
        return _ClassificationAccumulator(
            logits=(*accumulator.logits, np.asarray(output.logits)),
            labels=(*accumulator.labels, np.asarray(output.labels)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _ClassificationAccumulator) -> Mapping[str, float]:
        if not accumulator.logits:
            raise ValueError("classification evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        metrics = classification_metrics(
            np.concatenate(accumulator.logits)[valid],
            np.concatenate(accumulator.labels)[valid],
        )
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = [
    "ClassificationBatchOutput",
    "ClassificationEvaluator",
    "classification_metrics",
]
