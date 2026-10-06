"""Pointwise scorer regression and embedding-MSE evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Encoder, Route, encode, score_logits
from representax.tasks.cross_encoder import PointwiseBatch
from representax.tasks.distillation import EmbeddingDistillationBatch
from representax.tasks.reward_modeling import PointwiseRewardBatch


class RegressionBatchOutput(eqx.Module):
    predictions: Float[Array, "batch value"]
    targets: Float[Array, "batch value"]
    valid: Bool[Array, " batch"]


@dataclass(frozen=True, slots=True)
class _RegressionAccumulator:
    predictions: tuple[np.ndarray, ...] = ()
    targets: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or not predictions.size:
        raise ValueError("regression predictions and targets must be aligned")
    error = predictions - targets

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        left = left.reshape(-1) - np.mean(left)
        right = right.reshape(-1) - np.mean(right)
        return float(
            np.dot(left, right)
            / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12)
        )

    def ranks(values: np.ndarray) -> np.ndarray:
        flat = values.reshape(-1)
        order = np.argsort(flat, kind="stable")
        ranked = np.empty(len(order), dtype=np.float64)
        sorted_values = flat[order]
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranked[order[start:stop]] = (start + stop - 1) / 2
            start = stop
        return ranked

    return {
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "pearson": correlation(predictions, targets),
        "spearman": correlation(ranks(predictions), ranks(targets)),
    }


@dataclass(frozen=True, slots=True)
class MSEEvaluator:
    """Evaluate scalar scorer targets or teacher embedding reconstruction."""

    name: str = "mse"
    route: Route = Route.GENERIC

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/mse"

    def evaluate_batch(
        self,
        model: Any,
        batch: PointwiseBatch | PointwiseRewardBatch | EmbeddingDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> RegressionBatchOutput:
        if isinstance(batch, (PointwiseBatch, PointwiseRewardBatch)):
            values = jnp.asarray(score_logits(model, batch.inputs, key=key))
            if values.ndim == 1:
                values = values[:, None]
            targets = jnp.asarray(batch.labels, dtype=jnp.float32)[:, None]
            valid = batch.valid
        elif isinstance(batch, EmbeddingDistillationBatch):
            if not isinstance(model, Encoder):
                raise TypeError("embedding MSE requires an Encoder")
            keys = (
                (None,) * len(batch.inputs)
                if key is None
                else tuple(jax.random.split(key, len(batch.inputs)))
            )
            columns = tuple(
                encode(model, inputs, route=self.route, key=column_key)
                for inputs, column_key in zip(batch.inputs, keys, strict=True)
            )
            values = jnp.concatenate(columns, axis=-1)
            targets = jnp.concatenate(tuple(batch.teacher_embeddings), axis=-1)
            valid = batch.valid
        else:
            raise TypeError("unsupported MSE evaluation batch")
        if values.shape != targets.shape:
            raise ValueError("MSE predictions and targets must have identical shapes")
        return RegressionBatchOutput(
            predictions=values,
            targets=targets,
            valid=valid,
        )

    def initialize(self) -> _RegressionAccumulator:
        return _RegressionAccumulator()

    def accumulate(
        self,
        accumulator: _RegressionAccumulator,
        output: RegressionBatchOutput,
    ) -> _RegressionAccumulator:
        return _RegressionAccumulator(
            predictions=(*accumulator.predictions, np.asarray(output.predictions)),
            targets=(*accumulator.targets, np.asarray(output.targets)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _RegressionAccumulator) -> Mapping[str, float]:
        if not accumulator.predictions:
            raise ValueError("MSE evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        metrics = regression_metrics(
            np.concatenate(accumulator.predictions)[valid],
            np.concatenate(accumulator.targets)[valid],
        )
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = ["MSEEvaluator", "RegressionBatchOutput", "regression_metrics"]
