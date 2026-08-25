"""Frozen-embedding linear-probe evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
from jaxtyping import PRNGKeyArray

from representax.core import Route

from .representation import (
    LabeledEmbeddingAccumulator,
    LabeledEmbeddingOutput,
    LabeledEvaluationBatch,
    accumulate_labeled_embeddings,
    evaluate_labeled_embeddings,
    linear_probe_metrics,
    materialize_labeled_embeddings,
)


@dataclass(frozen=True, slots=True)
class ClassificationProbeEvaluator:
    """Measure linearly accessible label information in frozen embeddings."""

    name: str = "classification_probe"
    inverse_regularization: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    normalization: Literal["none", "l2"] = "l2"
    max_iterations: int = 1000
    seed: int = 0
    route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.inverse_regularization or any(
            value <= 0 for value in self.inverse_regularization
        ):
            raise ValueError("inverse_regularization must contain positive values")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/accuracy"

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
        metrics = linear_probe_metrics(
            embeddings,
            labels,
            splits,
            inverse_regularization=self.inverse_regularization,
            normalization=self.normalization,
            max_iterations=self.max_iterations,
            seed=self.seed,
        )
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = ["ClassificationProbeEvaluator"]
