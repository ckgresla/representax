"""Composition of compatible evaluators over one batch stream."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
from jaxtyping import PRNGKeyArray

from .protocol import Evaluator


@dataclass(frozen=True, slots=True)
class SequentialEvaluator:
    """Run several compatible evaluators in one compiled batch traversal."""

    evaluators: tuple[Evaluator[Any, Any], ...]
    primary_metric: str
    name: str = "sequential"

    def __post_init__(self) -> None:
        if not self.evaluators:
            raise ValueError("sequential evaluation requires at least one evaluator")
        if not self.primary_metric.startswith("valid/"):
            raise ValueError("sequential primary_metric must use valid/")

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Any, ...]:
        return tuple(
            evaluator.evaluate_batch(model, batch, key=key)
            for evaluator in self.evaluators
        )

    def initialize(self) -> tuple[Any, ...]:
        return tuple(evaluator.initialize() for evaluator in self.evaluators)

    def accumulate(
        self, accumulator: tuple[Any, ...], output: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return tuple(
            evaluator.accumulate(current, value)
            for evaluator, current, value in zip(
                self.evaluators, accumulator, output, strict=True
            )
        )

    def finalize(self, accumulator: tuple[Any, ...]) -> Mapping[str, float]:
        metrics: dict[str, float] = {}
        for evaluator, current in zip(self.evaluators, accumulator, strict=True):
            values = evaluator.finalize(current)
            overlap = set(metrics) & set(values)
            if overlap:
                raise ValueError(f"sequential metric collision: {sorted(overlap)}")
            metrics.update(values)
        if self.primary_metric not in metrics:
            raise ValueError(
                f"sequential primary metric was not emitted: {self.primary_metric!r}"
            )
        return metrics


__all__ = ["SequentialEvaluator"]
