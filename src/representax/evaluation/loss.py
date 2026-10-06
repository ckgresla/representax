"""Task-loss evaluation through the general evaluator protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import Task, evaluate_loss
from representax.tasks._batch import payload_row_count


class LossBatchOutput(eqx.Module):
    """One scalar task loss and its exact aggregation weight."""

    loss: Float[Array, ""]
    weight: Float[Array, ""]


@dataclass(frozen=True, slots=True)
class _LossAccumulator:
    weighted_loss: float = 0.0
    weight: float = 0.0


@dataclass(frozen=True, slots=True)
class LossEvaluator:
    """Evaluate the configured training objective under ``valid/...``."""

    task: Task[Any]
    name: str = "loss"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossBatchOutput:
        output = evaluate_loss(self.task, model, batch, key=key)
        metric_weight = getattr(self.task, "accumulation_weight", None)
        weight = (
            metric_weight(batch)
            if callable(metric_weight)
            else jnp.asarray(payload_row_count(batch, name="batch"), jnp.float32)
        )
        return LossBatchOutput(loss=output.loss, weight=weight)

    def initialize(self) -> _LossAccumulator:
        return _LossAccumulator()

    def accumulate(
        self,
        accumulator: _LossAccumulator,
        output: LossBatchOutput,
    ) -> _LossAccumulator:
        weight = float(output.weight)
        return _LossAccumulator(
            weighted_loss=accumulator.weighted_loss + float(output.loss) * weight,
            weight=accumulator.weight + weight,
        )

    def finalize(self, accumulator: _LossAccumulator) -> Mapping[str, float]:
        if accumulator.weight <= 0:
            raise ValueError("loss evaluator received no valid examples")
        prefix = "valid" if self.name == "loss" else f"valid/{self.name}"
        return {f"{prefix}/loss": accumulator.weighted_loss / accumulator.weight}


__all__ = ["LossBatchOutput", "LossEvaluator"]
