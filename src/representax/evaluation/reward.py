"""Reward-model evaluation with task-specific metric contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import score_logits
from representax.tasks.reward_modeling import (
    ListwiseRewardBatch,
    PairwiseRewardBatch,
    PointwiseRewardBatch,
    ProcessRewardBatch,
)

from .ranking import _flatten, _scalar_scores, ranking_metrics

RewardEvaluationKind = Literal["pairwise", "listwise", "pointwise", "process"]


class RewardBatchOutput(eqx.Module):
    scores: Float[Array, "batch target"]
    targets: Float[Array, "batch target"]
    valid: Bool[Array, "batch target"]
    kind: RewardEvaluationKind = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class _RewardAccumulator:
    scores: tuple[np.ndarray, ...] = ()
    targets: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()
    kind: RewardEvaluationKind | None = None


@dataclass(frozen=True, slots=True)
class RewardEvaluator:
    """Evaluate exactly one pairwise, listwise, pointwise, or process contract."""

    kind: RewardEvaluationKind = "pairwise"
    name: str = "reward"
    at_k: tuple[int, ...] = (1, 3, 5, 10)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.at_k or any(value <= 0 for value in self.at_k):
            raise ValueError("reward ranking cutoffs must be positive")

    @property
    def primary_metric(self) -> str:
        suffix = {
            "pairwise": "pairwise_accuracy",
            "listwise": f"ndcg@{max(self.at_k)}",
            "pointwise": "mse",
            "process": "accuracy",
        }[self.kind]
        return f"valid/{self.name}/{suffix}"

    def evaluate_batch(
        self,
        model: Any,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> RewardBatchOutput:
        if self.kind == "pairwise" and isinstance(batch, PairwiseRewardBatch):
            keys = (None, None) if key is None else jax.random.split(key)
            chosen = _scalar_scores(score_logits(model, batch.chosen, key=keys[0]))
            rejected = _scalar_scores(score_logits(model, batch.rejected, key=keys[1]))
            scores = jnp.stack((chosen, rejected), axis=-1)
            targets = jnp.broadcast_to(batch.margins[:, None], scores.shape)
            valid = jnp.broadcast_to(batch.valid[:, None], scores.shape)
        elif self.kind == "listwise" and isinstance(batch, ListwiseRewardBatch):
            scores = _scalar_scores(
                score_logits(
                    model,
                    _flatten(batch.candidates, batch.preferences.shape),
                    key=key,
                )
            ).reshape(batch.preferences.shape)
            targets = batch.preferences
            valid = batch.valid
        elif self.kind == "pointwise" and isinstance(batch, PointwiseRewardBatch):
            scores = _scalar_scores(score_logits(model, batch.inputs, key=key))[:, None]
            targets = batch.labels[:, None]
            valid = batch.valid[:, None]
        elif self.kind == "process" and isinstance(batch, ProcessRewardBatch):
            scores = jnp.asarray(score_logits(model, batch.inputs, key=key))
            targets = batch.labels
            valid = batch.valid
        else:
            raise TypeError(
                f"{self.kind} reward evaluation received an incompatible batch"
            )
        return RewardBatchOutput(
            scores=scores,
            targets=targets,
            valid=valid,
            kind=self.kind,
        )

    def initialize(self) -> _RewardAccumulator:
        return _RewardAccumulator()

    def accumulate(
        self,
        accumulator: _RewardAccumulator,
        output: RewardBatchOutput,
    ) -> _RewardAccumulator:
        if accumulator.kind is not None and accumulator.kind != output.kind:
            raise ValueError("one reward evaluation cannot mix batch contracts")
        return _RewardAccumulator(
            scores=(*accumulator.scores, np.asarray(output.scores)),
            targets=(*accumulator.targets, np.asarray(output.targets)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
            kind=output.kind,
        )

    def finalize(self, accumulator: _RewardAccumulator) -> Mapping[str, float]:
        if not accumulator.scores or accumulator.kind is None:
            raise ValueError("reward evaluation received no batches")
        scores = np.concatenate(accumulator.scores)
        targets = np.concatenate(accumulator.targets)
        valid = np.concatenate(accumulator.valid)
        if accumulator.kind == "pairwise":
            active = np.all(valid, axis=-1)
            if not np.any(active):
                raise ValueError("pairwise reward evaluation received no valid pairs")
            chosen = scores[active, 0]
            rejected = scores[active, 1]
            margins = chosen - rejected
            required = targets[active, 0]
            metrics = {
                "pairwise_accuracy": float(np.mean(margins > 0)),
                "margin_accuracy": float(np.mean(margins > required)),
                "score_margin_mean": float(np.mean(margins)),
                "score_margin_std": float(np.std(margins)),
                "chosen_score_mean": float(np.mean(chosen)),
                "rejected_score_mean": float(np.mean(rejected)),
            }
        elif accumulator.kind == "listwise":
            metrics = ranking_metrics(scores, targets, valid, at_k=self.at_k)
            active_rows = np.any(valid, axis=-1)
            masked_scores = np.where(valid, scores, -np.inf)
            masked_targets = np.where(valid, targets, -np.inf)
            metrics["top1_accuracy"] = float(
                np.mean(
                    np.argmax(masked_scores[active_rows], axis=-1)
                    == np.argmax(masked_targets[active_rows], axis=-1)
                )
            )
        else:
            active_scores = scores[valid]
            active_targets = targets[valid]
            if not len(active_scores):
                raise ValueError("reward evaluation received no valid targets")
            error = active_scores - active_targets
            predictions = active_scores >= 0
            binary = active_targets >= 0.5
            metrics = {
                "accuracy": float(np.mean(predictions == binary)),
                "mse": float(np.mean(np.square(error))),
                "mae": float(np.mean(np.abs(error))),
            }
            if accumulator.kind == "process":
                active_rows = np.any(valid, axis=-1)
                correct = (scores >= 0) == (targets >= 0.5)
                metrics["sequence_accuracy"] = float(
                    np.mean(np.all(correct[active_rows] | ~valid[active_rows], axis=-1))
                )
        prefix = f"valid/{self.name}"
        return {f"{prefix}/{name}": value for name, value in metrics.items()}


__all__ = [
    "RewardBatchOutput",
    "RewardEvaluationKind",
    "RewardEvaluator",
]
