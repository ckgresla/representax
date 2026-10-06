"""Cross-encoder reranking evaluation over finite candidate sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import score_logits
from representax.tasks.cross_encoder import (
    ListwiseRankingBatch,
    PairwiseRankingBatch,
)

RankingKind = Literal["pairwise", "listwise"]


class RankingBatchOutput(eqx.Module):
    scores: Float[Array, "batch candidate"]
    labels: Float[Array, "batch candidate"]
    valid: Bool[Array, "batch candidate"]
    kind: RankingKind = eqx.field(static=True)


@dataclass(frozen=True, slots=True)
class _RankingAccumulator:
    scores: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()
    kind: RankingKind | None = None


def _scalar_scores(logits: Array) -> Array:
    values = jnp.asarray(logits)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError("ranking models must emit one score per input")


def _flatten(inputs: Any, shape: tuple[int, int]) -> Any:
    return jax.tree.map(
        lambda value: (
            value.reshape((shape[0] * shape[1], *value.shape[2:]))
            if isinstance(value, jax.Array)
            else value
        ),
        inputs,
    )


def _dcg(values: np.ndarray, k: int) -> float:
    return float(
        np.sum((np.power(2.0, values[:k]) - 1.0) / np.log2(np.arange(k) + 2.0))
    )


def ranking_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    *,
    at_k: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute exact row-mean MRR, nDCG, MAP, precision, and recall."""

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("ranking scores, labels, and validity must align")
    rows = []
    for row_scores, row_labels, row_valid in zip(scores, labels, valid, strict=True):
        if not np.any(row_valid):
            continue
        values = row_labels[row_valid]
        order = np.argsort(-row_scores[row_valid], kind="stable")
        rows.append((values[order], np.sort(values)[::-1]))
    if not rows:
        raise ValueError("ranking evaluation received no rows")
    metrics: dict[str, float] = {}
    for k in at_k:
        if k <= 0:
            raise ValueError("ranking cutoffs must be positive")
        reciprocal_ranks = []
        ndcgs = []
        average_precisions = []
        precisions = []
        recalls = []
        for predicted, ideal in rows:
            relevant = predicted > 0
            relevant_count = max(int(np.sum(ideal > 0)), 1)
            reciprocal_ranks.append(
                next(
                    (
                        1.0 / (index + 1)
                        for index, hit in enumerate(relevant[:k])
                        if hit
                    ),
                    0.0,
                )
            )
            ideal_dcg = _dcg(ideal, min(k, len(ideal)))
            ndcgs.append(
                _dcg(predicted, min(k, len(predicted))) / max(ideal_dcg, 1e-12)
            )
            hits = 0
            precision_sum = 0.0
            for rank, hit in enumerate(relevant[:k], start=1):
                if hit:
                    hits += 1
                    precision_sum += hits / rank
            average_precisions.append(precision_sum / min(relevant_count, k))
            precisions.append(hits / k)
            recalls.append(hits / relevant_count)
        metrics.update(
            {
                f"mrr@{k}": float(np.mean(reciprocal_ranks)),
                f"ndcg@{k}": float(np.mean(ndcgs)),
                f"map@{k}": float(np.mean(average_precisions)),
                f"precision@{k}": float(np.mean(precisions)),
                f"recall@{k}": float(np.mean(recalls)),
            }
        )
    return metrics


@dataclass(frozen=True, slots=True)
class RerankingEvaluator:
    """Evaluate pairwise or listwise cross-encoder rankings."""

    name: str = "reranking"
    at_k: tuple[int, ...] = (1, 3, 5, 10)

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/ndcg@{max(self.at_k)}"

    def evaluate_batch(
        self,
        model: Any,
        batch: Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> RankingBatchOutput:
        if isinstance(batch, PairwiseRankingBatch):
            left_inputs = batch.positive
            right_inputs = batch.negative
            keys = (None, None) if key is None else jax.random.split(key)
            left = _scalar_scores(score_logits(model, left_inputs, key=keys[0]))
            right = _scalar_scores(score_logits(model, right_inputs, key=keys[1]))
            scores = jnp.stack((left, right), axis=-1)
            labels = jnp.broadcast_to(jnp.asarray((1.0, 0.0)), scores.shape)
            valid = jnp.broadcast_to(batch.valid[:, None], scores.shape)
            kind: RankingKind = "pairwise"
        elif isinstance(batch, ListwiseRankingBatch):
            labels = batch.labels
            inputs = batch.inputs
            scores = _scalar_scores(
                score_logits(model, _flatten(inputs, labels.shape), key=key)
            ).reshape(labels.shape)
            valid = batch.valid
            kind = "listwise"
        else:
            raise TypeError("reranking requires a pairwise or listwise ranking batch")
        return RankingBatchOutput(scores=scores, labels=labels, valid=valid, kind=kind)

    def initialize(self) -> _RankingAccumulator:
        return _RankingAccumulator()

    def accumulate(
        self, accumulator: _RankingAccumulator, output: RankingBatchOutput
    ) -> _RankingAccumulator:
        if accumulator.kind is not None and accumulator.kind != output.kind:
            raise ValueError("one reranking pass cannot mix batch contracts")
        return _RankingAccumulator(
            scores=(*accumulator.scores, np.asarray(output.scores)),
            labels=(*accumulator.labels, np.asarray(output.labels)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
            kind=output.kind,
        )

    def finalize(self, accumulator: _RankingAccumulator) -> Mapping[str, float]:
        if not accumulator.scores or accumulator.kind is None:
            raise ValueError("reranking evaluation received no batches")
        scores = np.concatenate(accumulator.scores)
        labels = np.concatenate(accumulator.labels)
        valid = np.concatenate(accumulator.valid)
        prefix = f"valid/{self.name}"
        metrics = ranking_metrics(scores, labels, valid, at_k=self.at_k)
        if accumulator.kind == "pairwise":
            active = np.all(valid, axis=-1)
            if not np.any(active):
                raise ValueError("pairwise evaluation received no valid pairs")
            margins = scores[active, 0] - scores[active, 1]
            metrics["pairwise_accuracy"] = float(np.mean(margins > 0))
            metrics["score_margin"] = float(np.mean(margins))
        return {f"{prefix}/{name}": value for name, value in metrics.items()}


__all__ = [
    "RankingBatchOutput",
    "RankingKind",
    "RerankingEvaluator",
    "ranking_metrics",
]
