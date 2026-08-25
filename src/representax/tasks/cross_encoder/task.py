"""Shared native task implementations for cross-encoders and rerankers."""

from __future__ import annotations

from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import LossOutput, Scorer, score_logits

from .batch import (
    CrossMNRBatch,
    ListwiseRankingBatch,
    PairwiseRankingBatch,
    PointwiseBatch,
)
from .config import LambdaWeighting, ReductionLog, ScoreActivation
from .losses import (
    activate_scores,
    binary_cross_entropy,
    cross_mnr_loss,
    lambda_loss,
    list_mle_loss,
    listnet_loss,
    multiclass_cross_entropy,
    ranknet_loss,
    score_mse,
)

PointwiseObjective = Literal["binary_cross_entropy", "cross_entropy", "mse"]
ListwiseObjective = Literal["ranknet", "lambda", "listnet", "list_mle"]


def _single_score(logits: Array) -> Array:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError("this objective requires one score logit per input")


def _flatten_candidate_inputs(inputs, shape: tuple[int, ...]):
    if len(shape) != 2:
        raise ValueError("listwise labels must have [query, document] shape")

    def flatten(value):
        if not isinstance(value, jax.Array):
            return value
        if value.shape[:2] != shape:
            raise ValueError("listwise input leaves must align with labels")
        return value.reshape((shape[0] * shape[1], *value.shape[2:]))

    return jax.tree.map(flatten, inputs)


def _score_candidate_grid(
    model: Scorer,
    batch: CrossMNRBatch,
    *,
    key: PRNGKeyArray | None,
    chunk_size: int | None,
) -> Array:
    shape = batch.valid.shape
    flattened = _flatten_candidate_inputs(batch.inputs, shape)
    pair_count = shape[0] * shape[1]
    if chunk_size is None or chunk_size >= pair_count:
        return _single_score(score_logits(model, flattened, key=key)).reshape(shape)
    padding = (-pair_count) % chunk_size

    def chunk(value):
        if not isinstance(value, jax.Array):
            return value
        if padding:
            repeated = jnp.broadcast_to(value[-1], (padding, *value.shape[1:]))
            value = jnp.concatenate((value, repeated), axis=0)
        return value.reshape((-1, chunk_size, *value.shape[1:]))

    chunks = jax.tree.map(chunk, flattened)
    chunk_count = jax.tree.leaves(chunks)[0].shape[0]
    keys = None if key is None else jax.random.split(key, chunk_count)

    if keys is None:

        def body(_, inputs):
            return None, _single_score(score_logits(model, inputs))

        scan_inputs = chunks
    else:

        def body(_, values):
            inputs, chunk_key = values
            return None, _single_score(score_logits(model, inputs, key=chunk_key))

        scan_inputs = (chunks, keys)
    rematerialized = jax.checkpoint(
        body,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    _, values = jax.lax.scan(rematerialized, None, scan_inputs)
    return values.reshape(-1)[:pair_count].reshape(shape)


class PointwiseScoringTask(eqx.Module):
    """Binary classification, multiclass classification, or score regression."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "accuracy": "mean",
        "mse": "mean",
        "valid_rows": "sum",
    }

    objective: PointwiseObjective = eqx.field(static=True)
    activation: ScoreActivation = eqx.field(static=True, default="identity")
    positive_weight: float | None = eqx.field(static=True, default=None)

    def accumulation_weight(self, batch: PointwiseBatch) -> Array:
        """Return the number of rows in this exact mean reduction."""

        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: PointwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if not isinstance(batch, PointwiseBatch):
            raise TypeError("pointwise scoring requires PointwiseBatch")
        logits = score_logits(model, batch.inputs, key=key)
        if self.objective == "cross_entropy":
            if not jnp.issubdtype(batch.labels.dtype, jnp.integer):
                raise TypeError("cross-entropy labels must be integers")
            loss = multiclass_cross_entropy(
                logits,
                batch.labels,
                batch.valid,
            )
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.sum(
                jnp.where(batch.valid, predictions == batch.labels, False)
            ) / jnp.maximum(jnp.sum(batch.valid), 1).astype(jnp.float32)
            metrics = {"accuracy": accuracy}
        else:
            if not jnp.issubdtype(batch.labels.dtype, jnp.floating):
                raise TypeError(f"{self.objective} labels must be floating point")
            values = activate_scores(_single_score(logits), self.activation)
            if self.objective == "binary_cross_entropy":
                loss = binary_cross_entropy(
                    values,
                    batch.labels,
                    batch.valid,
                    positive_weight=self.positive_weight,
                )
                predictions = (
                    values >= 0 if self.activation == "identity" else values >= 0.5
                )
                accuracy = jnp.sum(
                    jnp.where(batch.valid, predictions == (batch.labels >= 0.5), False)
                ) / jnp.maximum(jnp.sum(batch.valid), 1).astype(jnp.float32)
                metrics = {"accuracy": accuracy}
            elif self.objective == "mse":
                loss = score_mse(values, batch.labels, batch.valid)
                metrics = {"mse": loss}
            else:
                raise ValueError(f"unsupported pointwise objective {self.objective!r}")
        return LossOutput(
            loss=loss,
            metrics={**metrics, "valid_rows": jnp.sum(batch.valid)},
        )


class MarginMSETask(eqx.Module):
    """Match a teacher or gold positive-minus-negative score margin."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "margin_mse": "mean",
        "valid_rows": "sum",
    }

    activation: ScoreActivation = eqx.field(static=True, default="identity")

    def accumulation_weight(self, batch: PairwiseRankingBatch) -> Array:
        """Return the number of pairs in this exact mean reduction."""

        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: PairwiseRankingBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if not isinstance(batch, PairwiseRankingBatch):
            raise TypeError("margin MSE requires PairwiseRankingBatch")
        positive_key = negative_key = None
        if key is not None:
            positive_key, negative_key = jax.random.split(key)
        positive = activate_scores(
            _single_score(score_logits(model, batch.positive, key=positive_key)),
            self.activation,
        )
        negative = activate_scores(
            _single_score(score_logits(model, batch.negative, key=negative_key)),
            self.activation,
        )
        margin = positive - negative
        loss = score_mse(margin, batch.margins, batch.valid)
        return LossOutput(
            loss=loss,
            metrics={"margin_mse": loss, "valid_rows": jnp.sum(batch.valid)},
        )


class ListwiseScoringTask(eqx.Module):
    """RankNet, ListNet, or (position-aware) ListMLE over candidate lists."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "ranknet": "mean",
        "lambda": "mean",
        "listnet": "mean",
        "list_mle": "mean",
        "valid_candidates": "sum",
    }

    objective: ListwiseObjective = eqx.field(static=True)
    activation: ScoreActivation = eqx.field(static=True, default="identity")
    sigma: float = eqx.field(static=True, default=1.0)
    k: int | None = eqx.field(static=True, default=None)
    reduction_log: ReductionLog = eqx.field(static=True, default="binary")
    weighting: LambdaWeighting = eqx.field(static=True, default="ndcg_loss2pp")
    mu: float = eqx.field(static=True, default=10.0)
    respect_input_order: bool = eqx.field(static=True, default=True)
    position_aware: bool = eqx.field(static=True, default=False)

    @property
    def supports_gradient_accumulation(self) -> bool:
        """Whether query microbatches preserve this objective exactly."""

        return self.objective in {"listnet", "list_mle"}

    def accumulation_weight(self, batch: ListwiseRankingBatch) -> Array:
        """Return the number of query lists in this exact mean reduction."""

        return jnp.sum(jnp.any(batch.valid, axis=-1)).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: ListwiseRankingBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if not isinstance(batch, ListwiseRankingBatch):
            raise TypeError("listwise scoring requires ListwiseRankingBatch")
        flattened = activate_scores(
            _single_score(
                score_logits(
                    model,
                    _flatten_candidate_inputs(batch.inputs, batch.labels.shape),
                    key=key,
                )
            ),
            self.activation,
        )
        scores = flattened.reshape(batch.labels.shape)
        if self.objective == "ranknet":
            loss = ranknet_loss(
                scores,
                batch.labels,
                batch.valid,
                sigma=self.sigma,
                k=self.k,
                reduction_log=self.reduction_log,
            )
        elif self.objective == "lambda":
            loss = lambda_loss(
                scores,
                batch.labels,
                batch.valid,
                weighting=self.weighting,
                k=self.k,
                sigma=self.sigma,
                reduction_log=self.reduction_log,
                mu=self.mu,
            )
        elif self.objective == "listnet":
            loss = listnet_loss(scores, batch.labels, batch.valid)
        elif self.objective == "list_mle":
            loss = list_mle_loss(
                scores,
                batch.labels,
                batch.valid,
                respect_input_order=self.respect_input_order,
                position_aware=self.position_aware,
            )
        else:
            raise ValueError(f"unsupported listwise objective {self.objective!r}")
        return LossOutput(
            loss=loss,
            metrics={
                self.objective: loss,
                "valid_candidates": jnp.sum(batch.valid),
            },
        )


class CrossMNRTask(eqx.Module):
    """Exact cross-encoder InfoNCE over every candidate in the global batch."""

    activation: ScoreActivation = eqx.field(static=True, default="sigmoid")
    scale: float = eqx.field(static=True, default=10.0)

    def loss_with_chunk_size(
        self,
        model: Scorer,
        batch: CrossMNRBatch,
        *,
        key: PRNGKeyArray | None = None,
        chunk_size: int | None = None,
    ) -> LossOutput:
        if not isinstance(batch, CrossMNRBatch):
            raise TypeError("cross MNR requires CrossMNRBatch")
        logits = _score_candidate_grid(model, batch, key=key, chunk_size=chunk_size)
        loss = cross_mnr_loss(
            logits,
            batch.positive_indices,
            batch.valid,
            activation=self.activation,
            scale=self.scale,
        )
        scores = jnp.where(
            batch.valid,
            activate_scores(logits, self.activation) * self.scale,
            -jnp.inf,
        )
        query_valid = jnp.any(batch.valid, axis=-1)
        accuracy = jnp.sum(
            jnp.where(
                query_valid,
                jnp.argmax(scores, axis=-1) == batch.positive_indices,
                False,
            )
        ) / jnp.maximum(jnp.sum(query_valid), 1).astype(jnp.float32)
        return LossOutput(
            loss=loss,
            metrics={"accuracy": accuracy, "valid_queries": jnp.sum(query_valid)},
        )

    def loss(
        self,
        model: Scorer,
        batch: CrossMNRBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        return self.loss_with_chunk_size(model, batch, key=key)


__all__ = [
    "CrossMNRTask",
    "ListwiseScoringTask",
    "MarginMSETask",
    "PointwiseScoringTask",
]
