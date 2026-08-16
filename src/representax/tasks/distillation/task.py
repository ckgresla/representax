"""Runtime tasks for offline teacher-target distillation."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import Encoder, LossOutput, Route, encode

from .batch import (
    DistributionDistillationBatch,
    EmbeddingDistillationBatch,
    MarginDistillationBatch,
)
from .losses import (
    EmbeddingDistillationDistance,
    ScoreSimilarity,
    distribution_kl_loss_terms,
    embedding_distillation_loss_terms,
    margin_mse_loss_terms,
)


def _keys(
    key: PRNGKeyArray | None,
    count: int,
) -> tuple[PRNGKeyArray | None, ...]:
    if key is None:
        return (None,) * count
    return tuple(jax.random.split(key, count))


def _resolved_routes(routes: tuple[Route, ...], count: int) -> tuple[Route, ...]:
    if len(routes) == 1:
        return routes * count
    if len(routes) != count:
        raise ValueError("routes must contain one shared route or one route per input")
    return routes


class EmbeddingDistillationTask(eqx.Module):
    """Match one or more student representation columns to teacher embeddings."""

    distance: EmbeddingDistillationDistance = eqx.field(static=True, default="cosine")
    routes: tuple[Route, ...] = eqx.field(static=True, default=(Route.GENERIC,))

    def __post_init__(self) -> None:
        if self.distance not in {"mse", "l2", "cosine"}:
            raise ValueError(
                f"unsupported embedding distillation distance {self.distance!r}"
            )
        if not self.routes:
            raise ValueError("embedding distillation requires at least one route")

    def loss(
        self,
        model: Encoder,
        batch: EmbeddingDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        routes = _resolved_routes(self.routes, len(batch.inputs))
        keys = _keys(key, len(batch.inputs))
        embeddings = jnp.stack(
            tuple(
                encode(model, payload, route=route, key=column_key)
                for payload, route, column_key in zip(
                    batch.inputs,
                    routes,
                    keys,
                    strict=True,
                )
            ),
            axis=0,
        )
        return self.loss_from_embeddings(embeddings, batch)

    def loss_from_embeddings(
        self,
        embeddings: Float[Array, "column batch representation"],
        batch: EmbeddingDistillationBatch,
    ) -> LossOutput:
        terms = embedding_distillation_loss_terms(
            embeddings,
            batch.teacher_embeddings,
            valid=batch.valid,
            distance=self.distance,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "distilled_columns": jnp.asarray(embeddings.shape[0]),
                "valid_examples": jnp.sum(batch.valid),
            },
        )


class MarginDistillationTask(eqx.Module):
    """Regress student positive-minus-negative scores onto teacher margins."""

    similarity: ScoreSimilarity = eqx.field(static=True, default="dot")
    query_route: Route = eqx.field(static=True, default=Route.QUERY)
    document_route: Route = eqx.field(static=True, default=Route.DOCUMENT)

    def __post_init__(self) -> None:
        if self.similarity not in {"dot", "cosine"}:
            raise ValueError(f"unsupported score similarity {self.similarity!r}")

    def loss(
        self,
        model: Encoder,
        batch: MarginDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        keys = _keys(key, len(batch.negatives) + 2)
        query = encode(model, batch.query, route=self.query_route, key=keys[0])
        positive = encode(
            model,
            batch.positive,
            route=self.document_route,
            key=keys[1],
        )
        negatives = jnp.stack(
            tuple(
                encode(
                    model,
                    payload,
                    route=self.document_route,
                    key=negative_key,
                )
                for payload, negative_key in zip(
                    batch.negatives,
                    keys[2:],
                    strict=True,
                )
            ),
            axis=0,
        )
        return self.loss_from_embeddings(query, positive, negatives, batch)

    def loss_from_embeddings(
        self,
        query: Float[Array, "batch representation"],
        positive: Float[Array, "batch representation"],
        negatives: Float[Array, "negative batch representation"],
        batch: MarginDistillationBatch,
    ) -> LossOutput:
        terms = margin_mse_loss_terms(
            query,
            positive,
            negatives,
            batch.teacher_margins,
            valid=batch.valid,
            similarity=self.similarity,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "margin_error_mean": jnp.sqrt(terms.loss),
                "valid_examples": jnp.sum(batch.valid),
            },
        )


class DistributionDistillationTask(eqx.Module):
    """Match teacher and student candidate-score distributions."""

    similarity: ScoreSimilarity = eqx.field(static=True, default="dot")
    temperature: float = eqx.field(static=True, default=1.0)
    query_route: Route = eqx.field(static=True, default=Route.QUERY)
    candidate_route: Route = eqx.field(static=True, default=Route.DOCUMENT)

    def __post_init__(self) -> None:
        if self.similarity not in {"dot", "cosine"}:
            raise ValueError(f"unsupported score similarity {self.similarity!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("distillation temperature must be finite and positive")

    def loss(
        self,
        model: Encoder,
        batch: DistributionDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        keys = _keys(key, len(batch.candidates) + 1)
        query = encode(model, batch.query, route=self.query_route, key=keys[0])
        candidates = jnp.stack(
            tuple(
                encode(
                    model,
                    payload,
                    route=self.candidate_route,
                    key=candidate_key,
                )
                for payload, candidate_key in zip(
                    batch.candidates,
                    keys[1:],
                    strict=True,
                )
            ),
            axis=0,
        )
        return self.loss_from_embeddings(query, candidates, batch)

    def loss_from_embeddings(
        self,
        query: Float[Array, "batch representation"],
        candidates: Float[Array, "candidate batch representation"],
        batch: DistributionDistillationBatch,
    ) -> LossOutput:
        terms = distribution_kl_loss_terms(
            query,
            candidates,
            batch.teacher_scores,
            valid=batch.valid,
            similarity=self.similarity,
            temperature=self.temperature,
        )
        teacher_entropy = -jnp.sum(
            terms.teacher_probabilities * jnp.log(terms.teacher_probabilities),
            axis=1,
        )
        valid_count = jnp.maximum(jnp.sum(batch.valid), 1)
        return LossOutput(
            loss=terms.loss,
            metrics={
                "teacher_entropy_mean": jnp.sum(
                    jnp.where(batch.valid, teacher_entropy, 0.0)
                )
                / valid_count.astype(jnp.float32),
                "valid_examples": jnp.sum(batch.valid),
            },
        )


__all__ = [
    "DistributionDistillationTask",
    "EmbeddingDistillationTask",
    "MarginDistillationTask",
]
