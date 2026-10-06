"""Runtime tasks for offline teacher-target distillation."""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

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

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "distilled_columns": "mean",
        "valid_examples": "sum",
    }

    distance: EmbeddingDistillationDistance = eqx.field(static=True, default="cosine")
    routes: tuple[Route, ...] = eqx.field(static=True, default=(Route.GENERIC,))

    def __post_init__(self) -> None:
        if self.distance not in {"mse", "l2", "cosine"}:
            raise ValueError(
                f"unsupported embedding distillation distance {self.distance!r}"
            )
        if not self.routes:
            raise ValueError("embedding distillation requires at least one route")

    def accumulation_weight(self, batch: EmbeddingDistillationBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def accumulation_batch_size(self, batch: EmbeddingDistillationBatch) -> int:
        return batch.valid.shape[0]

    def accumulation_microbatch(
        self,
        batch: EmbeddingDistillationBatch,
        index: Array,
        steps: int,
    ) -> EmbeddingDistillationBatch:
        size = batch.valid.shape[0] // steps
        start = index * size

        def rows(value: Array) -> Array:
            return jax.lax.dynamic_slice_in_dim(value, start, size, axis=0)

        return EmbeddingDistillationBatch(
            inputs=tuple(jax.tree.map(rows, payload) for payload in batch.inputs),
            teacher_embeddings=jax.lax.dynamic_slice_in_dim(
                batch.teacher_embeddings,
                start,
                size,
                axis=1,
            ),
            valid=rows(batch.valid),
        )

    def loss(
        self,
        model: Encoder,
        batch: EmbeddingDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: EmbeddingDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, ...]:
        routes = _resolved_routes(self.routes, len(batch.inputs))
        keys = _keys(key, len(batch.inputs))
        return tuple(
            encode_fn(model, payload, route=route, key=column_key)
            for payload, route, column_key in zip(
                batch.inputs,
                routes,
                keys,
                strict=True,
            )
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, ...],
        batch: EmbeddingDistillationBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(jnp.stack(representations), batch)

    def dimension_batch(
        self,
        batch: EmbeddingDistillationBatch,
        *,
        dimension: int,
        full_dimension: int,
    ) -> EmbeddingDistillationBatch:
        """Truncate embedding targets exactly when they match the full student width."""

        teacher = batch.teacher_embeddings
        if teacher.shape[-1] == full_dimension:
            teacher = teacher[..., :dimension]
        return EmbeddingDistillationBatch(
            inputs=batch.inputs,
            teacher_embeddings=teacher,
            valid=batch.valid,
        )

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

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "margin_error_mean": "root_mean_square",
        "valid_examples": "sum",
    }

    similarity: ScoreSimilarity = eqx.field(static=True, default="dot")
    query_route: Route = eqx.field(static=True, default=Route.QUERY)
    document_route: Route = eqx.field(static=True, default=Route.DOCUMENT)

    def __post_init__(self) -> None:
        if self.similarity not in {"dot", "cosine"}:
            raise ValueError(f"unsupported score similarity {self.similarity!r}")

    def accumulation_weight(self, batch: MarginDistillationBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Encoder,
        batch: MarginDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: MarginDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, ...]:
        keys = _keys(key, len(batch.negatives) + 2)
        query = encode_fn(model, batch.query, route=self.query_route, key=keys[0])
        positive = encode_fn(
            model,
            batch.positive,
            route=self.document_route,
            key=keys[1],
        )
        negatives = tuple(
            encode_fn(
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
        )
        return query, positive, *negatives

    def loss_from_representations(
        self,
        representations: tuple[Array, ...],
        batch: MarginDistillationBatch,
    ) -> LossOutput:
        query, positive, *negatives = representations
        return self.loss_from_embeddings(
            query,
            positive,
            jnp.stack(tuple(negatives)),
            batch,
        )

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

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "teacher_entropy_mean": "mean",
        "valid_examples": "sum",
    }

    similarity: ScoreSimilarity = eqx.field(static=True, default="dot")
    temperature: float = eqx.field(static=True, default=1.0)
    query_route: Route = eqx.field(static=True, default=Route.QUERY)
    candidate_route: Route = eqx.field(static=True, default=Route.DOCUMENT)

    def __post_init__(self) -> None:
        if self.similarity not in {"dot", "cosine"}:
            raise ValueError(f"unsupported score similarity {self.similarity!r}")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("distillation temperature must be finite and positive")

    def accumulation_weight(self, batch: DistributionDistillationBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Encoder,
        batch: DistributionDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: DistributionDistillationBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, ...]:
        keys = _keys(key, len(batch.candidates) + 1)
        query = encode_fn(model, batch.query, route=self.query_route, key=keys[0])
        candidates = tuple(
            encode_fn(
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
        )
        return query, *candidates

    def loss_from_representations(
        self,
        representations: tuple[Array, ...],
        batch: DistributionDistillationBatch,
    ) -> LossOutput:
        query, *candidates = representations
        return self.loss_from_embeddings(query, jnp.stack(tuple(candidates)), batch)

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
