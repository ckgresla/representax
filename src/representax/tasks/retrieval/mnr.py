"""Multiple-negatives ranking objectives."""

from __future__ import annotations

import math
from numbers import Real

import equinox as eqx
import jax
import jax.numpy as jnp

from representax.core import Encoder, LossOutput, Route, encode

from .batch import RetrievalBatch


class MNRLossTerms(eqx.Module):
    """Auditable intermediates for direct and Matryoshka MNR."""

    loss: jax.Array
    forward_loss: jax.Array
    reverse_loss: jax.Array
    row_losses: jax.Array
    reverse_row_losses: jax.Array
    cosine_similarity: jax.Array
    scaled_logits: jax.Array


def _normalize(values: jax.Array) -> jax.Array:
    values = values.astype(jnp.float32)
    norm = jnp.linalg.norm(values, ord=2, axis=1, keepdims=True)
    return values / jnp.maximum(norm, jnp.asarray(1e-12, dtype=values.dtype))


def mnr_loss_terms(
    query_embeddings: jax.Array,
    document_embeddings: jax.Array,
    positive_mask: jax.Array,
    *,
    positive_weights: jax.Array | None = None,
    query_valid: jax.Array | None = None,
    document_valid: jax.Array | None = None,
    scale: float = 20.0,
    symmetric: bool = False,
) -> MNRLossTerms:
    """Compute query-balanced cosine MNR over a dense positive relation."""

    queries = jnp.asarray(query_embeddings)
    documents = jnp.asarray(document_embeddings)
    mask = jnp.asarray(positive_mask)
    if queries.ndim != 2 or documents.ndim != 2:
        raise ValueError("query and document embeddings must be matrices")
    if queries.shape[1] != documents.shape[1]:
        raise ValueError("query and document embedding dimensions must match")
    if mask.shape != (queries.shape[0], documents.shape[0]):
        raise ValueError("positive_mask shape must match the embedding rows")
    if mask.dtype != jnp.bool_:
        raise TypeError("positive_mask must be boolean")
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("scale must be a real number")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    query_valid = (
        jnp.ones((queries.shape[0],), dtype=jnp.bool_)
        if query_valid is None
        else jnp.asarray(query_valid, dtype=jnp.bool_)
    )
    document_valid = (
        jnp.ones((documents.shape[0],), dtype=jnp.bool_)
        if document_valid is None
        else jnp.asarray(document_valid, dtype=jnp.bool_)
    )
    if query_valid.shape != (queries.shape[0],):
        raise ValueError("query_valid shape must match query rows")
    if document_valid.shape != (documents.shape[0],):
        raise ValueError("document_valid shape must match document rows")

    cosine = _normalize(queries) @ _normalize(documents).T
    raw_logits = cosine * jnp.asarray(scale, dtype=jnp.float32)
    logits = jnp.where(document_valid[None, :], raw_logits, -jnp.inf)
    active = mask & query_valid[:, None] & document_valid[None, :]
    weights = (
        jnp.ones_like(raw_logits)
        if positive_weights is None
        else jnp.asarray(positive_weights, dtype=jnp.float32)
    )
    if weights.shape != mask.shape:
        raise ValueError("positive_weights must match positive_mask")
    weights = jnp.where(active, weights, 0.0)

    forward_partition = jax.nn.logsumexp(logits, axis=1)
    row_weight = jnp.sum(weights, axis=1)
    row_losses = jnp.sum(
        jnp.where(active, weights * (forward_partition[:, None] - raw_logits), 0.0),
        axis=1,
    ) / jnp.maximum(row_weight, 1.0)
    active_queries = query_valid & (row_weight > 0)
    forward_loss = jnp.sum(jnp.where(active_queries, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(active_queries), 1
    ).astype(jnp.float32)

    if symmetric:
        reverse_partition = jax.nn.logsumexp(
            jnp.where(query_valid[:, None], raw_logits, -jnp.inf), axis=0
        )
        document_weight = jnp.sum(weights, axis=0)
        reverse_row_losses = jnp.sum(
            jnp.where(
                active,
                weights * (reverse_partition[None, :] - raw_logits),
                0.0,
            ),
            axis=0,
        ) / jnp.maximum(document_weight, 1.0)
        active_documents = document_valid & (document_weight > 0)
        reverse_loss = jnp.sum(
            jnp.where(active_documents, reverse_row_losses, 0.0)
        ) / jnp.maximum(jnp.sum(active_documents), 1).astype(jnp.float32)
        loss = (forward_loss + reverse_loss) / 2.0
    else:
        reverse_row_losses = jnp.zeros((documents.shape[0],), dtype=jnp.float32)
        reverse_loss = jnp.asarray(0.0, dtype=jnp.float32)
        loss = forward_loss

    return MNRLossTerms(
        loss=loss,
        forward_loss=forward_loss,
        reverse_loss=reverse_loss,
        row_losses=row_losses,
        reverse_row_losses=reverse_row_losses,
        cosine_similarity=cosine,
        scaled_logits=logits,
    )


class MNRTask(eqx.Module):
    """Retrieval task supporting direct, symmetric, and Matryoshka MNR."""

    scale: float = eqx.field(static=True, default=20.0)
    symmetric: bool = eqx.field(static=True, default=False)
    dimensions: tuple[int, ...] | None = eqx.field(static=True, default=None)
    dimension_weights: tuple[float, ...] | None = eqx.field(static=True, default=None)

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if self.dimensions is not None:
            if not self.dimensions or any(d <= 0 for d in self.dimensions):
                raise ValueError("Matryoshka dimensions must be positive")
            if tuple(sorted(set(self.dimensions))) != self.dimensions:
                raise ValueError("Matryoshka dimensions must be sorted and unique")
            if self.dimension_weights is not None:
                if len(self.dimension_weights) != len(self.dimensions):
                    raise ValueError("dimension weights must match dimensions")
                if any(not math.isfinite(w) or w <= 0 for w in self.dimension_weights):
                    raise ValueError("dimension weights must be finite and positive")
        elif self.dimension_weights is not None:
            raise ValueError("dimension weights require Matryoshka dimensions")

    def loss(
        self,
        model: Encoder,
        batch: RetrievalBatch,
        *,
        key: jax.Array | None = None,
    ) -> LossOutput:
        if key is None:
            query_key = document_key = None
        else:
            query_key, document_key = jax.random.split(key)
        queries = encode(model, batch.query, route=Route.QUERY, key=query_key)
        documents = encode(
            model, batch.document, route=Route.DOCUMENT, key=document_key
        )
        dimensions = self.dimensions or (queries.shape[1],)
        if dimensions[-1] > queries.shape[1]:
            raise ValueError("Matryoshka dimension exceeds encoder output dimension")
        raw_weights = self.dimension_weights or tuple(1.0 for _ in dimensions)
        weights = jnp.asarray(raw_weights, dtype=jnp.float32)
        weights = weights / jnp.sum(weights)
        terms = tuple(
            mnr_loss_terms(
                queries[:, :dimension],
                documents[:, :dimension],
                batch.positive_mask,
                positive_weights=batch.positive_weights,
                query_valid=batch.query_valid,
                document_valid=batch.document_valid,
                scale=self.scale,
                symmetric=self.symmetric,
            )
            for dimension in dimensions
        )
        dimension_losses = jnp.stack([term.loss for term in terms])
        forward_losses = jnp.stack([term.forward_loss for term in terms])
        reverse_losses = jnp.stack([term.reverse_loss for term in terms])
        return LossOutput(
            loss=jnp.sum(weights * dimension_losses),
            metrics={
                "forward_loss": jnp.sum(weights * forward_losses),
                "reverse_loss": jnp.sum(weights * reverse_losses),
                "dimension_losses": dimension_losses,
            },
        )
