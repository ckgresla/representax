"""Multiple-negatives ranking objectives."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Encoder, LossOutput, Route, encode

from .batch import RetrievalBatch


class MNRLossTerms(eqx.Module):
    """Auditable intermediates for direct and Matryoshka MNR."""

    loss: Float[Array, ""]
    forward_loss: Float[Array, ""]
    reverse_loss: Float[Array, ""]
    row_losses: Float[Array, " query"]
    reverse_row_losses: Float[Array, " document"]
    cosine_similarity: Float[Array, "query document"]
    scaled_logits: Float[Array, "query document"]


class _MNRLossValues(eqx.Module):
    """Training-facing MNR values without diagnostic similarity matrices."""

    loss: Float[Array, ""]
    forward_loss: Float[Array, ""]
    reverse_loss: Float[Array, ""]
    row_losses: Float[Array, " query"]
    reverse_row_losses: Float[Array, " document"]


def _normalize(
    values: Float[Array, "batch representation"],
) -> Float[Array, "batch representation"]:
    values = values.astype(jnp.float32)
    norm = jnp.linalg.norm(values, ord=2, axis=1, keepdims=True)
    return values / jnp.maximum(norm, jnp.asarray(1e-12, dtype=values.dtype))


def _prepare_mnr_inputs(
    query_embeddings: Float[Array, "query representation"],
    document_embeddings: Float[Array, "document representation"],
    positive_mask: Bool[Array, "query document"],
    *,
    positive_weights: Float[Array, "query document"] | None,
    query_valid: Bool[Array, " query"] | None,
    document_valid: Bool[Array, " document"] | None,
    scale: float,
) -> tuple[
    Float[Array, "query representation"],
    Float[Array, "document representation"],
    Bool[Array, "query document"],
    Float[Array, "query document"],
    Bool[Array, " query"],
    Bool[Array, " document"],
]:
    """Validate and canonicalize inputs for every MNR execution schedule."""

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

    resolved_query_valid = (
        jnp.ones((queries.shape[0],), dtype=jnp.bool_)
        if query_valid is None
        else jnp.asarray(query_valid, dtype=jnp.bool_)
    )
    resolved_document_valid = (
        jnp.ones((documents.shape[0],), dtype=jnp.bool_)
        if document_valid is None
        else jnp.asarray(document_valid, dtype=jnp.bool_)
    )
    if resolved_query_valid.shape != (queries.shape[0],):
        raise ValueError("query_valid shape must match query rows")
    if resolved_document_valid.shape != (documents.shape[0],):
        raise ValueError("document_valid shape must match document rows")

    weights = (
        jnp.ones(mask.shape, dtype=jnp.float32)
        if positive_weights is None
        else jnp.asarray(positive_weights, dtype=jnp.float32)
    )
    if weights.shape != mask.shape:
        raise ValueError("positive_weights must match positive_mask")
    return (
        queries,
        documents,
        mask,
        weights,
        resolved_query_valid,
        resolved_document_valid,
    )


def _direction_row_terms(
    row_embeddings: Float[Array, "row representation"],
    candidate_embeddings: Float[Array, "candidate representation"],
    positive_mask: Bool[Array, "row candidate"],
    positive_weights: Float[Array, "row candidate"],
    row_valid: Bool[Array, " row"],
    candidate_valid: Bool[Array, " candidate"],
    *,
    scale: float,
) -> tuple[
    Float[Array, "row candidate"],
    Float[Array, "row candidate"],
    Float[Array, " row"],
    Bool[Array, " row"],
]:
    """Canonical MNR formula for one contiguous block of objective rows."""

    cosine = _normalize(row_embeddings) @ _normalize(candidate_embeddings).T
    raw_logits = cosine * jnp.asarray(scale, dtype=jnp.float32)
    logits = jnp.where(candidate_valid[None, :], raw_logits, -jnp.inf)
    active = positive_mask & row_valid[:, None] & candidate_valid[None, :]
    weights = jnp.where(active, positive_weights, 0.0)
    partition = jax.nn.logsumexp(logits, axis=1)
    row_weight = jnp.sum(weights, axis=1)
    row_losses = jnp.sum(
        jnp.where(active, weights * (partition[:, None] - raw_logits), 0.0),
        axis=1,
    ) / jnp.maximum(row_weight, 1.0)
    active_rows = row_valid & (row_weight > 0)
    return cosine, logits, row_losses, active_rows


def _tiled_direction_loss(
    row_embeddings: Float[Array, "row representation"],
    candidate_embeddings: Float[Array, "candidate representation"],
    positive_mask: Bool[Array, "row candidate"],
    positive_weights: Float[Array, "row candidate"],
    row_valid: Bool[Array, " row"],
    candidate_valid: Bool[Array, " candidate"],
    *,
    scale: float,
    row_chunk_size: int,
) -> tuple[Float[Array, ""], Float[Array, " row"]]:
    """Evaluate and differentiate MNR with only one score-row tile live."""

    row_count = row_embeddings.shape[0]
    chunk_count = (row_count + row_chunk_size - 1) // row_chunk_size
    padded_count = chunk_count * row_chunk_size
    padding = padded_count - row_count

    def pad_rows(value: Array, fill_value: int | float = 0) -> Array:
        widths = ((0, padding),) + ((0, 0),) * (value.ndim - 1)
        return jnp.pad(value, widths, constant_values=fill_value).reshape(
            chunk_count, row_chunk_size, *value.shape[1:]
        )

    row_chunks = pad_rows(row_embeddings)
    mask_chunks = pad_rows(positive_mask)
    weight_chunks = pad_rows(positive_weights)
    valid_chunks = pad_rows(row_valid)

    def body(
        totals: tuple[Array, Array],
        values: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[Array, Array], Array]:
        rows, mask, weights, valid = values
        _, _, row_losses, active_rows = _direction_row_terms(
            rows,
            candidate_embeddings,
            mask,
            weights,
            valid,
            candidate_valid,
            scale=scale,
        )
        loss_sum, active_count = totals
        return (
            loss_sum + jnp.sum(jnp.where(active_rows, row_losses, 0.0)),
            active_count + jnp.sum(active_rows),
        ), row_losses

    rematerialized_body = jax.checkpoint(
        body,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    # Derive zero carries from mapped inputs so shard_map's value-mapping
    # analysis preserves the enclosing axis type without coupling MNR to a
    # particular mesh-axis name.
    initial = (
        jnp.zeros_like(jnp.sum(row_embeddings, dtype=jnp.float32)),
        jnp.zeros_like(jnp.sum(row_valid, dtype=jnp.int32)),
    )
    (loss_sum, active_count), loss_chunks = jax.lax.scan(
        rematerialized_body,
        initial,
        (row_chunks, mask_chunks, weight_chunks, valid_chunks),
    )
    loss = loss_sum / jnp.maximum(active_count, 1).astype(jnp.float32)
    return loss, loss_chunks.reshape(-1)[:row_count]


def _mnr_loss_values(
    query_embeddings: Float[Array, "query representation"],
    document_embeddings: Float[Array, "document representation"],
    positive_mask: Bool[Array, "query document"],
    *,
    positive_weights: Float[Array, "query document"] | None = None,
    query_valid: Bool[Array, " query"] | None = None,
    document_valid: Bool[Array, " document"] | None = None,
    scale: float = 20.0,
    symmetric: bool = False,
    row_chunk_size: int | None = None,
) -> _MNRLossValues:
    """Compute canonical MNR values, optionally with bounded score-row tiles."""

    if row_chunk_size is None:
        terms = mnr_loss_terms(
            query_embeddings,
            document_embeddings,
            positive_mask,
            positive_weights=positive_weights,
            query_valid=query_valid,
            document_valid=document_valid,
            scale=scale,
            symmetric=symmetric,
        )
        return _MNRLossValues(
            loss=terms.loss,
            forward_loss=terms.forward_loss,
            reverse_loss=terms.reverse_loss,
            row_losses=terms.row_losses,
            reverse_row_losses=terms.reverse_row_losses,
        )
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive when set")

    queries, documents, mask, weights, query_valid, document_valid = (
        _prepare_mnr_inputs(
            query_embeddings,
            document_embeddings,
            positive_mask,
            positive_weights=positive_weights,
            query_valid=query_valid,
            document_valid=document_valid,
            scale=scale,
        )
    )

    forward_loss, row_losses = _tiled_direction_loss(
        queries,
        documents,
        mask,
        weights,
        query_valid,
        document_valid,
        scale=scale,
        row_chunk_size=row_chunk_size,
    )
    if symmetric:
        reverse_loss, reverse_row_losses = _tiled_direction_loss(
            documents,
            queries,
            mask.T,
            weights.T,
            document_valid,
            query_valid,
            scale=scale,
            row_chunk_size=row_chunk_size,
        )
        loss = (forward_loss + reverse_loss) / 2.0
    else:
        reverse_loss = jnp.asarray(0.0, dtype=jnp.float32)
        reverse_row_losses = jnp.zeros((documents.shape[0],), dtype=jnp.float32)
        loss = forward_loss
    return _MNRLossValues(
        loss=loss,
        forward_loss=forward_loss,
        reverse_loss=reverse_loss,
        row_losses=row_losses,
        reverse_row_losses=reverse_row_losses,
    )


def mnr_loss_terms(
    query_embeddings: Float[Array, "query representation"],
    document_embeddings: Float[Array, "document representation"],
    positive_mask: Bool[Array, "query document"],
    *,
    positive_weights: Float[Array, "query document"] | None = None,
    query_valid: Bool[Array, " query"] | None = None,
    document_valid: Bool[Array, " document"] | None = None,
    scale: float = 20.0,
    symmetric: bool = False,
) -> MNRLossTerms:
    """Compute query-balanced cosine MNR over a dense positive relation."""

    queries, documents, mask, weights, query_valid, document_valid = (
        _prepare_mnr_inputs(
            query_embeddings,
            document_embeddings,
            positive_mask,
            positive_weights=positive_weights,
            query_valid=query_valid,
            document_valid=document_valid,
            scale=scale,
        )
    )
    cosine, logits, row_losses, active_queries = _direction_row_terms(
        queries,
        documents,
        mask,
        weights,
        query_valid,
        document_valid,
        scale=scale,
    )
    forward_loss = jnp.sum(jnp.where(active_queries, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(active_queries), 1
    ).astype(jnp.float32)

    if symmetric:
        _, _, reverse_row_losses, active_documents = _direction_row_terms(
            documents,
            queries,
            mask.T,
            weights.T,
            document_valid,
            query_valid,
            scale=scale,
        )
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
    negative_scope: Literal["local", "global"] = eqx.field(
        static=True,
        default="global",
    )

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
        if self.negative_scope not in {"local", "global"}:
            raise ValueError("negative_scope must be 'local' or 'global'")

    def loss(
        self,
        model: Encoder,
        batch: RetrievalBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if key is None:
            query_key = document_key = None
        else:
            query_key, document_key = jax.random.split(key)
        queries = encode(model, batch.query, route=Route.QUERY, key=query_key)
        documents = encode(
            model, batch.document, route=Route.DOCUMENT, key=document_key
        )
        return self.loss_from_embeddings(queries, documents, batch)

    def loss_from_embeddings(
        self,
        query_embeddings: Float[Array, "query representation"],
        document_embeddings: Float[Array, "document representation"],
        batch: RetrievalBatch,
        *,
        row_chunk_size: int | None = None,
    ) -> LossOutput:
        """Evaluate the same MNR objective from already-computed representations."""

        queries = jnp.asarray(query_embeddings)
        documents = jnp.asarray(document_embeddings)
        dimensions = self.dimensions or (queries.shape[1],)
        if dimensions[-1] > queries.shape[1]:
            raise ValueError("Matryoshka dimension exceeds encoder output dimension")
        raw_weights = self.dimension_weights or tuple(1.0 for _ in dimensions)
        weights = jnp.asarray(raw_weights, dtype=jnp.float32)
        weights = weights / jnp.sum(weights)
        terms = tuple(
            _mnr_loss_values(
                queries[:, :dimension],
                documents[:, :dimension],
                batch.positive_mask,
                positive_weights=batch.positive_weights,
                query_valid=batch.query_valid,
                document_valid=batch.document_valid,
                scale=self.scale,
                symmetric=self.symmetric,
                row_chunk_size=row_chunk_size,
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
