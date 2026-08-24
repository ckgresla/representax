"""ColBERT-style late-interaction contrastive training."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import (
    EncodeFunction,
    LateInteractionEncoder,
    LateInteractionRepresentation,
    LossOutput,
    Route,
    encode_late_interaction,
)
from representax.tasks.retrieval import RetrievalBatch

from .scoring import maxsim_scores


class LateInteractionLossTerms(eqx.Module):
    """Auditable MaxSim contrastive-loss intermediates."""

    loss: Float[Array, ""]
    forward_loss: Float[Array, ""]
    reverse_loss: Float[Array, ""]
    scores: Float[Array, "query document"]
    scaled_logits: Float[Array, "query document"]


def _direction_loss(
    logits: Float[Array, "row candidate"],
    positive_mask: Bool[Array, "row candidate"],
    positive_weights: Float[Array, "row candidate"],
    row_valid: Bool[Array, " row"],
    candidate_valid: Bool[Array, " candidate"],
) -> Float[Array, ""]:
    raw_logits = logits
    logits = jnp.where(candidate_valid[None, :], raw_logits, -jnp.inf)
    active = positive_mask & row_valid[:, None] & candidate_valid[None, :]
    weights = jnp.where(active, positive_weights, 0.0)
    row_weight = jnp.sum(weights, axis=1)
    partition = jax.nn.logsumexp(logits, axis=1)
    row_losses = jnp.sum(
        jnp.where(active, weights * (partition[:, None] - raw_logits), 0.0),
        axis=1,
    ) / jnp.maximum(row_weight, 1.0)
    active_rows = row_valid & (row_weight > 0)
    return jnp.sum(jnp.where(active_rows, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(active_rows), 1
    ).astype(jnp.float32)


def late_interaction_loss_terms(
    queries: LateInteractionRepresentation,
    documents: LateInteractionRepresentation,
    positive_mask: Bool[Array, "query document"],
    *,
    positive_weights: Float[Array, "query document"] | None = None,
    query_valid: Bool[Array, " query"] | None = None,
    document_valid: Bool[Array, " document"] | None = None,
    temperature: float = 0.02,
    symmetric: bool = False,
    document_chunk_size: int | None = None,
) -> LateInteractionLossTerms:
    """Compute exact weighted in-batch contrastive loss over MaxSim scores."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    query_count = queries.values.shape[0]
    document_count = documents.values.shape[0]
    mask = jnp.asarray(positive_mask)
    if mask.shape != (query_count, document_count) or mask.dtype != jnp.bool_:
        raise TypeError("positive_mask must be a matching boolean matrix")
    weights = (
        jnp.ones(mask.shape, dtype=jnp.float32)
        if positive_weights is None
        else jnp.asarray(positive_weights, dtype=jnp.float32)
    )
    if weights.shape != mask.shape:
        raise ValueError("positive_weights must match positive_mask")
    query_valid = (
        jnp.ones((query_count,), dtype=jnp.bool_)
        if query_valid is None
        else jnp.asarray(query_valid, dtype=jnp.bool_)
    )
    document_valid = (
        jnp.ones((document_count,), dtype=jnp.bool_)
        if document_valid is None
        else jnp.asarray(document_valid, dtype=jnp.bool_)
    )
    if query_valid.shape != (query_count,):
        raise ValueError("query_valid must match query rows")
    if document_valid.shape != (document_count,):
        raise ValueError("document_valid must match document rows")

    scores = maxsim_scores(
        queries,
        documents,
        document_chunk_size=document_chunk_size,
    )
    logits = scores / jnp.asarray(temperature, dtype=jnp.float32)
    forward_loss = _direction_loss(
        logits,
        mask,
        weights,
        query_valid,
        document_valid,
    )
    if symmetric:
        reverse_loss = _direction_loss(
            logits.T,
            mask.T,
            weights.T,
            document_valid,
            query_valid,
        )
        loss = (forward_loss + reverse_loss) / 2.0
    else:
        reverse_loss = jnp.asarray(0.0, dtype=jnp.float32)
        loss = forward_loss
    return LateInteractionLossTerms(
        loss=loss,
        forward_loss=forward_loss,
        reverse_loss=reverse_loss,
        scores=scores,
        scaled_logits=logits,
    )


class LateInteractionTask(eqx.Module):
    """Direct or exact-cached ColBERT-style retrieval training."""

    temperature: float = eqx.field(static=True, default=0.02)
    symmetric: bool = eqx.field(static=True, default=False)
    negative_scope: Literal["local", "global"] = eqx.field(
        static=True,
        default="global",
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.negative_scope not in {"local", "global"}:
            raise ValueError("negative_scope must be 'local' or 'global'")

    def loss(
        self,
        model: LateInteractionEncoder,
        batch: RetrievalBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(
            model,
            batch,
            key=key,
            encode_fn=encode_late_interaction,
        )
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: LateInteractionEncoder,
        batch: RetrievalBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode_late_interaction,
    ) -> tuple[LateInteractionRepresentation, LateInteractionRepresentation]:
        if key is None:
            query_key = document_key = None
        else:
            query_key, document_key = jax.random.split(key)
        return (
            encode_fn(model, batch.query, route=Route.QUERY, key=query_key),
            encode_fn(model, batch.document, route=Route.DOCUMENT, key=document_key),
        )

    def loss_from_representations(
        self,
        representations: tuple[
            LateInteractionRepresentation,
            LateInteractionRepresentation,
        ],
        batch: RetrievalBatch,
        *,
        row_chunk_size: int | None = None,
    ) -> LossOutput:
        queries, documents = representations
        terms = late_interaction_loss_terms(
            queries,
            documents,
            batch.positive_mask,
            positive_weights=batch.positive_weights,
            query_valid=batch.query_valid,
            document_valid=batch.document_valid,
            temperature=self.temperature,
            symmetric=self.symmetric,
            document_chunk_size=row_chunk_size,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "forward_loss": terms.forward_loss,
                "reverse_loss": terms.reverse_loss,
            },
        )


__all__ = [
    "LateInteractionLossTerms",
    "LateInteractionTask",
    "late_interaction_loss_terms",
]
