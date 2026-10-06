"""Mega-batch hardest-negative margin objective."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class MegaBatchMarginTerms(eqx.Module):
    loss: Float[Array, ""]
    row_losses: Float[Array, " batch"]
    positive_scores: Float[Array, " batch"]
    negative_scores: Float[Array, " batch"]
    negative_indices: Int[Array, " batch"]


def _normalize(values: Array) -> Array:
    values = values.astype(jnp.float32)
    return values / jnp.maximum(
        jnp.linalg.norm(values, axis=-1, keepdims=True),
        jnp.asarray(1e-12, values.dtype),
    )


def mega_batch_margin_loss_terms(
    anchor: Float[Array, "batch representation"],
    positive: Float[Array, "batch representation"],
    *,
    valid: Bool[Array, " batch"] | None = None,
    positive_margin: float = 0.8,
    negative_margin: float = 0.3,
) -> MegaBatchMarginTerms:
    """Mine each row's hardest non-paired positive and apply two margins."""

    if anchor.shape != positive.shape or anchor.ndim != 2:
        raise ValueError("mega-batch anchor and positive embeddings must align")
    batch_size = anchor.shape[0]
    resolved_valid = (
        jnp.ones((batch_size,), dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    scores = _normalize(anchor) @ _normalize(positive).T
    candidate_mask = resolved_valid[None, :] & ~jnp.eye(batch_size, dtype=jnp.bool_)
    mining_scores = jnp.where(candidate_mask, scores, -jnp.inf)
    negative_indices = jnp.argmax(mining_scores, axis=1)
    negative_scores = mining_scores[jnp.arange(batch_size), negative_indices]
    positive_scores = jnp.diag(scores)
    row_losses = jax.nn.relu(positive_margin - positive_scores) + jax.nn.relu(
        negative_scores - negative_margin
    )
    loss = jnp.sum(jnp.where(resolved_valid, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(resolved_valid), 1
    ).astype(jnp.float32)
    return MegaBatchMarginTerms(
        loss=loss,
        row_losses=row_losses,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        negative_indices=negative_indices,
    )


def selected_mega_batch_margin_loss_terms(
    anchor: Float[Array, "batch representation"],
    positive: Float[Array, "batch representation"],
    negative: Float[Array, "batch representation"],
    *,
    valid: Bool[Array, " batch"],
    positive_margin: float,
    negative_margin: float,
) -> MegaBatchMarginTerms:
    """Evaluate margins after an execution schedule has selected negatives."""

    normalized_anchor = _normalize(anchor)
    positive_scores = jnp.sum(normalized_anchor * _normalize(positive), axis=-1)
    negative_scores = jnp.sum(normalized_anchor * _normalize(negative), axis=-1)
    row_losses = jax.nn.relu(positive_margin - positive_scores) + jax.nn.relu(
        negative_scores - negative_margin
    )
    loss = jnp.sum(jnp.where(valid, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(valid), 1
    ).astype(jnp.float32)
    return MegaBatchMarginTerms(
        loss=loss,
        row_losses=row_losses,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        negative_indices=jnp.zeros((anchor.shape[0],), dtype=jnp.int32),
    )
