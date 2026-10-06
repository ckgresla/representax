"""Contrastive-tension pair and in-batch objectives."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float


class ContrastiveTensionTerms(eqx.Module):
    loss: Float[Array, ""]
    scores: Float[Array, "*batch"]
    row_losses: Float[Array, " batch"]


def contrastive_tension_loss_terms(
    first: Float[Array, "pair representation"],
    second: Float[Array, "pair representation"],
    labels: Float[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
) -> ContrastiveTensionTerms:
    """Summed binary cross entropy over aligned dual-encoder dot products."""

    if first.shape != second.shape or labels.shape != first.shape[:1]:
        raise ValueError("contrastive-tension pairs and labels must align")
    resolved_valid = (
        jnp.ones(labels.shape, dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    scores = jnp.sum(first * second, axis=-1)
    row_losses = optax.sigmoid_binary_cross_entropy(scores, labels)
    return ContrastiveTensionTerms(
        loss=jnp.sum(jnp.where(resolved_valid, row_losses, 0.0)),
        scores=scores,
        row_losses=row_losses,
    )


def _normalize(values: Array) -> Array:
    return values / jnp.maximum(
        jnp.linalg.norm(values, axis=-1, keepdims=True),
        jnp.asarray(1e-12, values.dtype),
    )


def contrastive_tension_in_batch_loss_terms(
    first: Float[Array, "batch representation"],
    second: Float[Array, "batch representation"],
    logit_scale: Float[Array, ""],
    *,
    similarity: Literal["cosine", "dot"] = "cosine",
    valid: Bool[Array, " batch"] | None = None,
) -> ContrastiveTensionTerms:
    """Symmetric in-batch classification with a trainable log scale."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("in-batch contrastive-tension embeddings must align")
    if similarity not in {"cosine", "dot"}:
        raise ValueError("contrastive-tension similarity must be cosine or dot")
    batch_size = first.shape[0]
    resolved_valid = (
        jnp.ones((batch_size,), dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    if resolved_valid.shape != (batch_size,):
        raise ValueError("contrastive-tension valid must match batch rows")
    if similarity == "cosine":
        first, second = _normalize(first), _normalize(second)
    scores = (first @ second.T) * jnp.exp(logit_scale)
    masked_scores = jnp.where(resolved_valid[None, :], scores, -jnp.inf)
    masked_scores = jnp.where(resolved_valid[:, None], masked_scores, 0.0)
    labels = jnp.arange(batch_size)
    forward = -jax.nn.log_softmax(masked_scores, axis=1)[labels, labels]
    reverse = -jax.nn.log_softmax(masked_scores.T, axis=1)[labels, labels]
    row_losses = (forward + reverse) / 2.0
    loss = jnp.sum(jnp.where(resolved_valid, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(resolved_valid), 1
    ).astype(jnp.float32)
    return ContrastiveTensionTerms(
        loss=loss,
        scores=scores,
        row_losses=row_losses,
    )
