"""Causal token reconstruction loss."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class DenoisingLossTerms(eqx.Module):
    loss: Float[Array, ""]
    token_losses: Float[Array, "batch target"]
    token_valid: Bool[Array, "batch target"]


def denoising_autoencoder_loss_terms(
    logits: Float[Array, "batch target vocabulary"],
    target_ids: Int[Array, "batch target"],
    *,
    pad_token_id: int,
    row_valid: Bool[Array, " batch"] | None = None,
) -> DenoisingLossTerms:
    """Mean next-token cross entropy, ignoring decoder padding tokens."""

    if logits.ndim != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("denoising logits and shifted targets must align")
    valid_rows = (
        jnp.ones(target_ids.shape[:1], dtype=jnp.bool_)
        if row_valid is None
        else jnp.asarray(row_valid, dtype=jnp.bool_)
    )
    token_valid = (target_ids != pad_token_id) & valid_rows[:, None]
    token_losses = -jax.nn.log_softmax(logits, axis=-1)[
        jnp.arange(logits.shape[0])[:, None],
        jnp.arange(logits.shape[1])[None, :],
        target_ids,
    ]
    loss = jnp.sum(jnp.where(token_valid, token_losses, 0.0)) / jnp.maximum(
        jnp.sum(token_valid), 1
    ).astype(jnp.float32)
    return DenoisingLossTerms(
        loss=loss,
        token_losses=token_losses,
        token_valid=token_valid,
    )
