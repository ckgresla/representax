"""Pair feature construction and softmax classification."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class SoftmaxClassificationTerms(eqx.Module):
    loss: Float[Array, ""]
    logits: Float[Array, "pair class"]
    row_losses: Float[Array, " pair"]


def pair_features(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    *,
    concatenate_representations: bool = True,
    concatenate_difference: bool = True,
    concatenate_product: bool = False,
) -> Float[Array, "pair feature"]:
    """Construct the released SoftmaxLoss pair feature vector."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("pair classification embeddings must be aligned matrices")
    features = []
    if concatenate_representations:
        features.extend((left, right))
    if concatenate_difference:
        features.append(jnp.abs(left - right))
    if concatenate_product:
        features.append(left * right)
    if not features:
        raise ValueError("pair classification requires at least one feature family")
    return jnp.concatenate(features, axis=-1)


def softmax_classification_loss_terms(
    logits: Float[Array, "pair class"],
    labels: Int[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
) -> SoftmaxClassificationTerms:
    """Mean multiclass cross entropy over valid aligned pairs."""

    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("classification logits and labels must align")
    resolved_valid = (
        jnp.ones(labels.shape, dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    if resolved_valid.shape != labels.shape:
        raise ValueError("classification valid must match labels")
    row_losses = -jax.nn.log_softmax(logits, axis=-1)[
        jnp.arange(logits.shape[0]), labels
    ]
    loss = jnp.sum(jnp.where(resolved_valid, row_losses, 0.0)) / jnp.maximum(
        jnp.sum(resolved_valid), 1
    ).astype(jnp.float32)
    return SoftmaxClassificationTerms(
        loss=loss,
        logits=logits,
        row_losses=row_losses,
    )
