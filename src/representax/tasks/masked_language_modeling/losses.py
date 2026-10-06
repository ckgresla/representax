"""Token-normalized cross entropy over sparse MLM targets."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class MaskedLanguageModelLossTerms(eqx.Module):
    loss: Float[Array, ""]
    accuracy: Float[Array, ""]
    token_losses: Float[Array, "batch target"]
    token_valid: Bool[Array, "batch target"]


def masked_language_model_loss_terms(
    logits: Float[Array, "batch target vocabulary"],
    labels: Int[Array, "batch target"],
    *,
    row_valid: Bool[Array, " batch"] | None = None,
) -> MaskedLanguageModelLossTerms:
    """Mean FP32 CE over labels other than -100, excluding invalid rows.

    Labels must be vocabulary indices or -100. No supervised tokens produces
    zero loss and zero gradients, including for an empty accumulation shard.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("MLM logits and labels must align on batch and target axes")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError("MLM labels must be integers")
    if logits.shape[-1] < 1:
        raise ValueError("MLM vocabulary must be nonempty")
    if row_valid is None:
        row_valid = jnp.ones(labels.shape[:1], dtype=jnp.bool_)
    if row_valid.shape != labels.shape[:1] or row_valid.dtype != jnp.bool_:
        raise TypeError("MLM row_valid must be a matching boolean vector")
    valid = (labels != -100) & row_valid[:, None]
    # Remove ignored logits before softmax, so ignored NaNs cannot leak into VJPs.
    values = jnp.where(valid[..., None], logits.astype(jnp.float32), 0.0)
    targets = jnp.where(valid, labels, 0)
    selected = jnp.take_along_axis(
        jax.nn.log_softmax(values, axis=-1), targets[..., None], axis=-1
    )[..., 0]
    token_losses = jnp.where(valid, -selected, 0.0)
    denominator = jnp.maximum(jnp.sum(valid), 1).astype(jnp.float32)
    return MaskedLanguageModelLossTerms(
        loss=jnp.sum(token_losses) / denominator,
        accuracy=jnp.sum(valid & (jnp.argmax(values, axis=-1) == targets))
        / denominator,
        token_losses=token_losses,
        token_valid=valid,
    )
