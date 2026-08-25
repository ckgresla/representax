"""Pure-JAX reward-modeling objectives."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

RewardObjective = Literal["binary_cross_entropy", "mse"]


def _weighted_mean(values: Array, valid: Bool[Array, ...]) -> Float[Array, ""]:
    weights = valid.astype(jnp.float32)
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(jnp.sum(weights), 1.0)


def bradley_terry_loss(
    chosen: Float[Array, " batch"],
    rejected: Float[Array, " batch"],
    margins: Float[Array, " batch"],
    valid: Bool[Array, " batch"],
    *,
    center_rewards_coefficient: float | None = None,
) -> Float[Array, ""]:
    """TRL-compatible preference loss with optional reward centering."""

    if not (chosen.shape == rejected.shape == margins.shape == valid.shape):
        raise ValueError("pairwise reward tensors must align")
    loss = _weighted_mean(jax.nn.softplus(-(chosen - rejected - margins)), valid)
    if center_rewards_coefficient is not None:
        loss = loss + center_rewards_coefficient * _weighted_mean(
            jnp.square(chosen + rejected), valid
        )
    return loss


def plackett_luce_loss(
    rewards: Float[Array, "batch candidate"],
    preferences: Float[Array, "batch candidate"],
    valid: Bool[Array, "batch candidate"],
) -> Float[Array, ""]:
    """Negative log likelihood of each preference-ordered candidate list."""

    if rewards.shape != preferences.shape or valid.shape != rewards.shape:
        raise ValueError("listwise reward tensors must align")
    order = jnp.argsort(jnp.where(valid, preferences, -jnp.inf), axis=-1)[:, ::-1]
    ordered_rewards = jnp.take_along_axis(rewards, order, axis=-1)
    ordered_valid = jnp.take_along_axis(valid, order, axis=-1)
    masked = jnp.where(ordered_valid, ordered_rewards, -jnp.inf)
    suffix = jnp.flip(
        jax.lax.associative_scan(jnp.logaddexp, jnp.flip(masked, axis=-1), axis=-1),
        axis=-1,
    )
    terms = jnp.where(ordered_valid, suffix - ordered_rewards, 0.0)
    row_loss = jnp.sum(terms, axis=-1)
    return _weighted_mean(row_loss, jnp.any(valid, axis=-1))


def pointwise_reward_loss(
    logits: Float[Array, ...],
    labels: Float[Array, ...],
    valid: Bool[Array, ...],
    *,
    objective: RewardObjective,
) -> Float[Array, ""]:
    if logits.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("pointwise reward tensors must align")
    if objective == "binary_cross_entropy":
        values = (1.0 - labels) * jax.nn.softplus(logits)
        values = values + labels * jax.nn.softplus(-logits)
    elif objective == "mse":
        values = jnp.square(logits - labels)
    else:  # pragma: no cover - closed config literal
        raise ValueError(f"unsupported reward objective {objective!r}")
    return _weighted_mean(values, valid)


__all__ = ["bradley_terry_loss", "plackett_luce_loss", "pointwise_reward_loss"]
