"""Native pairwise, listwise, pointwise, and process reward tasks."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import LossOutput, Scorer, score_logits

from .batch import (
    ListwiseRewardBatch,
    PairwiseRewardBatch,
    PointwiseRewardBatch,
    ProcessRewardBatch,
    concatenate_pairwise_payloads,
)
from .config import RewardObjective
from .losses import bradley_terry_loss, plackett_luce_loss, pointwise_reward_loss


def _scalar_rewards(logits: Array) -> Array:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    raise ValueError("reward models must emit one scalar per scored input")


def _flatten_candidates(inputs, shape: tuple[int, int]):
    def flatten(value):
        if not isinstance(value, jax.Array):
            return value
        if value.shape[:2] != shape:
            raise ValueError("reward candidate leaves must align with preferences")
        return value.reshape((shape[0] * shape[1], *value.shape[2:]))

    return jax.tree.map(flatten, inputs)


class PairwiseRewardTask(eqx.Module):
    """Bradley--Terry outcome reward modeling."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "accuracy": "mean",
        "reward_margin": "mean",
        "valid_preferences": "sum",
    }

    center_rewards_coefficient: float | None = eqx.field(static=True, default=None)

    def accumulation_weight(self, batch: PairwiseRewardBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: PairwiseRewardBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        rows = batch.margins.shape[0]
        rewards = _scalar_rewards(
            score_logits(
                model,
                concatenate_pairwise_payloads(batch.chosen, batch.rejected),
                key=key,
            )
        )
        if rewards.shape[0] != rows * 2:
            raise ValueError("pairwise reward scorer must emit two rows per pair")
        chosen, rejected = rewards[:rows], rewards[rows:]
        margin = chosen - rejected
        count = jnp.maximum(jnp.sum(batch.valid), 1).astype(jnp.float32)
        return LossOutput(
            loss=bradley_terry_loss(
                chosen,
                rejected,
                batch.margins,
                batch.valid,
                center_rewards_coefficient=self.center_rewards_coefficient,
            ),
            metrics={
                "accuracy": jnp.sum(jnp.where(batch.valid, chosen > rejected, False))
                / count,
                "reward_margin": jnp.sum(jnp.where(batch.valid, margin, 0.0)) / count,
                "valid_preferences": jnp.sum(batch.valid),
            },
        )


class ListwiseRewardTask(eqx.Module):
    """Plackett--Luce reward modeling over finite candidate lists."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "top_choice_accuracy": "mean",
        "valid_prompts": "sum",
    }

    def accumulation_weight(self, batch: ListwiseRewardBatch) -> Array:
        return jnp.sum(jnp.any(batch.valid, axis=-1)).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: ListwiseRewardBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        shape = (batch.preferences.shape[0], batch.preferences.shape[1])
        rewards = _scalar_rewards(
            score_logits(
                model,
                _flatten_candidates(batch.candidates, shape),
                key=key,
            )
        ).reshape(shape)
        masked_rewards = jnp.where(batch.valid, rewards, -jnp.inf)
        masked_preferences = jnp.where(batch.valid, batch.preferences, -jnp.inf)
        row_valid = jnp.any(batch.valid, axis=-1)
        count = jnp.maximum(jnp.sum(row_valid), 1).astype(jnp.float32)
        return LossOutput(
            loss=plackett_luce_loss(rewards, batch.preferences, batch.valid),
            metrics={
                "top_choice_accuracy": jnp.sum(
                    jnp.where(
                        row_valid,
                        jnp.argmax(masked_rewards, axis=-1)
                        == jnp.argmax(masked_preferences, axis=-1),
                        False,
                    )
                )
                / count,
                "valid_prompts": jnp.sum(row_valid),
            },
        )


class PointwiseRewardTask(eqx.Module):
    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "reward_loss": "mean",
        "valid_rewards": "sum",
    }

    objective: RewardObjective = eqx.field(static=True, default="mse")

    def accumulation_weight(self, batch: PointwiseRewardBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: PointwiseRewardBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        rewards = _scalar_rewards(score_logits(model, batch.inputs, key=key))
        loss = pointwise_reward_loss(
            rewards, batch.labels, batch.valid, objective=self.objective
        )
        return LossOutput(
            loss=loss,
            metrics={"reward_loss": loss, "valid_rewards": jnp.sum(batch.valid)},
        )


class ProcessRewardTask(eqx.Module):
    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "step_accuracy": "mean",
        "valid_steps": "sum",
    }

    objective: RewardObjective = eqx.field(static=True, default="binary_cross_entropy")

    def accumulation_weight(self, batch: ProcessRewardBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Scorer,
        batch: ProcessRewardBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        logits = score_logits(model, batch.inputs, key=key)
        if logits.shape != batch.labels.shape:
            raise ValueError("process reward logits must have [batch, step] shape")
        loss = pointwise_reward_loss(
            logits, batch.labels, batch.valid, objective=self.objective
        )
        count = jnp.maximum(jnp.sum(batch.valid), 1).astype(jnp.float32)
        accuracy = (
            jnp.sum(
                jnp.where(batch.valid, (logits >= 0) == (batch.labels >= 0.5), False)
            )
            / count
        )
        return LossOutput(
            loss=loss,
            metrics={"step_accuracy": accuracy, "valid_steps": jnp.sum(batch.valid)},
        )


__all__ = [
    "ListwiseRewardTask",
    "PairwiseRewardTask",
    "PointwiseRewardTask",
    "ProcessRewardTask",
]
