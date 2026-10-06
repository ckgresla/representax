"""Runtime contrastive-tension tasks over explicit dual-encoder state."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import LossOutput, Route, encode

if TYPE_CHECKING:
    from representax.models import EncoderPair

from .batch import ContrastiveTensionBatch, ContrastiveTensionExamples
from .losses import (
    contrastive_tension_in_batch_loss_terms,
    contrastive_tension_loss_terms,
)


class ContrastiveTensionTask(eqx.Module):
    """Binary supervision for independently trainable encoder branches."""

    accumulation_loss_reduction: ClassVar[str] = "sum"
    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "valid_pairs": "sum",
    }

    def accumulation_weight(self, batch: ContrastiveTensionBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: EncoderPair,
        batch: ContrastiveTensionBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if key is None:
            first_key = second_key = None
        else:
            first_key, second_key = jax.random.split(key)
        first = encode(model.first, batch.first, route=Route.GENERIC, key=first_key)
        second = encode(
            model.second,
            batch.second,
            route=Route.GENERIC,
            key=second_key,
        )
        terms = contrastive_tension_loss_terms(
            first,
            second,
            batch.labels,
            valid=batch.valid,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={"valid_pairs": jnp.sum(batch.valid)},
        )


class ContrastiveTensionInBatchTask(eqx.Module):
    """Symmetric in-batch contrastive tension with trainable temperature."""

    similarity: Literal["cosine", "dot"] = eqx.field(static=True, default="cosine")

    def __post_init__(self) -> None:
        if self.similarity not in {"cosine", "dot"}:
            raise ValueError("contrastive-tension similarity must be cosine or dot")

    def loss(
        self,
        model: EncoderPair,
        batch: ContrastiveTensionExamples,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if model.logit_scale is None:
            raise ValueError("in-batch contrastive tension requires model logit_scale")
        if key is None:
            first_key = second_key = None
        else:
            first_key, second_key = jax.random.split(key)
        first = encode(
            model.first,
            batch.examples,
            route=Route.GENERIC,
            key=first_key,
        )
        second = encode(
            model.second,
            batch.examples,
            route=Route.GENERIC,
            key=second_key,
        )
        terms = contrastive_tension_in_batch_loss_terms(
            first,
            second,
            model.logit_scale,
            similarity=self.similarity,
            valid=batch.valid,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "scale": jnp.exp(model.logit_scale),
                "valid_examples": jnp.sum(batch.valid),
            },
        )
