"""Runtime mega-batch margin task."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import MegaBatch
from .losses import (
    mega_batch_margin_loss_terms,
    selected_mega_batch_margin_loss_terms,
)


class MegaBatchMarginTask(eqx.Module):
    """Mine hard positives-as-negatives and enforce cosine margins."""

    positive_margin: float = eqx.field(static=True, default=0.8)
    negative_margin: float = eqx.field(static=True, default=0.3)
    anchor_route: Route = eqx.field(static=True, default=Route.GENERIC)
    positive_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if not math.isfinite(self.positive_margin) or not math.isfinite(
            self.negative_margin
        ):
            raise ValueError("mega-batch margins must be finite")

    def representations(
        self,
        model: Encoder,
        batch: MegaBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        if key is None:
            anchor_key = positive_key = None
        else:
            anchor_key, positive_key = jax.random.split(key)
        return (
            encode_fn(model, batch.anchor, route=self.anchor_route, key=anchor_key),
            encode_fn(
                model,
                batch.positive,
                route=self.positive_route,
                key=positive_key,
            ),
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, Array],
        batch: MegaBatch,
    ) -> LossOutput:
        terms = mega_batch_margin_loss_terms(
            *representations,
            valid=batch.valid,
            positive_margin=self.positive_margin,
            negative_margin=self.negative_margin,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "positive_score_mean": jnp.mean(terms.positive_scores),
                "negative_score_mean": jnp.mean(terms.negative_scores),
                "valid_examples": jnp.sum(batch.valid),
            },
        )

    def loss_from_selected_embeddings(
        self,
        anchor: Array,
        positive: Array,
        negative: Array,
        batch: MegaBatch,
    ) -> LossOutput:
        terms = selected_mega_batch_margin_loss_terms(
            anchor,
            positive,
            negative,
            valid=batch.valid,
            positive_margin=self.positive_margin,
            negative_margin=self.negative_margin,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "positive_score_mean": jnp.mean(terms.positive_scores),
                "negative_score_mean": jnp.mean(terms.negative_scores),
                "valid_examples": jnp.sum(batch.valid),
            },
        )

    def loss(
        self,
        model: Encoder,
        batch: MegaBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)
