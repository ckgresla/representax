"""Guide-filtered retrieval task."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import GISTBatch
from .losses import gist_loss_terms


class GISTTask(eqx.Module):
    """Filter likely false negatives using precomputed guide representations."""

    temperature: float = eqx.field(static=True, default=0.01)
    margin_strategy: Literal["absolute", "relative"] = eqx.field(
        static=True,
        default="absolute",
    )
    margin: float = eqx.field(static=True, default=0.0)
    contrast_anchors: bool = eqx.field(static=True, default=True)
    contrast_positives: bool = eqx.field(static=True, default=True)

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("GIST temperature must be finite and positive")
        if self.margin_strategy not in {"absolute", "relative"}:
            raise ValueError("GIST margin_strategy must be 'absolute' or 'relative'")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("GIST margin must be finite and non-negative")

    def representations(
        self,
        model: Encoder,
        batch: GISTBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, ...]:
        keys = (
            (None,) * (len(batch.negatives) + 2)
            if key is None
            else tuple(jax.random.split(key, len(batch.negatives) + 2))
        )
        anchor = encode_fn(model, batch.anchor, route=Route.QUERY, key=keys[0])
        positive = encode_fn(model, batch.positive, route=Route.DOCUMENT, key=keys[1])
        negatives = tuple(
            encode_fn(model, value, route=Route.DOCUMENT, key=column_key)
            for value, column_key in zip(batch.negatives, keys[2:], strict=True)
        )
        return anchor, positive, *negatives

    def loss_from_representations(
        self,
        representations: tuple[Array, ...],
        batch: GISTBatch,
        *,
        row_chunk_size: int | None = None,
    ) -> LossOutput:
        terms = gist_loss_terms(
            representations,
            (batch.guide_anchor, batch.guide_positive, *batch.guide_negatives),
            valid=batch.valid,
            temperature=self.temperature,
            margin_strategy=self.margin_strategy,
            margin=self.margin,
            contrast_anchors=self.contrast_anchors,
            contrast_positives=self.contrast_positives,
            row_chunk_size=row_chunk_size,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "masked_candidates": terms.masked_candidates,
                "valid_examples": jnp.sum(batch.valid),
            },
        )

    def loss(
        self,
        model: Encoder,
        batch: GISTBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)
