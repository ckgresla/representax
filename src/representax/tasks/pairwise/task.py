"""Runtime tasks for labeled-pair representation objectives."""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import PairwiseBatch
from .losses import (
    PairDistance,
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
)


def _encode_pairs(
    model: Encoder,
    batch: PairwiseBatch,
    *,
    left_route: Route,
    right_route: Route,
    key: PRNGKeyArray | None,
    encode_fn: EncodeFunction = encode,
) -> tuple[
    Float[Array, "pair representation"],
    Float[Array, "pair representation"],
]:
    if key is None:
        left_key = right_key = None
    else:
        left_key, right_key = jax.random.split(key)
    return (
        encode_fn(model, batch.left, route=left_route, key=left_key),
        encode_fn(model, batch.right, route=right_route, key=right_key),
    )


def _valid_mean(
    values: Float[Array, " pair"],
    valid: Array,
) -> Float[Array, ""]:
    count = jnp.sum(valid)
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(count, 1).astype(
        jnp.float32
    )


class CosineRegressionTask(eqx.Module):
    """Supervise the cosine similarity of aligned representation pairs."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "pair_similarity_mean": "mean",
        "valid_pairs": "sum",
    }

    left_route: Route = eqx.field(static=True, default=Route.GENERIC)
    right_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def accumulation_weight(self, batch: PairwiseBatch) -> Array:
        """Return the exact denominator used by the batch-mean objective."""

        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        return _encode_pairs(
            model,
            batch,
            left_route=self.left_route,
            right_route=self.right_route,
            key=key,
            encode_fn=encode_fn,
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, Array],
        batch: PairwiseBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(*representations, batch)

    def loss_from_embeddings(
        self,
        left: Float[Array, "pair representation"],
        right: Float[Array, "pair representation"],
        batch: PairwiseBatch,
    ) -> LossOutput:
        terms = cosine_regression_loss_terms(
            left,
            right,
            batch.labels,
            valid=batch.valid,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "pair_similarity_mean": _valid_mean(terms.scores, batch.valid),
                "valid_pairs": jnp.sum(batch.valid),
            },
        )


class ContrastiveTask(eqx.Module):
    """Contrastive distance objective with an explicit mining policy."""

    accumulation_metric_reductions: ClassVar[dict[str, str]] = {
        "pair_distance_mean": "mean",
        "selected_pairs": "sum",
        "valid_pairs": "sum",
    }

    distance: PairDistance = eqx.field(static=True, default="cosine")
    margin: float = eqx.field(static=True, default=0.5)
    online: bool = eqx.field(static=True, default=False)
    left_route: Route = eqx.field(static=True, default=Route.GENERIC)
    right_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if self.distance not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(f"unsupported contrastive distance {self.distance!r}")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("contrastive margin must be finite and positive")

    @property
    def supports_gradient_accumulation(self) -> bool:
        """Ordinary pair losses decompose; online mining uses the whole batch."""

        return not self.online

    def accumulation_weight(self, batch: PairwiseBatch) -> Array:
        return jnp.sum(batch.valid).astype(jnp.float32)

    def loss(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        return _encode_pairs(
            model,
            batch,
            left_route=self.left_route,
            right_route=self.right_route,
            key=key,
            encode_fn=encode_fn,
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, Array],
        batch: PairwiseBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(*representations, batch)

    def loss_from_embeddings(
        self,
        left: Float[Array, "pair representation"],
        right: Float[Array, "pair representation"],
        batch: PairwiseBatch,
    ) -> LossOutput:
        loss_function = (
            online_contrastive_loss_terms if self.online else contrastive_loss_terms
        )
        terms = loss_function(
            left,
            right,
            batch.labels,
            valid=batch.valid,
            metric=self.distance,
            margin=self.margin,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "pair_distance_mean": _valid_mean(terms.scores, batch.valid),
                "selected_pairs": jnp.sum(terms.selected),
                "valid_pairs": jnp.sum(batch.valid),
            },
        )


class CoSENTTask(eqx.Module):
    """Order cosine similarities by the labels of aligned pairs."""

    scale: float = eqx.field(static=True, default=20.0)
    left_route: Route = eqx.field(static=True, default=Route.GENERIC)
    right_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("CoSENT scale must be finite and positive")

    def loss(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        return _encode_pairs(
            model,
            batch,
            left_route=self.left_route,
            right_route=self.right_route,
            key=key,
            encode_fn=encode_fn,
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, Array],
        batch: PairwiseBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(*representations, batch)

    def loss_from_embeddings(
        self,
        left: Float[Array, "pair representation"],
        right: Float[Array, "pair representation"],
        batch: PairwiseBatch,
    ) -> LossOutput:
        terms = pair_ranking_loss_terms(
            left,
            right,
            batch.labels,
            valid=batch.valid,
            scale=self.scale,
            similarity="cosine",
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "ordered_comparisons": jnp.sum(terms.ordered_pairs),
                "valid_pairs": jnp.sum(batch.valid),
            },
        )


class AnglETask(eqx.Module):
    """Order AnglE complex-space similarities by aligned-pair labels."""

    scale: float = eqx.field(static=True, default=20.0)
    left_route: Route = eqx.field(static=True, default=Route.GENERIC)
    right_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("AnglE scale must be finite and positive")

    def loss(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        return _encode_pairs(
            model,
            batch,
            left_route=self.left_route,
            right_route=self.right_route,
            key=key,
            encode_fn=encode_fn,
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, Array],
        batch: PairwiseBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(*representations, batch)

    def loss_from_embeddings(
        self,
        left: Float[Array, "pair representation"],
        right: Float[Array, "pair representation"],
        batch: PairwiseBatch,
    ) -> LossOutput:
        terms = pair_ranking_loss_terms(
            left,
            right,
            batch.labels,
            valid=batch.valid,
            scale=self.scale,
            similarity="angle",
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "ordered_comparisons": jnp.sum(terms.ordered_pairs),
                "valid_pairs": jnp.sum(batch.valid),
            },
        )


__all__ = [
    "AnglETask",
    "CoSENTTask",
    "ContrastiveTask",
    "CosineRegressionTask",
]
