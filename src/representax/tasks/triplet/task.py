"""Runtime tasks for explicit and within-batch-mined triplet objectives."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import ExplicitTripletBatch, LabeledExamplesBatch
from .losses import (
    BatchTripletDistance,
    ExplicitTripletDistance,
    batch_all_triplet_loss_terms,
    batch_hard_triplet_loss_terms,
    batch_semi_hard_triplet_loss_terms,
    explicit_triplet_loss_terms,
)

BatchTripletMining = Literal["all", "hard", "hard_soft_margin", "semi_hard"]


def _valid_mean(
    values: Float[Array, " row"],
    valid: Array,
) -> Float[Array, ""]:
    count = jnp.sum(valid)
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(count, 1).astype(
        jnp.float32
    )


class ExplicitTripletTask(eqx.Module):
    """Margin triplet supervision over supplied aligned rows."""

    distance: ExplicitTripletDistance = eqx.field(static=True, default="euclidean")
    margin: float = eqx.field(static=True, default=5.0)
    anchor_route: Route = eqx.field(static=True, default=Route.GENERIC)
    positive_route: Route = eqx.field(static=True, default=Route.GENERIC)
    negative_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if self.distance not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError(f"unsupported triplet distance {self.distance!r}")
        if not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("triplet margin must be finite and positive")

    def loss(
        self,
        model: Encoder,
        batch: ExplicitTripletBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: ExplicitTripletBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array, Array]:
        if key is None:
            anchor_key = positive_key = negative_key = None
        else:
            anchor_key, positive_key, negative_key = jax.random.split(key, 3)
        anchor = encode_fn(
            model,
            batch.anchor,
            route=self.anchor_route,
            key=anchor_key,
        )
        positive = encode_fn(
            model,
            batch.positive,
            route=self.positive_route,
            key=positive_key,
        )
        negative = encode_fn(
            model,
            batch.negative,
            route=self.negative_route,
            key=negative_key,
        )
        return anchor, positive, negative

    def loss_from_representations(
        self,
        representations: tuple[Array, Array, Array],
        batch: ExplicitTripletBatch,
    ) -> LossOutput:
        return self.loss_from_embeddings(*representations, batch)

    def loss_from_embeddings(
        self,
        anchor: Float[Array, "triplet representation"],
        positive: Float[Array, "triplet representation"],
        negative: Float[Array, "triplet representation"],
        batch: ExplicitTripletBatch,
    ) -> LossOutput:
        terms = explicit_triplet_loss_terms(
            anchor,
            positive,
            negative,
            valid=batch.valid,
            metric=self.distance,
            margin=self.margin,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "positive_distance_mean": _valid_mean(
                    terms.positive_distances, batch.valid
                ),
                "negative_distance_mean": _valid_mean(
                    terms.negative_distances, batch.valid
                ),
                "valid_triplets": jnp.sum(batch.valid),
            },
        )


class BatchTripletTask(eqx.Module):
    """Triplet supervision mined from one class-labeled representation batch."""

    mining: BatchTripletMining = eqx.field(static=True, default="hard")
    distance: BatchTripletDistance = eqx.field(static=True, default="euclidean")
    margin: float | None = eqx.field(static=True, default=5.0)
    route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if self.mining not in {"all", "hard", "hard_soft_margin", "semi_hard"}:
            raise ValueError(f"unsupported triplet mining policy {self.mining!r}")
        if self.distance not in {"cosine", "euclidean", "squared_euclidean"}:
            raise ValueError(f"unsupported batch triplet distance {self.distance!r}")
        if self.mining == "hard_soft_margin":
            if self.margin is not None:
                raise ValueError(
                    "hard soft-margin mining does not accept a fixed margin"
                )
        elif self.margin is None or not math.isfinite(self.margin) or self.margin <= 0:
            raise ValueError("triplet margin must be finite and positive")

    def loss(
        self,
        model: Encoder,
        batch: LabeledExamplesBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)

    def representations(
        self,
        model: Encoder,
        batch: LabeledExamplesBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array]:
        return (encode_fn(model, batch.examples, route=self.route, key=key),)

    def loss_from_representations(
        self,
        representations: tuple[Array],
        batch: LabeledExamplesBatch,
    ) -> LossOutput:
        (embeddings,) = representations
        return self.loss_from_embeddings(embeddings, batch)

    def loss_from_embeddings(
        self,
        embeddings: Float[Array, "example representation"],
        batch: LabeledExamplesBatch,
    ) -> LossOutput:
        if self.mining == "all":
            assert self.margin is not None
            terms = batch_all_triplet_loss_terms(
                embeddings,
                batch.labels,
                valid=batch.valid,
                metric=self.distance,
                margin=self.margin,
            )
        elif self.mining == "semi_hard":
            assert self.margin is not None
            terms = batch_semi_hard_triplet_loss_terms(
                embeddings,
                batch.labels,
                valid=batch.valid,
                metric=self.distance,
                margin=self.margin,
            )
        else:
            terms = batch_hard_triplet_loss_terms(
                embeddings,
                batch.labels,
                valid=batch.valid,
                metric=self.distance,
                margin=0.0 if self.margin is None else self.margin,
                soft_margin=self.mining == "hard_soft_margin",
            )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "selected_triplets": jnp.sum(terms.selected),
                "valid_examples": jnp.sum(batch.valid),
            },
        )


__all__ = ["BatchTripletMining", "BatchTripletTask", "ExplicitTripletTask"]
