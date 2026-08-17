"""Pair classification using an encoder-plus-head model tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import EncodeFunction, LossOutput, Route, encode

if TYPE_CHECKING:
    from representax.models import PairClassifier

from .batch import PairClassificationBatch
from .losses import pair_features, softmax_classification_loss_terms


class SoftmaxClassificationTask(eqx.Module):
    """Classify aligned representation pairs through explicit trainable state."""

    concatenate_representations: bool = eqx.field(static=True, default=True)
    concatenate_difference: bool = eqx.field(static=True, default=True)
    concatenate_product: bool = eqx.field(static=True, default=False)
    left_route: Route = eqx.field(static=True, default=Route.GENERIC)
    right_route: Route = eqx.field(static=True, default=Route.GENERIC)

    def __post_init__(self) -> None:
        if not any(
            (
                self.concatenate_representations,
                self.concatenate_difference,
                self.concatenate_product,
            )
        ):
            raise ValueError("softmax classification requires at least one feature")

    def representations(
        self,
        model: PairClassifier,
        batch: PairClassificationBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, Array]:
        if key is None:
            left_key = right_key = None
        else:
            left_key, right_key = jax.random.split(key)
        return (
            encode_fn(model.encoder, batch.left, route=self.left_route, key=left_key),
            encode_fn(
                model.encoder,
                batch.right,
                route=self.right_route,
                key=right_key,
            ),
        )

    def loss_from_representations(
        self,
        model: PairClassifier,
        representations: tuple[Array, Array],
        batch: PairClassificationBatch,
    ) -> LossOutput:
        features = pair_features(
            *representations,
            concatenate_representations=self.concatenate_representations,
            concatenate_difference=self.concatenate_difference,
            concatenate_product=self.concatenate_product,
        )
        terms = softmax_classification_loss_terms(
            model.classify(features),
            batch.labels,
            valid=batch.valid,
        )
        predictions = jnp.argmax(terms.logits, axis=-1)
        accuracy = jnp.sum(
            jnp.where(batch.valid, predictions == batch.labels, False)
        ) / jnp.maximum(jnp.sum(batch.valid), 1).astype(jnp.float32)
        return LossOutput(
            loss=terms.loss,
            metrics={"accuracy": accuracy, "valid_pairs": jnp.sum(batch.valid)},
        )

    def loss(
        self,
        model: PairClassifier,
        batch: PairClassificationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(model, representations, batch)
