"""Runtime global orthogonal representation regularization."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import RegularizationBatch
from .losses import global_orthogonal_regularization_terms


class GlobalOrthogonalRegularizationTask(eqx.Module):
    """Apply independently auditable GOR terms to each representation column."""

    similarity: Literal["cosine", "dot"] = eqx.field(static=True, default="cosine")
    mean_weight: float = eqx.field(static=True, default=1.0)
    second_moment_weight: float = eqx.field(static=True, default=1.0)
    aggregation: Literal["mean", "sum"] = eqx.field(static=True, default="mean")
    routes: tuple[Route, ...] = eqx.field(static=True, default=(Route.GENERIC,))

    def representations(
        self,
        model: Encoder,
        batch: RegularizationBatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> tuple[Array, ...]:
        routes = (
            self.routes * len(batch.inputs) if len(self.routes) == 1 else self.routes
        )
        if len(routes) != len(batch.inputs):
            raise ValueError("GOR routes must be shared or match input columns")
        keys = (
            (None,) * len(batch.inputs)
            if key is None
            else tuple(jax.random.split(key, len(batch.inputs)))
        )
        return tuple(
            encode_fn(model, payload, route=route, key=column_key)
            for payload, route, column_key in zip(
                batch.inputs,
                routes,
                keys,
                strict=True,
            )
        )

    def loss_from_representations(
        self,
        representations: tuple[Array, ...],
        batch: RegularizationBatch,
    ) -> LossOutput:
        terms = global_orthogonal_regularization_terms(
            representations,
            valid=batch.valid,
            similarity=self.similarity,
            mean_weight=self.mean_weight,
            second_moment_weight=self.second_moment_weight,
            aggregation=self.aggregation,
        )
        return LossOutput(
            loss=terms.loss,
            metrics={
                "gor_mean": terms.mean_term,
                "gor_second_moment": terms.second_moment_term,
                "valid_examples": jnp.sum(batch.valid),
            },
        )

    def loss(
        self,
        model: Encoder,
        batch: RegularizationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        representations = self.representations(model, batch, key=key)
        return self.loss_from_representations(representations, batch)
