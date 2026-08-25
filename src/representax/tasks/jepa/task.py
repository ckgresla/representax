"""LeJEPA as an ordinary stateless representation task."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from representax.core import EncodeFunction, Encoder, LossOutput, Route, encode

from .batch import JEPABatch
from .losses import invariance_loss, sigreg_loss


class LeJEPATask(eqx.Module):
    regularization_weight: float = eqx.field(static=True, default=0.02)
    knots: int = eqx.field(static=True, default=17)
    slices: int = eqx.field(static=True, default=256)
    max_frequency: float = eqx.field(static=True, default=3.0)
    route: Route = eqx.field(static=True, default=Route.GENERIC)

    def representations(
        self,
        model: Encoder,
        batch: JEPABatch,
        *,
        key: PRNGKeyArray | None = None,
        encode_fn: EncodeFunction = encode,
    ) -> Array:
        shape = batch.valid.shape

        def flatten(value):
            if not isinstance(value, jax.Array):
                return value
            return value.reshape((shape[0] * shape[1], *value.shape[2:]))

        values = encode_fn(
            model,
            jax.tree.map(flatten, batch.views),
            route=self.route,
            key=key,
        )
        return values.reshape((shape[0], shape[1], values.shape[-1]))

    def loss_from_representations(
        self,
        projections: Array,
        batch: JEPABatch,
        *,
        key: PRNGKeyArray,
    ) -> LossOutput:
        directions = jax.random.normal(
            key,
            (projections.shape[-1], self.slices),
            dtype=jnp.float32,
        )
        invariance = invariance_loss(projections, batch.valid)
        regularization = sigreg_loss(
            projections.astype(jnp.float32),
            batch.valid,
            directions,
            knots=self.knots,
            max_frequency=self.max_frequency,
        )
        weight = self.regularization_weight
        return LossOutput(
            loss=(1.0 - weight) * invariance + weight * regularization,
            metrics={
                "invariance": invariance,
                "sigreg": regularization,
                "valid_samples": jnp.sum(jnp.any(batch.valid, axis=-1)),
            },
        )

    def loss(
        self,
        model: Encoder,
        batch: JEPABatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        if key is None:
            raise ValueError("LeJEPA requires a PRNG key for SIGReg sketches")
        encode_key, loss_key = jax.random.split(key)
        projections = self.representations(model, batch, key=encode_key)
        return self.loss_from_representations(
            projections,
            batch,
            key=loss_key,
        )


__all__ = ["LeJEPATask"]
