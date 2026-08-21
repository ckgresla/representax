"""Small native encoder used as an executable reference integration."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.core.sharding import (
    activation_out_sharding,
    constrain_activation,
    replicate,
)
from representax.precision import compute_parameter, linear_matmul


class DenseEncoder(eqx.Module):
    """A shared projection encoder for numeric features.

    This model keeps the initial repository end-to-end trainable while the
    first Hugging Face architecture integration is ported and parity-gated.
    """

    projection: eqx.nn.Linear
    metadata: EncoderMetadata
    normalize: bool = eqx.field(static=True)

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        key: PRNGKeyArray,
        normalize: bool = True,
    ) -> None:
        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("input and output dimensions must be positive")
        self.projection = eqx.nn.Linear(input_dimension, output_dimension, key=key)
        self.metadata = EncoderMetadata(
            model_id="representax/dense",
            revision="1",
            output_dimension=output_dimension,
            routes=frozenset(Route),
            modalities=frozenset(Modality),
        )
        self.normalize = normalize

    def encode(
        self,
        inputs: Float[Array, "batch input"],
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route, key
        values = jnp.asarray(inputs)
        if values.ndim != 2:
            raise ValueError("DenseEncoder inputs must have shape [batch, features]")
        output = linear_matmul(
            values,
            replicate(compute_parameter(self.projection.weight)).T,
            out_sharding=activation_out_sharding(2),
        )
        if self.projection.bias is not None:
            output = output + replicate(compute_parameter(self.projection.bias))
        output = constrain_activation(output.astype(jnp.float32))
        if not self.normalize:
            return output
        norm = jnp.linalg.norm(output, axis=-1, keepdims=True)
        return output / jnp.maximum(norm, jnp.asarray(1e-12, output.dtype))
