"""Core model protocol tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from representax.core import (
    EncoderMetadata,
    Modality,
    Route,
    bind,
    encode,
)
from representax.models import DenseEncoder


def test_dense_encoder_obeys_route_aware_contract():
    model = DenseEncoder(4, 3, key=jax.random.key(0))
    inputs = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    queries = encode(model, inputs, route=Route.QUERY)
    documents = bind(model, route=Route.DOCUMENT)(inputs)

    assert queries.shape == (2, 3)
    assert jnp.allclose(jnp.linalg.norm(queries, axis=-1), 1.0)
    assert jnp.allclose(queries, documents)


def test_encoder_rejects_unsupported_route():
    class RestrictedEncoder(eqx.Module):
        metadata: EncoderMetadata

        def encode(self, inputs, *, route, key=None):
            del route, key
            return inputs

    model = RestrictedEncoder(
        EncoderMetadata(
            model_id="restricted",
            revision="1",
            output_dimension=3,
            routes=frozenset({Route.QUERY}),
            modalities=frozenset({Modality.TEXT}),
        )
    )

    with pytest.raises(ValueError, match="does not support route"):
        encode(model, jnp.ones((1, 4)), route=Route.DOCUMENT)


def test_modality_is_extensible_but_fusion_is_composition():
    depth = Modality("depth_map")

    assert str(depth) == "depth_map"
    assert depth.value == "depth_map"
    assert not hasattr(Modality, "FUSED")
    with pytest.raises(ValueError, match="lowercase identifiers"):
        Modality("Depth Map")
