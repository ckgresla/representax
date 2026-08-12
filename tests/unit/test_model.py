import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import representax as rx


def test_dense_encoder_obeys_route_aware_contract():
    model = rx.models.DenseEncoder(4, 3, key=jax.random.key(0))
    inputs = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

    queries = rx.encode(model, inputs, route=rx.Route.QUERY)
    documents = rx.bind(model, route=rx.Route.DOCUMENT)(inputs)

    assert queries.shape == (2, 3)
    assert jnp.allclose(jnp.linalg.norm(queries, axis=-1), 1.0)
    assert jnp.allclose(queries, documents)


def test_encoder_rejects_unsupported_route():
    class RestrictedEncoder(eqx.Module):
        metadata: rx.EncoderMetadata

        def encode(self, inputs, *, route, key=None):
            del route, key
            return inputs

    model = RestrictedEncoder(
        rx.EncoderMetadata(
            model_id="restricted",
            revision="1",
            output_dimension=3,
            routes=frozenset({rx.Route.QUERY}),
            modalities=frozenset({rx.Modality.TEXT}),
        )
    )

    with pytest.raises(ValueError, match="does not support route"):
        rx.encode(model, jnp.ones((1, 4)), route=rx.Route.DOCUMENT)
