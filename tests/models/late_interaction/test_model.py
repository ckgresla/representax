"""Fast contracts for the native late-interaction model composition."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route, encode_late_interaction
from representax.models import LateInteractionTextEncoder
from representax.models.components import Linear


class _Batch(eqx.Module):
    input_ids: jax.Array
    attention_mask: jax.Array


class _Backbone(eqx.Module):
    table: jax.Array

    def hidden_states(self, inputs: _Batch) -> jax.Array:
        return self.table[inputs.input_ids]


def test_text_encoder_projects_normalizes_and_applies_route_masks():
    model = LateInteractionTextEncoder(
        backbone=_Backbone(jnp.arange(24, dtype=jnp.float32).reshape(6, 4)),
        projection=Linear(weight=jnp.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])),
        metadata=EncoderMetadata(
            model_id="representax/test-late-interaction-text",
            revision="1",
            output_dimension=2,
            routes=frozenset({Route.QUERY, Route.DOCUMENT}),
            modalities=frozenset({Modality.TEXT}),
        ),
        skip_token_ids=(2,),
        query_expansion=True,
    )
    batch = _Batch(
        input_ids=jnp.asarray([[1, 2, 0]]),
        attention_mask=jnp.asarray([[1, 1, 0]]),
    )

    query = jax.jit(
        lambda value: encode_late_interaction(model, value, route=Route.QUERY)
    )(batch)
    document = jax.jit(
        lambda value: encode_late_interaction(model, value, route=Route.DOCUMENT)
    )(batch)

    np.testing.assert_array_equal(query.valid, [[True, True, True]])
    np.testing.assert_array_equal(document.valid, [[True, False, False]])
    np.testing.assert_allclose(
        jnp.linalg.norm(query.values, axis=-1),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(document.values[:, 1:], 0.0)
