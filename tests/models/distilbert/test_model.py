"""Fast native contracts for DistilBERT."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.distilbert import (
    DistilBertBatch,
    DistilBertCheckpointAdapter,
    DistilBertConfig,
    DistilBertEncoder,
    distilbert_weight_names,
)


def _config() -> DistilBertConfig:
    return DistilBertConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        max_position_embeddings=16,
        hidden_dropout_probability=0.1,
        attention_dropout_probability=0.1,
        norm_epsilon=1e-12,
        pad_token_id=0,
    )


def test_config_maps_the_distilbert_schema_bidirectionally() -> None:
    config = _config()
    restored = DistilBertConfig.from_hf_config(config.to_hf_config())
    assert restored == config


def test_forward_gradients_and_checkpoint_roundtrip() -> None:
    config = _config()
    adapter = DistilBertCheckpointAdapter(rematerialization="none")
    model = DistilBertEncoder.init(
        config,
        key=jax.random.key(157),
        rematerialization="none",
    )
    batch = DistilBertBatch(
        input_ids=jnp.asarray(((1, 2, 3, 0), (4, 5, 0, 0))),
        attention_mask=jnp.asarray(((1, 1, 1, 0), (1, 1, 0, 0))),
    )
    hidden = model.hidden_states(batch)
    encoded = model.encode(batch, route=Route.GENERIC)
    assert hidden.shape == (2, 4, 12)
    assert encoded.shape == (2, 12)
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=1), 1.0, atol=1e-6)

    gradients = jax.grad(lambda candidate: jnp.sum(candidate.hidden_states(batch)))(
        model
    )
    gradient_leaves = [
        value for value in jax.tree.leaves(gradients) if value is not None
    ]
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradient_leaves)

    state = adapter.state_dict(model)
    assert frozenset(state) == distilbert_weight_names(config)
    restored = adapter.from_state_dict(config, state)
    for name, expected in state.items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], expected)


def test_checkpoint_tokenizer_segments_must_be_zero() -> None:
    batch = DistilBertEncoder.make_batch(
        input_ids=jnp.asarray(((1, 2),)),
        attention_mask=jnp.ones((1, 2), dtype=bool),
        token_type_ids=jnp.zeros((1, 2), dtype=jnp.int32),
    )
    assert isinstance(batch, DistilBertBatch)
