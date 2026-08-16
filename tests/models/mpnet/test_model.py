"""Native MPNet configuration, model, and checkpoint contracts."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.mpnet import (
    MPNetBatch,
    MPNetCheckpointAdapter,
    MPNetConfig,
    MPNetEncoder,
    create_mpnet_position_ids,
    mpnet_relative_position_bucket,
    mpnet_weight_names,
)


def tiny_config() -> MPNetConfig:
    return MPNetConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        max_position_embeddings=16,
        relative_attention_num_buckets=32,
        hidden_dropout_probability=0.2,
        attention_dropout_probability=0.1,
    )


def tiny_batch() -> MPNetBatch:
    return MPNetBatch(
        input_ids=jnp.asarray([[0, 4, 5, 2, 1], [0, 6, 2, 1, 1]]),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
    )


def test_config_maps_transformers_values_and_rejects_unsupported_buckets():
    config = MPNetConfig.from_hf_config(
        {
            "vocab_size": 101,
            "hidden_size": 24,
            "intermediate_size": 48,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "max_position_embeddings": 66,
            "relative_attention_num_buckets": 32,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.2,
            "attention_probs_dropout_prob": 0.3,
            "layer_norm_eps": 1e-6,
            "initializer_range": 0.01,
            "pad_token_id": 1,
            "bos_token_id": 0,
            "eos_token_id": 2,
        }
    )
    assert config.head_dimension == 6
    assert config.hidden_activation == "silu"
    assert config.hidden_dropout_probability == 0.2
    assert config.to_hf_config()["model_type"] == "mpnet"

    values = config.model_dump()
    values["relative_attention_num_buckets"] = 16
    with pytest.raises(ValueError, match="requires relative_attention_num_buckets=32"):
        MPNetConfig(**values)


def test_position_ids_and_relative_buckets_match_pinned_transformers():
    input_ids = jnp.asarray([[0, 4, 1, 5, 1]], dtype=jnp.int32)
    np.testing.assert_array_equal(
        create_mpnet_position_ids(input_ids),
        [[2, 3, 1, 4, 1]],
    )
    relative = jnp.asarray(
        [[-129, -128, -17, -8, -7, -1, 0, 1, 7, 8, 17, 128, 129]],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(
        mpnet_relative_position_bucket(relative),
        [[15, 15, 10, 8, 7, 1, 0, 17, 23, 24, 26, 31, 31]],
    )


def test_native_mpnet_is_scanned_jittable_and_dropout_is_keyed():
    model = MPNetEncoder.init(tiny_config(), key=jax.random.key(0))
    batch = tiny_batch()

    @eqx.filter_jit
    def inference(candidate, inputs):
        hidden = candidate.hidden_states(inputs)
        pooled = candidate.pooler_output(inputs)
        representation = candidate.encode(inputs, route=Route.QUERY)
        return hidden, pooled, representation

    hidden, pooled, representation = inference(model, batch)
    assert hidden.shape == (2, 5, 12)
    assert pooled.shape == (2, 12)
    np.testing.assert_allclose(
        jnp.linalg.norm(representation, axis=-1),
        jnp.ones((2,)),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(hidden, model.hidden_states(batch))

    first = model.hidden_states(batch, key=jax.random.key(1))
    same = model.hidden_states(batch, key=jax.random.key(1))
    second = model.hidden_states(batch, key=jax.random.key(2))
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, second)

    @eqx.filter_jit
    def hidden_only(candidate, inputs):
        return candidate.hidden_states(inputs)

    lowered = cast(Any, hidden_only).lower(model, batch).as_text()
    assert lowered.count("stablehlo.while") == 1


def test_checkpoint_mapping_round_trips_the_depth_major_tree():
    config = tiny_config()
    model = MPNetEncoder.init(config, key=jax.random.key(3))
    adapter = MPNetCheckpointAdapter(rematerialization="none")
    state = adapter.state_dict(model)
    assert set(state) == mpnet_weight_names(config)

    restored = adapter.from_state_dict(config, state)
    expected = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    actual = jax.tree.leaves(eqx.filter(restored, eqx.is_array))
    assert len(actual) == len(expected)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)

    broken = dict(state)
    broken["encoder.relative_attention_bias.weight"] = broken[
        "encoder.relative_attention_bias.weight"
    ][:-1]
    with pytest.raises(ValueError, match="expected"):
        adapter.from_state_dict(config, broken)


def test_native_mpnet_accepts_input_embeddings_and_differentiates_them():
    config = tiny_config()
    model = MPNetEncoder.init(config, key=jax.random.key(4))
    embeddings = jax.random.normal(jax.random.key(5), (2, 5, config.hidden_size))
    attention_mask = tiny_batch().attention_mask

    def objective(values):
        batch = MPNetBatch(inputs_embeds=values, attention_mask=attention_mask)
        return jnp.sum(model.encode(batch, route=Route.GENERIC))

    gradient = jax.grad(objective)(embeddings)
    assert gradient.shape == embeddings.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
