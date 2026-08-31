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
from representax.models.sentence import (
    SentenceEncoder,
    SentenceNormalize,
    SentencePooling,
)
from representax.tasks.retrieval import MNRTask, retrieval_batch


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


def test_native_mpnet_is_statically_unrolled_jittable_and_dropout_is_keyed():
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
    np.testing.assert_allclose(
        hidden,
        model.hidden_states(batch),
        rtol=2e-6,
        atol=3e-7,
    )
    layerwise = model.encode_layers(batch, route=Route.QUERY)
    assert layerwise.shape == (tiny_config().num_hidden_layers + 1, 2, 12)
    np.testing.assert_allclose(
        layerwise[-1],
        representation,
        rtol=1e-6,
        atol=1e-6,
    )

    first = model.hidden_states(batch, key=jax.random.key(1))
    same = model.hidden_states(batch, key=jax.random.key(1))
    second = model.hidden_states(batch, key=jax.random.key(2))
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, second)

    @eqx.filter_jit
    def hidden_only(candidate, inputs):
        return candidate.hidden_states(inputs)

    lowered = cast(Any, hidden_only).lower(model, batch).as_text()
    assert "stablehlo.while" not in lowered


def test_checkpoint_mapping_round_trips_the_depth_major_tree():
    config = tiny_config()
    model = MPNetEncoder.init(config, key=jax.random.key(3))
    adapter = MPNetCheckpointAdapter(rematerialization="none")
    first_layer = model.tower.layers.layer(0)
    assert first_layer.attention.query.weight_layout == "input_output"
    state = adapter.state_dict(model)
    assert set(state) == mpnet_weight_names(config)
    np.testing.assert_array_equal(
        state["encoder.layer.0.attention.attn.q.weight"],
        first_layer.attention.query.output_major().weight,
    )

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


def test_packed_mpnet_sentence_embeddings_match_independent_rows():
    backbone = MPNetEncoder.init(tiny_config(), key=jax.random.key(6))
    model = SentenceEncoder(
        backbone=backbone,
        pooling=SentencePooling(input_dimension=12, modes=("mean",)),
        postprocessors=(SentenceNormalize(),),
        metadata=backbone.metadata,
    )
    packed = MPNetBatch(
        input_ids=jnp.asarray([[0, 4, 5, 2, 0, 6, 2, 1]]),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 1, 1, 1, 0]]),
        position_ids=jnp.asarray([[2, 3, 4, 5, 2, 3, 4, 1]]),
        segment_ids=jnp.asarray([[0, 0, 0, 0, 1, 1, 1, -1]]),
        logical_batch_size=2,
    )

    expected = model.encode(tiny_batch(), route=Route.QUERY)
    actual = model.encode(packed, route=Route.QUERY)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=3e-7)

    changed = eqx.tree_at(
        lambda batch: batch.input_ids,
        packed,
        jnp.asarray([[0, 4, 5, 2, 0, 7, 2, 1]]),
    )
    changed_output = model.encode(changed, route=Route.QUERY)
    np.testing.assert_allclose(changed_output[0], actual[0], rtol=2e-6, atol=3e-7)
    assert not np.allclose(changed_output[1], actual[1])


def test_packed_mpnet_preserves_mnr_loss_and_parameter_gradients():
    values = tiny_config().model_dump()
    values.update(
        hidden_dropout_probability=0.0,
        attention_dropout_probability=0.0,
    )
    backbone = MPNetEncoder.init(MPNetConfig(**values), key=jax.random.key(7))
    model = SentenceEncoder(
        backbone=backbone,
        pooling=SentencePooling(input_dimension=12, modes=("mean",)),
        postprocessors=(SentenceNormalize(),),
        metadata=backbone.metadata,
    )
    packed = MPNetBatch(
        input_ids=jnp.asarray([[0, 4, 5, 2, 0, 6, 2, 1]]),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 1, 1, 1, 0]]),
        position_ids=jnp.asarray([[2, 3, 4, 5, 2, 3, 4, 1]]),
        segment_ids=jnp.asarray([[0, 0, 0, 0, 1, 1, 1, -1]]),
        logical_batch_size=2,
    )
    positives = jnp.eye(2, dtype=jnp.bool_)
    independent_batch = retrieval_batch(
        query=tiny_batch(),
        document=tiny_batch(),
        positive_mask=positives,
    )
    packed_batch = retrieval_batch(
        query=packed,
        document=packed,
        positive_mask=positives,
    )
    task = MNRTask()

    @eqx.filter_value_and_grad
    def objective(candidate, batch):
        return task.loss(candidate, batch).loss

    expected_loss, expected_gradient = objective(model, independent_batch)
    actual_loss, actual_gradient = objective(model, packed_batch)
    np.testing.assert_allclose(actual_loss, expected_loss, rtol=2e-6, atol=2e-7)
    expected_leaves = jax.tree.leaves(
        eqx.filter(expected_gradient, eqx.is_inexact_array)
    )
    actual_leaves = jax.tree.leaves(eqx.filter(actual_gradient, eqx.is_inexact_array))
    for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
