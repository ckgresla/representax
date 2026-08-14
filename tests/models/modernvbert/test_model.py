"""Native ModernVBERT model and checkpoint tests."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.modernvbert import (
    ModernVBERTBatch,
    ModernVBERTCheckpointAdapter,
    ModernVBERTConfig,
    ModernVBERTEncoder,
    ModernVBERTTextBatch,
    ModernVBERTTextCheckpointAdapter,
    ModernVBERTTextConfig,
    ModernVBERTTextEncoder,
    ModernVBERTVisionConfig,
    merge_image_features,
    modernvbert_text_weight_map,
    modernvbert_vision_weight_map,
    pixel_shuffle,
)
from representax.models.modernvbert.model import _scaled_dot_product_attention


def tiny_config() -> ModernVBERTTextConfig:
    return ModernVBERTTextConfig(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        layer_types=("full_attention", "sliding_attention"),
        local_attention=4,
        full_attention_rope_theta=10_000.0,
        sliding_attention_rope_theta=1_000.0,
        norm_epsilon=1e-5,
        max_position_embeddings=32,
    )


def tiny_multimodal_config() -> ModernVBERTConfig:
    return ModernVBERTConfig(
        text=tiny_config().model_copy(update={"vocab_size": 18}),
        vision=ModernVBERTVisionConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=8,
            patch_size=2,
            norm_epsilon=1e-6,
        ),
        image_token_id=17,
        pixel_shuffle_factor=2,
    )


def test_config_maps_nested_transformers_values():
    config = ModernVBERTTextConfig.from_hf_config(
        {
            "text_config": {
                "vocab_size": 17,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "layer_types": ["full_attention", "sliding_attention"],
                "local_attention": 4,
                "rope_parameters": {
                    "full_attention": {"rope_theta": 10_000.0},
                    "sliding_attention": {"rope_theta": 1_000.0},
                },
                "layer_norm_eps": 1e-5,
                "max_position_embeddings": 32,
                "initializer_range": 0.02,
            }
        }
    )

    assert config == tiny_config()
    assert config.head_dimension == 4


def test_native_encoder_is_jittable_differentiable_and_normalized():
    model = ModernVBERTTextEncoder.init(tiny_config(), key=jax.random.key(0))
    assert model.rematerialization == "full"
    batch = ModernVBERTTextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )

    hidden = eqx.filter_jit(lambda candidate, value: candidate.hidden_states(value))(
        model, batch
    )
    encoded = eqx.filter_jit(
        lambda candidate, value: candidate.encode(value, route=Route.QUERY)
    )(model, batch)
    assert hidden.shape == (2, 4, 8)
    assert encoded.shape == (2, 8)
    assert encoded.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(hidden))
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1.0, atol=1e-6)

    embeddings = model.tower.token_embedding[batch.input_ids]

    def objective(values):
        embedded = ModernVBERTTextBatch(
            inputs_embeds=values,
            attention_mask=batch.attention_mask,
        )
        return jnp.sum(model.encode(embedded, route=Route.DOCUMENT))

    gradients = jax.grad(objective)(embeddings)
    assert gradients.shape == embeddings.shape
    assert jnp.all(jnp.isfinite(gradients))


def _text_scan(model, batch):
    program = jax.make_jaxpr(lambda candidate: candidate.hidden_states(batch))(model)
    scans = [
        equation for equation in program.jaxpr.eqns if equation.primitive.name == "scan"
    ]
    assert len(scans) == 1
    return scans[0]


def test_text_depth_lowers_to_one_scan_with_explicit_rematerialization():
    batch = ModernVBERTTextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0]]),
    )

    full = ModernVBERTTextEncoder.init(
        tiny_config(),
        key=jax.random.key(3),
        rematerialization="full",
    )
    scan = _text_scan(full, batch)
    assert scan.params["length"] == tiny_config().num_hidden_layers
    remat_equations = scan.params["jaxpr"].jaxpr.eqns
    assert [equation.primitive.name for equation in remat_equations] == ["remat2"]
    assert (
        remat_equations[0].params["policy"] is jax.checkpoint_policies.nothing_saveable
    )

    selective = ModernVBERTTextEncoder.init(
        tiny_config(),
        key=jax.random.key(3),
        rematerialization="selective",
    )
    scan = _text_scan(selective, batch)
    remat_equations = scan.params["jaxpr"].jaxpr.eqns
    assert [equation.primitive.name for equation in remat_equations] == ["remat2"]
    assert (
        remat_equations[0].params["policy"]
        is jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    )

    uncheckpointed = ModernVBERTTextEncoder.init(
        tiny_config(),
        key=jax.random.key(3),
        rematerialization="none",
    )
    scan = _text_scan(uncheckpointed, batch)
    assert "remat2" not in {
        equation.primitive.name for equation in scan.params["jaxpr"].jaxpr.eqns
    }


def test_rematerialization_policies_preserve_values_and_parameter_gradients():
    batch = ModernVBERTTextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0], [4, 5, 6, 7]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 1]]),
    )

    def evaluate(policy):
        model = ModernVBERTTextEncoder.init(
            tiny_config(),
            key=jax.random.key(11),
            rematerialization=policy,
        )
        return eqx.filter_value_and_grad(
            lambda candidate: jnp.sum(candidate.encode(batch, route=Route.DOCUMENT))
        )(model)

    expected_value, expected_gradient = evaluate("none")
    for policy in ("selective", "full"):
        actual_value, actual_gradient = evaluate(policy)
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)
        expected_leaves = jax.tree.leaves(
            eqx.filter(expected_gradient, eqx.is_inexact_array)
        )
        actual_leaves = jax.tree.leaves(
            eqx.filter(actual_gradient, eqx.is_inexact_array)
        )
        assert len(actual_leaves) == len(expected_leaves)
        for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_local_attention_matches_explicit_dense_value_and_gradient():
    query_key, key_key, value_key = jax.random.split(jax.random.key(9), 3)
    query = jax.random.normal(query_key, (2, 7, 2, 4))
    key = jax.random.normal(key_key, (2, 7, 2, 4))
    value = jax.random.normal(value_key, (2, 7, 2, 4))
    mask = jnp.asarray([[1] * 7, [1] * 6 + [0]], dtype=bool)
    radius = 2

    def dense(q):
        positions = jnp.arange(q.shape[1])
        allowed = (jnp.abs(positions[:, None] - positions[None, :]) <= radius)[
            None, None, :, :
        ] & mask[:, None, None, :]
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, key) * (q.shape[-1] ** -0.5)
        probabilities = jax.nn.softmax(jnp.where(allowed, scores, -jnp.inf), axis=-1)
        return jnp.einsum("bhqk,bkhd->bqhd", probabilities, value)

    def native(q):
        return _scaled_dot_product_attention(
            q,
            key,
            value,
            mask,
            local_radius=radius,
            implementation="xla",
        )

    expected, expected_gradient = jax.value_and_grad(lambda q: jnp.sum(dense(q)))(query)
    actual, actual_gradient = jax.value_and_grad(lambda q: jnp.sum(native(q)))(query)

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-6, atol=2e-6)


def test_hf_state_dict_mapping_round_trips_the_native_tree():
    config = tiny_config()
    model = ModernVBERTTextEncoder.init(config, key=jax.random.key(5))
    adapter = ModernVBERTTextCheckpointAdapter(
        model_id="test/modernvbert", revision="fixture"
    )

    state_dict = adapter.state_dict(model)
    restored = adapter.from_state_dict(
        config,
        state_dict,
        rematerialization="selective",
    )

    assert set(state_dict) == set(modernvbert_text_weight_map(config).values())
    assert restored.rematerialization == "selective"
    assert len(restored.tower.layers) == config.num_hidden_layers
    assert restored.tower.layers[0].attention_norm is None
    assert restored.tower.layers[1].attention_norm is not None
    assert not bool(restored.tower.layers[0].sliding_attention)
    assert bool(restored.tower.layers[1].sliding_attention)
    assert restored.tower.layers.blocks.attention.qkv.weight.shape == (
        config.num_hidden_layers,
        3 * config.hidden_size,
        config.hidden_size,
    )
    native_parameter_count = sum(
        leaf.size for leaf in jax.tree.leaves(model) if eqx.is_inexact_array(leaf)
    )
    upstream_parameter_count = sum(value.size for value in state_dict.values())
    assert native_parameter_count == upstream_parameter_count
    original_leaves = jax.tree.leaves(model.tower)
    restored_leaves = jax.tree.leaves(restored.tower)
    assert len(original_leaves) == len(restored_leaves)
    for original, actual in zip(original_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_hf_state_dict_rejects_incompatible_shapes_and_unmapped_biases():
    config = tiny_config()
    model = ModernVBERTTextEncoder.init(config, key=jax.random.key(6))
    adapter = ModernVBERTTextCheckpointAdapter()
    state_dict = adapter.state_dict(model)
    embedding_name = modernvbert_text_weight_map(config)["tower.token_embedding"]
    state_dict[embedding_name] = state_dict[embedding_name][:-1]

    with np.testing.assert_raises_regex(ValueError, "expected"):
        adapter.from_state_dict(config, state_dict)

    biased_qkv = replace(
        model.tower.layers[0].attention.qkv,
        bias=jnp.zeros((3 * config.hidden_size,)),
    )
    biased_attention = replace(model.tower.layers[0].attention, qkv=biased_qkv)
    biased_layer = replace(model.tower.layers[0], attention=biased_attention)
    biased_tower = replace(
        model.tower,
        layers=(biased_layer, *model.tower.layers[1:]),
    )
    biased_model = replace(model, tower=biased_tower)
    with np.testing.assert_raises_regex(ValueError, "biasless"):
        adapter.state_dict(biased_model)


def test_pixel_shuffle_preserves_upstream_token_and_channel_order():
    hidden = jnp.arange(16 * 2).reshape(1, 16, 2)
    actual = pixel_shuffle(hidden, factor=2)
    expected = np.asarray(
        [
            [0, 1, 2, 3, 8, 9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15],
            [16, 17, 18, 19, 24, 25, 26, 27],
            [20, 21, 22, 23, 28, 29, 30, 31],
        ]
    )[None]
    np.testing.assert_array_equal(actual, expected)


def test_image_features_fill_image_tokens_in_record_order():
    token_embeddings = jnp.zeros((2, 7, 2))
    image_features = jnp.asarray(
        [
            [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]],
            [[[5.0, 50.0], [6.0, 60.0]], [[7.0, 70.0], [8.0, 80.0]]],
        ]
    )
    input_ids = jnp.asarray([[0, 17, 17, 1, 17, 17, 2], [17, 17, 3, 4, 5, 6, 7]])
    image_valid = jnp.asarray([[True, True], [True, False]])

    actual = merge_image_features(
        input_ids,
        token_embeddings,
        image_features,
        image_token_id=17,
        image_valid=image_valid,
    )

    np.testing.assert_array_equal(
        actual[0, [1, 2, 4, 5]], image_features[0].reshape(4, 2)
    )
    np.testing.assert_array_equal(actual[1, [0, 1]], image_features[1, 0])
    np.testing.assert_array_equal(actual[1, 2:], 0.0)


def test_multimodal_encoder_is_jittable_and_pixel_differentiable():
    config = tiny_multimodal_config()
    model = ModernVBERTEncoder.init(config, key=jax.random.key(21))
    input_ids = jnp.asarray([[1, 17, 17, 17, 17, 2]])
    pixels = jax.random.normal(jax.random.key(22), (1, 1, 3, 8, 8))

    def objective(candidate, pixel_values):
        batch = ModernVBERTBatch(
            input_ids=input_ids,
            attention_mask=jnp.ones_like(input_ids),
            pixel_values=pixel_values,
        )
        features = candidate.image_features(pixel_values)
        encoded = candidate.encode(batch, route=Route.GENERIC)
        return jnp.sum(encoded), (features, encoded)

    value, (features, encoded) = eqx.filter_jit(objective)(model, pixels)
    assert value.shape == ()
    assert features.shape == (1, 1, 4, 8)
    assert encoded.shape == (1, 8)
    pixel_gradient = jax.jit(jax.grad(lambda value: objective(model, value)[0]))(pixels)
    assert pixel_gradient.shape == pixels.shape
    assert jnp.all(jnp.isfinite(pixel_gradient))
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1.0, atol=1e-6)


def test_multimodal_hf_mapping_round_trips_the_executed_tree():
    config = tiny_multimodal_config()
    model = ModernVBERTEncoder.init(config, key=jax.random.key(23))
    adapter = ModernVBERTCheckpointAdapter(
        model_id="test/modernvbert", revision="fixture"
    )

    state_dict = adapter.state_dict(model)
    restored = adapter.from_state_dict(config, state_dict)
    expected_names = set(modernvbert_text_weight_map(config.text).values()) | set(
        modernvbert_vision_weight_map(config.vision).values()
    )

    assert set(state_dict) == expected_names
    original_leaves = jax.tree.leaves(model)
    restored_leaves = jax.tree.leaves(restored)
    assert len(original_leaves) == len(restored_leaves)
    for original, actual in zip(original_leaves, restored_leaves, strict=True):
        np.testing.assert_array_equal(actual, original)
