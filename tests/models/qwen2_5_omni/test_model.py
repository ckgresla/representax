from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.core import Route
from representax.models.qwen2_5_omni import (
    Qwen2_5OmniAudioConfig,
    Qwen2_5OmniCheckpointAdapter,
    Qwen2_5OmniConfig,
    Qwen2_5OmniEncoder,
    Qwen2_5OmniTextConfig,
    Qwen2_5OmniVisionConfig,
    audio_layout,
    batch_from_processor_output,
    qwen2_5_omni_weight_names,
    text_rotary_embedding,
    vision_layout,
)
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state


def tiny_config() -> Qwen2_5OmniConfig:
    text = Qwen2_5OmniTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dimension=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        mrope_section=(1, 1, 2),
        norm_epsilon=1e-6,
        layer_types=("full_attention", "full_attention"),
    )
    vision = Qwen2_5OmniVisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=24,
        num_attention_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        output_size=16,
        window_size=8,
        full_attention_layers=(1,),
    )
    audio = Qwen2_5OmniAudioConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_mel_bins=4,
        max_source_positions=32,
        window_size=4,
        output_size=16,
    )
    return Qwen2_5OmniConfig(
        text=text,
        vision=vision,
        audio=audio,
        pad_token_id=0,
        image_token_id=1,
        video_token_id=2,
        audio_token_id=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
        audio_start_token_id=6,
        audio_end_token_id=7,
        position_ids_per_second=25,
        seconds_per_chunk=2,
    )


def test_sectioned_text_rotary_embedding() -> None:
    config = tiny_config().text
    positions = jnp.asarray(
        [
            [[0, 1, 2]],
            [[0, 3, 4]],
            [[0, 5, 6]],
        ]
    )
    cosine, sine = text_rotary_embedding(config, positions)
    inverse = 1.0 / (
        config.rope_theta
        ** (np.arange(0, config.head_dimension, 2) / config.head_dimension)
    )
    frequency = np.asarray(positions)[..., None] * inverse
    embedding = np.concatenate((frequency, frequency), axis=-1)
    expected = np.concatenate(
        (
            embedding[0, ..., 0:1],
            embedding[1, ..., 1:2],
            embedding[2, ..., 2:4],
            embedding[0, ..., 4:5],
            embedding[1, ..., 5:6],
            embedding[2, ..., 6:8],
        ),
        axis=-1,
    )
    np.testing.assert_allclose(cosine, np.cos(expected), atol=1e-6)
    np.testing.assert_allclose(sine, np.sin(expected), atol=1e-6)


def test_vision_layout_is_window_ordered_and_invertible() -> None:
    config = tiny_config().vision
    layout = vision_layout(((1, 4, 8),), config, patch_bucket=64)
    order = layout["patch_order"]
    assert sorted(order.tolist()) == list(range(64))
    merged_order = order.reshape((-1, config.spatial_merge_unit))[:, 0] // 4
    restored = merged_order[layout["reverse_merged_indices"]]
    np.testing.assert_array_equal(restored, np.arange(16))
    assert layout["patch_valid"].sum() == 32
    assert np.unique(layout["window_segment_ids"][:32]).size == 2
    assert np.unique(layout["full_segment_ids"][:32]).tolist() == [0]


def test_audio_layout_preserves_exact_pool_pairs_across_chunks() -> None:
    config = tiny_config()
    features = np.arange(4 * 13, dtype=np.float32).reshape((1, 4, 13))
    mask = np.ones((1, 13), dtype=bool)
    layout = audio_layout(
        features,
        mask,
        config,
        chunk_count_buckets=(2,),
        token_count_buckets=(4,),
    )
    assert layout["input_features"].shape == (2, 4, 8)
    assert layout["after_cnn_valid"].sum() == 7
    np.testing.assert_array_equal(
        layout["pool_indices"][:3],
        np.asarray(((0, 1), (2, 3), (4, 5))),
    )
    assert layout["token_valid"].tolist() == [True, True, True, False]


def test_checkpoint_state_dict_round_trip_is_exact() -> None:
    config = tiny_config()
    model = Qwen2_5OmniEncoder.init(
        config,
        key=jax.random.key(11),
        rematerialization="none",
    )
    adapter = Qwen2_5OmniCheckpointAdapter(rematerialization="none")
    state = adapter.state_dict(model)
    assert set(state) == set(qwen2_5_omni_weight_names(config)) | {"lm_head.weight"}
    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    restored_state = adapter.state_dict(restored)
    for name in state:
        np.testing.assert_array_equal(restored_state[name], state[name])


def test_text_image_audio_forward_and_gradient_are_finite() -> None:
    config = tiny_config()
    pixel_values = (
        np.arange(16 * config.vision.patch_dimension, dtype=np.float32).reshape(
            (16, config.vision.patch_dimension)
        )
        / 100
    )
    audio_features = np.arange(4 * 8, dtype=np.float32).reshape((1, 4, 8)) / 100
    features = {
        "input_ids": np.asarray([[8, 4, 1, 1, 1, 1, 5, 6, 3, 3, 7, 9]]),
        "attention_mask": np.ones((1, 12), dtype=np.int32),
        "pixel_values": pixel_values,
        "image_grid_thw": np.asarray(((1, 4, 4),)),
        "input_features": audio_features,
        "feature_attention_mask": np.ones((1, 8), dtype=np.int32),
    }
    batch = batch_from_processor_output(
        features,
        config,
        sequence_length_buckets=(16,),
        patch_count_buckets=(16,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(2,),
    )
    model = Qwen2_5OmniEncoder.init(
        config,
        key=jax.random.key(7),
        rematerialization="none",
    )
    encoded = model.encode(batch, route=Route.GENERIC)
    assert encoded.shape == (1, 16)
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1, atol=1e-5)

    def loss(candidate: Qwen2_5OmniEncoder):
        return jnp.sum(candidate.encode(batch, route=Route.GENERIC) ** 2)

    _, gradients = eqx.filter_value_and_grad(loss)(model)
    gradient_leaves = [
        value for value in jax.tree.leaves(gradients) if eqx.is_array(value)
    ]
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(value))) for value in gradient_leaves)


def test_multimodal_encoder_runs_three_generic_training_steps() -> None:
    config = tiny_config()
    pixels = (
        np.arange(16 * config.vision.patch_dimension, dtype=np.float32).reshape(
            (16, config.vision.patch_dimension)
        )
        / 100
    )
    features = {
        "input_ids": np.asarray([[8, 4, 1, 1, 1, 1, 5, 6, 3, 3, 7, 9]]),
        "attention_mask": np.ones((1, 12), dtype=np.int32),
        "pixel_values": pixels,
        "image_grid_thw": np.asarray(((1, 4, 4),)),
        "input_features": (np.arange(4 * 8, dtype=np.float32).reshape((1, 4, 8)) / 100),
        "feature_attention_mask": np.ones((1, 8), dtype=np.int32),
    }
    left = batch_from_processor_output(
        features,
        config,
        sequence_length_buckets=(16,),
        patch_count_buckets=(16,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(2,),
    )
    right = eqx.tree_at(
        lambda value: value.input_ids,
        left,
        left.input_ids.at[0, -1].set(10),
    )
    batch = pairwise_batch(left=left, right=right, labels=np.asarray([0.25]))
    model = Qwen2_5OmniEncoder.init(
        config,
        key=jax.random.key(23),
        rematerialization="none",
    )
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    step = build_train_step(CosineRegressionTask(), optimizer, max_grad_norm=1.0)
    losses = []
    update_norms = []
    for _ in range(3):
        result = step(state, batch, None)
        state = result.state
        losses.append(float(result.metrics.loss))
        update_norms.append(float(result.metrics.update_global_norm))
        assert bool(result.metrics.numeric_finite)
        assert not bool(result.metrics.skipped_update)
    assert int(state.step) == 3
    assert all(np.isfinite(loss) for loss in losses)
    assert all(norm > 0 for norm in update_norms)
