"""Fast contracts for the native BidirLM Omni implementation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.bidirlm_omni import (
    BidirLMOmniAudioConfig,
    BidirLMOmniBatch,
    BidirLMOmniCheckpointAdapter,
    BidirLMOmniConfig,
    BidirLMOmniEncoder,
    bidirlm_omni_weight_names,
    convolution_output_length,
)
from representax.models.qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig


def tiny_config() -> BidirLMOmniConfig:
    return BidirLMOmniConfig(
        text=Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dimension=4,
            max_position_embeddings=32,
            rope_theta=10_000.0,
            mrope_section=(1, 1, 0),
            norm_epsilon=1e-6,
            pad_token_id=0,
        ),
        vision=Qwen3VLVisionConfig(
            depth=2,
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=1,
            output_size=16,
            num_position_embeddings=16,
            deepstack_visual_indexes=(0, 1),
        ),
        audio=BidirLMOmniAudioConfig(
            num_mel_bins=8,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_size=8,
            downsample_hidden_size=4,
            output_size=16,
            max_source_positions=32,
            window_size=4,
            inference_window_size=8,
            convolution_chunk_size=4,
        ),
        audio_token_id=10,
        audio_start_token_id=11,
        audio_end_token_id=12,
        image_token_id=13,
        video_token_id=14,
        vision_start_token_id=15,
        vision_end_token_id=16,
    )


def test_text_forward_is_bidirectional_and_mean_pooled() -> None:
    model = BidirLMOmniEncoder.init(
        tiny_config(), key=jax.random.key(3), rematerialization="none"
    )
    first = BidirLMOmniBatch(
        input_ids=jnp.asarray(((1, 2, 3),)),
        attention_mask=jnp.ones((1, 3), dtype=bool),
    )
    changed_future = BidirLMOmniBatch(
        input_ids=jnp.asarray(((1, 2, 4),)),
        attention_mask=jnp.ones((1, 3), dtype=bool),
    )
    first_hidden = model.hidden_states(first)
    changed_hidden = model.hidden_states(changed_future)
    assert first_hidden.shape == (1, 3, 16)
    assert not np.allclose(first_hidden[:, 0], changed_hidden[:, 0])
    actual = model.encode(first, route=Route.GENERIC)
    np.testing.assert_allclose(
        actual,
        np.asarray(first_hidden, dtype=np.float32).mean(axis=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_audio_padding_is_masked_at_every_convolution_boundary() -> None:
    model = BidirLMOmniEncoder.init(
        tiny_config(), key=jax.random.key(5), rematerialization="none"
    )
    values = jax.random.normal(jax.random.key(7), (1, 8, 10))
    padded = jnp.pad(values, ((0, 0), (0, 0), (0, 6)))
    output_length = int(convolution_output_length(10))
    short = model.audio(
        values,
        jnp.asarray((10,)),
        jnp.ones((output_length,), dtype=bool),
        jnp.zeros((output_length,), dtype=jnp.int32),
        compute_dtype=jnp.float32,
        attention_implementation="xla",
        rematerialization="none",
    )
    padded_output_length = int(convolution_output_length(16))
    long = model.audio(
        padded,
        jnp.asarray((10,)),
        jnp.asarray(
            [True] * output_length + [False] * (padded_output_length - output_length)
        ),
        jnp.asarray(
            [0] * output_length + [-1] * (padded_output_length - output_length)
        ),
        compute_dtype=jnp.float32,
        attention_implementation="xla",
        rematerialization="none",
    )
    np.testing.assert_allclose(short, long[:output_length], rtol=2e-5, atol=2e-5)


def test_audio_input_and_parameter_gradients_are_finite() -> None:
    model = BidirLMOmniEncoder.init(
        tiny_config(), key=jax.random.key(9), rematerialization="none"
    )
    features = jax.random.normal(jax.random.key(10), (1, 8, 10))

    def batch(value):
        return BidirLMOmniBatch(
            input_ids=jnp.asarray(((1, 10, 10, 2),)),
            attention_mask=jnp.ones((1, 4), dtype=bool),
            input_features=value,
            audio_chunk_lengths=jnp.asarray((10,)),
            audio_output_valid=jnp.ones((2,), dtype=bool),
            audio_segment_ids=jnp.zeros((2,), dtype=jnp.int32),
            audio_feature_indices=jnp.asarray((0, 1)),
            audio_token_indices=jnp.asarray((1, 2)),
            audio_token_valid=jnp.ones((2,), dtype=bool),
        )

    def input_loss(value):
        return jnp.square(model.encode(batch(value), route=Route.GENERIC)).mean()

    input_gradient = jax.grad(input_loss)(features)
    assert input_gradient.shape == features.shape
    assert bool(jnp.all(jnp.isfinite(input_gradient)))
    assert float(jnp.linalg.norm(input_gradient)) > 0

    def parameter_loss(candidate):
        return jnp.square(candidate.encode(batch(features), route=Route.GENERIC)).mean()

    parameter_gradient = jax.grad(parameter_loss)(model)
    leaves = [
        leaf
        for leaf in jax.tree.leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
    assert any(float(jnp.linalg.norm(leaf)) > 0 for leaf in leaves)


def test_checkpoint_adapter_roundtrips_every_executed_leaf() -> None:
    config = tiny_config()
    adapter = BidirLMOmniCheckpointAdapter(rematerialization="selective")
    model = BidirLMOmniEncoder.init(
        config, key=jax.random.key(11), rematerialization="none"
    )
    state = adapter.state_dict(model)
    assert frozenset(state) == bidirlm_omni_weight_names(config)
    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/bidirlm-omni",
        revision="fixture",
    )
    assert restored.rematerialization == "selective"
    actual = adapter.state_dict(restored)
    for name, expected in state.items():
        np.testing.assert_array_equal(actual[name], expected)


def test_real_config_omits_the_unreachable_third_deepstack_tap() -> None:
    config = BidirLMOmniConfig.from_hf_config(
        {
            "model_type": "bidirlm_omni",
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "rope_scaling": {"mrope_section": [1, 1, 0]},
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_heads": 2,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "out_hidden_size": 16,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0, 1, 2],
            },
            "audio_config": {
                "num_mel_bins": 8,
                "encoder_layers": 2,
                "encoder_attention_heads": 2,
                "encoder_ffn_dim": 16,
                "d_model": 8,
                "downsample_hidden_size": 4,
                "output_dim": 16,
                "max_source_positions": 32,
                "n_window": 4,
                "n_window_infer": 8,
                "conv_chunksize": 4,
            },
            "audio_token_id": 10,
            "audio_start_token_id": 11,
            "audio_end_token_id": 12,
            "image_token_id": 13,
            "video_token_id": 14,
            "vision_start_token_id": 15,
            "vision_end_token_id": 16,
        }
    )
    assert config.vision.deepstack_visual_indexes == (0, 1)
