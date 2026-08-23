"""Fast contracts for native Llama Nemotron VL."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import Linear
from representax.models.decoder import rotary_embedding
from representax.models.nemotron_vl import (
    LlamaNemotronVLBackbone,
    LlamaNemotronVLBatch,
    LlamaNemotronVLCheckpointAdapter,
    LlamaNemotronVLConfig,
    LlamaNemotronVLEncoder,
    LlamaNemotronVLReranker,
    nemotron_vl_weight_names,
)


def tiny_config(mode: str = "embedding") -> LlamaNemotronVLConfig:
    reranking = mode == "reranking"
    return LlamaNemotronVLConfig.from_hf_config(
        {
            "model_type": (
                "llama_nemotron_vl_rerank" if reranking else "llama_nemotron_vl"
            ),
            "img_context_token_id": 63,
            "downsample_ratio": 0.5,
            "pooling": "avg",
            "id2label": {"0": "LABEL_0"}
            if reranking
            else {"0": "LABEL_0", "1": "LABEL_1"},
            "llm_config": {
                "model_type": "llama_bidirec",
                "architectures": ["LlamaBidirectionalModel"],
                "vocab_size": 64,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 64,
                "rope_theta": 10_000.0,
                "rope_scaling": {
                    "rope_type": "llama3",
                    "factor": 8,
                    "low_freq_factor": 1,
                    "high_freq_factor": 4,
                    "original_max_position_embeddings": 8,
                },
                "rms_norm_eps": 1e-5,
                "temperature": 1,
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 8,
                "patch_size": 4,
                "layer_norm_eps": 1e-6,
                "hidden_act": "gelu_pytorch_tanh",
            },
        }
    )


def tiny_model(mode: str = "embedding"):
    config = tiny_config(mode)
    backbone = LlamaNemotronVLBackbone.init(config, key=jax.random.key(101))
    metadata = EncoderMetadata(
        model_id="test/nemotron",
        revision="test",
        output_dimension=config.output_dimension,
        routes=frozenset(Route),
        modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
    )
    if mode == "embedding":
        return LlamaNemotronVLEncoder(model=backbone, metadata=metadata)
    return LlamaNemotronVLReranker(
        model=backbone,
        score=Linear.init(
            config.text.hidden_size,
            config.output_dimension,
            key=jax.random.key(103),
            scale=0.02,
            dtype=jnp.float32,
        ),
        metadata=metadata,
    )


def test_config_round_trips_bidirectional_llama3_rope():
    for mode in ("embedding", "reranking"):
        config = tiny_config(mode)
        assert config.text.attention_mode == "bidirectional"
        assert config.image_sequence_length == 1
        assert LlamaNemotronVLConfig.from_hf_config(config.to_hf_config()) == config


def test_llama3_rotary_scaling_changes_only_long_frequencies():
    config = tiny_config().text
    cosine, sine = rotary_embedding(config, jnp.asarray([[0, 1, 7]], dtype=jnp.int32))
    np.testing.assert_array_equal(cosine[:, 0], np.ones((1, 4)))
    np.testing.assert_array_equal(sine[:, 0], np.zeros((1, 4)))
    assert cosine.shape == sine.shape == (1, 3, 4)
    assert bool(jnp.all(jnp.isfinite(cosine)))


def test_bidirectional_text_and_image_forward():
    model = tiny_model()
    batch = LlamaNemotronVLBatch(
        input_ids=jnp.asarray([[1, 63, 2, 3]]),
        attention_mask=jnp.ones((1, 4), dtype=bool),
        pixel_values=jnp.arange(3 * 8 * 8, dtype=jnp.float32).reshape(1, 3, 8, 8),
        visual_token_indices=jnp.asarray([1], dtype=jnp.int32),
        visual_token_valid=jnp.asarray([True]),
    )
    output = jax.jit(lambda value: model.encode(value, route=Route.DOCUMENT))(batch)
    assert output.shape == (1, 8)
    assert bool(jnp.all(jnp.isfinite(output)))

    first = model.model.hidden_states(
        LlamaNemotronVLBatch(
            input_ids=jnp.asarray([[1, 2, 3]]),
            attention_mask=jnp.ones((1, 3), dtype=bool),
        )
    )[:, 0]
    changed = model.model.hidden_states(
        LlamaNemotronVLBatch(
            input_ids=jnp.asarray([[1, 2, 4]]),
            attention_mask=jnp.ones((1, 3), dtype=bool),
        )
    )[:, 0]
    assert not bool(jnp.allclose(first, changed))


def test_image_cotangent_is_finite_and_nonzero():
    model = tiny_model()
    pixels = jnp.arange(3 * 8 * 8, dtype=jnp.float32).reshape(1, 3, 8, 8)

    def objective(pixel_values):
        batch = LlamaNemotronVLBatch(
            input_ids=jnp.asarray([[1, 63, 2, 3]]),
            attention_mask=jnp.ones((1, 4), dtype=bool),
            pixel_values=pixel_values,
            visual_token_indices=jnp.asarray([1], dtype=jnp.int32),
            visual_token_valid=jnp.asarray([True]),
        )
        return jnp.sum(model.encode(batch, route=Route.DOCUMENT))

    gradient = jax.grad(objective)(pixels)
    assert gradient.shape == pixels.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0


@pytest.mark.parametrize("mode", ("embedding", "reranking"))
def test_checkpoint_round_trip(mode, tmp_path):
    model = tiny_model(mode)
    adapter = LlamaNemotronVLCheckpointAdapter()
    state = adapter.state_dict(model)
    assert frozenset(state) == nemotron_vl_weight_names(model.model.config)
    target = adapter.save(model, tmp_path / mode)
    exported = json.loads((target / "config.json").read_text())
    assert exported["model_type"] == (
        "llama_nemotron_vl_rerank" if mode == "reranking" else "llama_nemotron_vl"
    )
    assert exported["vision_config"]["vision_use_head"] is False
    restored = adapter.load(target)
    actual = adapter.state_dict(restored)
    for name, expected in state.items():
        np.testing.assert_array_equal(actual[name], expected)
