"""Fast contracts for the native LLaVA-NeXT family."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.llava_next import (
    LlavaNextBatch,
    LlavaNextCheckpointAdapter,
    LlavaNextConfig,
    LlavaNextEncoder,
    image_pack_indices,
    llava_next_weight_names,
)


def tiny_config(*, family: str = "llama") -> LlavaNextConfig:
    return LlavaNextConfig.from_hf_config(
        {
            "model_type": "llava_next",
            "image_token_index": 63,
            "pad_token_id": 0,
            "image_grid_pinpoints": [[8, 8]],
            "vision_feature_layer": -2,
            "vision_feature_select_strategy": "default",
            "projector_hidden_act": "gelu",
            "text_config": {
                "model_type": family,
                "vocab_size": 64,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 32,
                "rope_theta": 10_000.0,
                "rms_norm_eps": 1e-6,
                "pad_token_id": 0,
            },
            "vision_config": {
                "model_type": "clip_vision_model",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
                "num_channels": 3,
                "hidden_act": "quick_gelu",
                "layer_norm_eps": 1e-5,
            },
        }
    )


@pytest.mark.parametrize("family", ("llama", "mistral"))
def test_config_maps_both_text_families(family):
    config = tiny_config(family=family)
    assert config.text.family == family
    assert config.selected_vision_layer_count == 1
    assert config.selected_vision_tokens == 4
    assert LlavaNextConfig.from_hf_config(config.to_hf_config()) == config


def test_real_configs_with_null_pad_token_are_valid():
    value = tiny_config().to_hf_config()
    value["pad_token_id"] = None
    value["text_config"].pop("pad_token_id", None)
    assert LlavaNextConfig.from_hf_config(value).pad_token_id == 0


def test_any_resolution_plan_is_one_static_gather_program():
    sources, tile_valid = image_pack_indices(
        ((8, 8),), tiny_config(), image_bucket=1, tile_bucket=2
    )
    np.testing.assert_array_equal(sources, (0, 1, 2, 3, 4, 5, 8, 6, 7, 8))
    np.testing.assert_array_equal(tile_valid, ((True, True),))


def test_init_forward_and_checkpoint_round_trip(tmp_path):
    config = tiny_config()
    adapter = LlavaNextCheckpointAdapter()
    model = LlavaNextEncoder.init(config, key=jax.random.key(71))
    state = adapter.state_dict(model)
    assert set(state) == set(
        llava_next_weight_names(
            config,
            text_prefix="language_model.",
            vision_prefix="vision_tower.",
        )
    )

    batch = LlavaNextBatch(
        input_ids=jnp.asarray([[1, *([63] * 10), 2]]),
        attention_mask=jnp.ones((1, 12), dtype=bool),
        pixel_values=jnp.arange(2 * 3 * 8 * 8, dtype=jnp.float32).reshape(
            1, 2, 3, 8, 8
        ),
        tile_valid=jnp.ones((1, 2), dtype=bool),
        pack_indices=jnp.asarray([0, 1, 2, 3, 4, 5, 8, 6, 7, 8, 0, 0]),
        pack_valid=jnp.asarray([True] * 10 + [False, False]),
        visual_token_indices=jnp.arange(12, dtype=jnp.int32),
    )
    representation = jax.jit(lambda current: model.encode(current, route=Route.QUERY))(
        batch
    )
    assert representation.shape == (1, 8)
    assert bool(jnp.all(jnp.isfinite(representation)))

    target = adapter.save(model, tmp_path / "checkpoint")
    assert json.loads((target / "config.json").read_text())["model_type"] == (
        "llava_next"
    )
    restored = adapter.load(target)
    for name, expected in state.items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], expected)
