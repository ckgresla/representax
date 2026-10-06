"""Native Qwen2/Qwen2.5-VL model contracts."""

from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models import Linear
from representax.models.adapters import LoRALinear
from representax.models.qwen2_5_omni.config import Qwen2_5OmniTextConfig
from representax.models.qwen2_vl import (
    Qwen2VLBatch,
    Qwen2VLCheckpointAdapter,
    Qwen2VLConfig,
    Qwen2VLEncoder,
    Qwen2VLReranker,
    Qwen2VLRerankerCheckpointAdapter,
    Qwen2VLVisionConfig,
    batch_from_processor_output,
    qwen2_vl_weight_names,
    vision_layout,
)


def tiny_config(generation="qwen2_5_vl") -> Qwen2VLConfig:
    text = Qwen2_5OmniTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dimension=4,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        mrope_section=(1, 1, 0),
        norm_epsilon=1e-6,
        layer_types=("full_attention", "full_attention"),
    )
    return Qwen2VLConfig(
        generation=generation,
        text=text,
        vision=Qwen2VLVisionConfig(
            generation=generation,
            depth=2,
            hidden_size=8,
            intermediate_size=12 if generation == "qwen2_5_vl" else 32,
            num_attention_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            output_size=8,
            window_size=8,
            full_attention_layers=(1,) if generation == "qwen2_5_vl" else (0, 1),
            norm="rms" if generation == "qwen2_5_vl" else "layer",
            mlp="swiglu" if generation == "qwen2_5_vl" else "quick_gelu",
        ),
        pad_token_id=0,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
    )


def tiny_batch() -> Qwen2VLBatch:
    return Qwen2VLBatch(
        input_ids=jnp.asarray([[1, 28, 29, 31, 2, 0]], dtype=jnp.int32),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 1, 0]], dtype=jnp.int32),
        position_ids=jnp.asarray(
            [[[0, 1, 2, 3, 4, 0]]] * 3,
            dtype=jnp.int32,
        ),
        pixel_values=jnp.arange(4 * 24, dtype=jnp.float32).reshape((4, 24)) / 100,
        patch_valid=jnp.ones((4,), dtype=bool),
        vision_full_segment_ids=jnp.zeros((4,), dtype=jnp.int32),
        vision_window_segment_ids=jnp.zeros((4,), dtype=jnp.int32),
        vision_position_ids=jnp.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32
        ),
        reverse_merged_indices=jnp.asarray([0], dtype=jnp.int32),
        visual_token_indices=jnp.asarray([2], dtype=jnp.int32),
        visual_token_valid=jnp.asarray([True]),
    )


@pytest.mark.parametrize("generation", ["qwen2_vl", "qwen2_5_vl"])
def test_encoder_is_jittable_and_differentiable(generation):
    model = Qwen2VLEncoder.init(tiny_config(generation), key=jax.random.key(0))
    batch = tiny_batch()
    hidden = eqx.filter_jit(lambda candidate, values: candidate.hidden_states(values))(
        model, batch
    )
    encoded = eqx.filter_jit(
        lambda candidate, values: candidate.encode(values, route=Route.QUERY)
    )(model, batch)
    assert hidden.shape == (1, 6, 8)
    assert encoded.shape == (1, 8)
    assert jnp.all(jnp.isfinite(hidden))
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1.0, atol=1e-6)

    assert batch.pixel_values is not None
    gradients = jax.grad(
        lambda pixels: jnp.sum(
            model.encode(
                eqx.tree_at(lambda value: value.pixel_values, batch, pixels),
                route=Route.DOCUMENT,
            )
        )
    )(batch.pixel_values)
    assert gradients.shape == batch.pixel_values.shape
    assert jnp.all(jnp.isfinite(gradients))


@pytest.mark.parametrize("generation", ["qwen2_vl", "qwen2_5_vl"])
def test_checkpoint_round_trip_covers_every_native_tensor(generation):
    config = tiny_config(generation)
    adapter = Qwen2VLCheckpointAdapter(rematerialization="selective")
    model = Qwen2VLEncoder.init(config, key=jax.random.key(7))
    state = adapter.state_dict(model)
    assert frozenset(state) == qwen2_vl_weight_names(config)
    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/qwen2-vl",
        revision="fixture",
    )
    for name, expected in state.items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], expected)


@pytest.mark.parametrize("generation", ["qwen2_vl", "qwen2_5_vl"])
def test_processor_output_becomes_one_finite_batch(generation):
    config = tiny_config(generation)
    pixels = np.arange(4 * config.vision.patch_dimension, dtype=np.float32).reshape(
        (4, config.vision.patch_dimension)
    )
    batch = batch_from_processor_output(
        {
            "input_ids": [[1, 28, 29, 31, 2]],
            "attention_mask": [[1, 1, 1, 1, 1]],
            "mm_token_type_ids": [[0, 0, 1, 0, 0]],
            "pixel_values": pixels,
            "image_grid_thw": [[1, 2, 2]],
        },
        config,
        sequence_length_buckets=(6,),
        patch_count_buckets=(8,),
        padding_side="left",
    )
    np.testing.assert_array_equal(batch.input_ids, [[0, 1, 28, 29, 31, 2]])
    assert batch.pixel_values is not None and batch.pixel_values.shape == (8, 24)
    assert batch.visual_token_indices is not None
    np.testing.assert_array_equal(batch.visual_token_indices, [3, 0])


def test_qwen2_layout_is_identity_while_qwen25_uses_window_order():
    qwen2 = vision_layout([(1, 8, 8)], tiny_config("qwen2_vl").vision)
    qwen25 = vision_layout([(1, 8, 8)], tiny_config("qwen2_5_vl").vision)
    np.testing.assert_array_equal(qwen2["patch_order"], np.arange(64))
    assert not np.array_equal(qwen25["patch_order"], np.arange(64))


def test_nomic_export_is_one_full_checkpoint_not_a_stale_peft_adapter(tmp_path):
    adapter = Qwen2VLCheckpointAdapter()
    model = Qwen2VLEncoder.init(
        tiny_config(),
        key=jax.random.key(41),
        model_id="nomic-ai/nomic-embed-multimodal-3b",
    )
    target = tmp_path / "nomic"
    target.mkdir()
    (target / "adapter_config.json").write_text("{}\n")
    (target / "adapter_model.safetensors").write_bytes(b"stale")

    adapter.save(model, target)

    config = json.loads((target / "config.json").read_text())
    assert config["representax_processor_mode"] == "nomic_embedding"
    assert (target / "model.safetensors").is_file()
    assert not (target / "adapter_config.json").exists()
    assert not (target / "adapter_model.safetensors").exists()
    restored = adapter.load(target, parameter_dtype=jnp.float32)
    for name, value in adapter.state_dict(model).items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], value)


def test_imported_lora_model_trains_only_adapter_parameters():
    model = Qwen2VLEncoder.init(tiny_config(), key=jax.random.key(43))
    base = model.text.layers.blocks.query
    adapted = LoRALinear(
        weight=base.weight,
        bias=base.bias,
        lora_a=jnp.zeros((*base.weight.shape[:-2], 2, base.weight.shape[-1])),
        lora_b=jnp.zeros((*base.weight.shape[:-2], base.weight.shape[-2], 2)),
        rank=2,
        alpha=4.0,
        weight_layout=base.weight_layout,
    )
    model = eqx.tree_at(
        lambda candidate: candidate.text.layers.blocks.query,
        model,
        adapted,
    )

    selected = model.training_filter()
    selected_paths = {
        jax.tree_util.keystr(path)
        for path, value in jax.tree_util.tree_flatten_with_path(selected)[0]
        if value
    }

    assert selected_paths == {
        ".text.layers.blocks.query.lora_a",
        ".text.layers.blocks.query.lora_b",
    }


def test_jina_reranker_checkpoint_round_trip_preserves_scoring_head(tmp_path):
    config = tiny_config("qwen2_vl")
    model = Qwen2VLReranker(
        model=Qwen2VLEncoder.init(config, key=jax.random.key(51)),
        hidden=Linear.init(
            8,
            8,
            key=jax.random.key(52),
            scale=0.02,
            dtype=jnp.float32,
            bias=True,
        ),
        output=Linear.init(
            8,
            1,
            key=jax.random.key(53),
            scale=0.02,
            dtype=jnp.float32,
            bias=True,
        ),
    )
    adapter = Qwen2VLRerankerCheckpointAdapter()
    target = adapter.save(model, tmp_path / "jina")
    config_json = json.loads((target / "config.json").read_text())
    assert config_json["architectures"] == ["JinaVLForRanking"]
    assert config_json["auto_map"] == {"AutoModel": "modeling.JinaVLForRanking"}
    assert config_json["representax_processor_mode"] == "reranking"

    restored = adapter.load(target, parameter_dtype=jnp.float32)
    for name, value in adapter.state_dict(model).items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], value)
