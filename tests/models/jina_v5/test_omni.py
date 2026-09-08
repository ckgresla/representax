"""Native Jina v5 Omni architecture and lifecycle contracts."""

from __future__ import annotations

from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import EncoderMetadata, Modality, Route
from representax.models.jina_v5 import (
    JinaV5OmniCheckpointAdapter,
    JinaV5OmniConfig,
    JinaV5OmniEncoder,
    batch_from_processor_output,
    jina_v5_omni_weight_names,
    make_jina_v5_omni_processor,
)
from representax.models.qwen2_5_omni import (
    Qwen2_5OmniAudioConfig,
    Qwen2_5OmniAudioTower,
)
from representax.models.qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
from representax.models.qwen3_vl.text import Qwen3VLTextTower
from representax.models.qwen3_vl.vision import Qwen3VLVisionTower
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state


def tiny_config(variant: str = "nano") -> JinaV5OmniConfig:
    return JinaV5OmniConfig(
        variant=variant,
        text=Qwen3VLTextConfig(
            vocab_size=48,
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2 if variant == "nano" else 1,
            head_dimension=4,
            max_position_embeddings=32,
            rope_theta=10_000.0,
            mrope_section=(1, 1, 0),
            norm_epsilon=1e-6,
            pad_token_id=0,
            qk_norm=variant == "small",
        ),
        vision=Qwen3VLVisionConfig(
            depth=2,
            hidden_size=8,
            intermediate_size=12,
            num_attention_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=2,
            temporal_patch_size=2,
            output_size=8,
            num_position_embeddings=16,
            deepstack_visual_indexes=(),
        ),
        audio=Qwen2_5OmniAudioConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_mel_bins=4,
            max_source_positions=32,
            window_size=4,
            output_size=8,
        ),
        audio_source_output_size=8,
        text_attention="bidirectional" if variant == "nano" else "causal",
        image_token_id=41,
        video_token_id=41 if variant == "nano" else 42,
        audio_token_id=43,
        vision_start_token_id=None if variant == "nano" else 44,
        vision_end_token_id=None if variant == "nano" else 45,
        audio_start_token_id=46,
        audio_end_token_id=47,
    )


def tiny_model(config: JinaV5OmniConfig) -> JinaV5OmniEncoder:
    text_key, vision_key, audio_key = jax.random.split(jax.random.key(7), 3)
    return JinaV5OmniEncoder(
        text=Qwen3VLTextTower.init(config.text, key=text_key, dtype=jnp.float32),
        vision=Qwen3VLVisionTower.init(
            config.vision, key=vision_key, dtype=jnp.float32
        ),
        audio=Qwen2_5OmniAudioTower.init(
            config.audio, key=audio_key, dtype=jnp.float32
        ),
        metadata=EncoderMetadata(
            model_id="test/jina-v5-omni",
            revision="fixture",
            output_dimension=config.text.hidden_size,
            routes=frozenset(Route),
            modalities=frozenset(
                {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
            ),
        ),
        config=config,
        compute_dtype=jnp.float32,
        attention_implementation="xla",
        rematerialization="none",
    )


def tiny_batch(config: JinaV5OmniConfig):
    image_token = config.image_token_id
    return batch_from_processor_output(
        {
            "input_ids": np.asarray(
                [[5, image_token, 46, 43, 43, 43, 47, 6]], dtype=np.int32
            ),
            "attention_mask": np.ones((1, 8), dtype=np.int32),
            "pixel_values": np.arange(
                4 * config.vision.patch_dimension, dtype=np.float32
            ).reshape((4, config.vision.patch_dimension))
            / 100,
            "image_grid_thw": np.asarray([[1, 2, 2]], dtype=np.int32),
            "input_features": np.arange(4 * 13, dtype=np.float32).reshape(1, 4, 13)
            / 100,
            "feature_attention_mask": np.ones((1, 13), dtype=np.int32),
        },
        config,
        sequence_length_buckets=(8,),
        patch_count_buckets=(4,),
        audio_chunk_count_buckets=(2,),
        audio_token_count_buckets=(3,),
    )


@pytest.mark.parametrize("variant", ["nano", "small"])
def test_both_architectures_execute_all_towers_and_round_trip(variant: str) -> None:
    config = tiny_config(variant)
    model = tiny_model(config)
    batch = tiny_batch(config)
    encoded = eqx.filter_jit(
        lambda candidate, inputs: candidate.encode(inputs, route=Route.QUERY)
    )(model, batch)

    assert encoded.shape == (1, config.text.hidden_size)
    assert bool(jnp.all(jnp.isfinite(encoded)))
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1.0, atol=1e-6)

    adapter = JinaV5OmniCheckpointAdapter(rematerialization="none")
    state = adapter.state_dict(model)
    assert frozenset(state) == jina_v5_omni_weight_names(config)
    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/jina-v5-omni",
        revision="fixture",
    )
    for name, expected in state.items():
        np.testing.assert_array_equal(adapter.state_dict(restored)[name], expected)


def test_processor_uses_checkpoint_image_bounds_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class Tokenizer:
        @classmethod
        def from_pretrained(cls, checkpoint, **options):
            return cls()

    class ImageProcessor:
        size = SimpleNamespace(shortest_edge=262_144, longest_edge=1_310_720)

        @classmethod
        def from_pretrained(cls, checkpoint, **options):
            assert options == {}
            return cls()

    monkeypatch.setattr(
        "representax.models.jina_v5.processing.import_module",
        lambda name: SimpleNamespace(
            Qwen2TokenizerFast=Tokenizer,
            Qwen2VLImageProcessor=ImageProcessor,
        ),
    )
    processor = make_jina_v5_omni_processor(
        tmp_path,
        tiny_config(),
        sequence_length_buckets=(8,),
        patch_count_buckets=(4,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(3,),
    )

    contract = processor.data_contract()
    assert contract["image_min_pixels"] == 262_144
    assert contract["image_max_pixels"] == 1_310_720
    assert contract["video_min_pixels"] == 262_144
    assert contract["video_max_pixels"] == 1_310_720


@pytest.mark.runtime
def test_omni_model_completes_one_optimizer_update() -> None:
    config = tiny_config("nano")
    model = tiny_model(config)
    left = tiny_batch(config)
    right = eqx.tree_at(
        lambda batch: batch.input_ids,
        tiny_batch(config),
        jnp.asarray([[7, 41, 46, 43, 43, 43, 47, 8]], dtype=jnp.int32),
    )
    batch = pairwise_batch(left=left, right=right, labels=jnp.asarray([0.5]))
    optimizer = optax.adamw(1e-3)
    result = build_train_step(CosineRegressionTask(), optimizer, max_grad_norm=1.0)(
        init_train_state(model, optimizer), batch, jax.random.key(9)
    )

    assert int(result.state.step) == 1
    assert bool(result.metrics.numeric_finite)
    assert not bool(result.metrics.skipped_update)
