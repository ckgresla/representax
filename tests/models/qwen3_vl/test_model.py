"""Native Qwen3-VL model contracts."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import Route
from representax.models.qwen3_vl import (
    Qwen3VLBatch,
    Qwen3VLCheckpointAdapter,
    Qwen3VLConfig,
    Qwen3VLEncoder,
    Qwen3VLReranker,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
    batch_from_processor_output,
    last_valid_token_indices,
    make_qwen3_vl_processor,
    multimodal_position_ids,
    qwen3_vl_weight_names,
    vision_layout,
)
from representax.models.qwen3_vl.processing import _reranking_conversation


def tiny_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text=Qwen3VLTextConfig(
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
            pad_token_id=0,
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
            deepstack_visual_indexes=(0,),
        ),
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
    )


def tiny_batch() -> Qwen3VLBatch:
    return Qwen3VLBatch(
        input_ids=jnp.asarray([[1, 28, 29, 31, 2, 0]], dtype=jnp.int32),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 1, 0]], dtype=jnp.int32),
        position_ids=jnp.asarray(
            [[[0, 1, 2, 3, 4, 0]], [[0, 1, 2, 3, 4, 0]], [[0, 1, 2, 3, 4, 0]]],
            dtype=jnp.int32,
        ),
        pixel_values=jnp.arange(4 * 24, dtype=jnp.float32).reshape(4, 24) / 100,
        patch_valid=jnp.ones((4,), dtype=bool),
        vision_segment_ids=jnp.zeros((4,), dtype=jnp.int32),
        vision_position_ids=jnp.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32
        ),
        position_interpolation_indices=jnp.asarray([[0, 1, 4, 5]] * 4, dtype=jnp.int32),
        position_interpolation_weights=jnp.asarray(
            [[1.0, 1.0, 1.0, 1.0], [0.0] * 4, [0.0] * 4, [0.0] * 4]
        ),
        visual_token_indices=jnp.asarray([2], dtype=jnp.int32),
        visual_token_valid=jnp.asarray([True]),
    )


def test_nested_hf_configuration_maps_exactly():
    config = Qwen3VLConfig.from_hf_config(
        {
            "model_type": "qwen3_vl",
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "max_position_embeddings": 32,
                "rope_theta": 10_000.0,
                "rope_scaling": {"mrope_section": [1, 1, 0]},
                "rms_norm_eps": 1e-6,
                "pad_token_id": 0,
            },
            "vision_config": {
                "depth": 2,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 8,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0],
            },
            "image_token_id": 29,
            "video_token_id": 30,
            "vision_start_token_id": 28,
            "vision_end_token_id": 31,
        }
    )
    assert config == tiny_config()


def test_multimodal_encoder_and_reranker_are_jittable_and_differentiable():
    model = Qwen3VLEncoder.init(tiny_config(), key=jax.random.key(0))
    batch = tiny_batch()
    hidden = eqx.filter_jit(lambda candidate, values: candidate.hidden_states(values))(
        model, batch
    )
    encoded = eqx.filter_jit(
        lambda candidate, values: candidate.encode(values, route=Route.QUERY)
    )(model, batch)
    score = eqx.filter_jit(lambda candidate, values: candidate.score(values))(
        Qwen3VLReranker(model, true_token_id=3, false_token_id=4),
        batch,
    )

    assert hidden.shape == (1, 6, 8)
    assert encoded.shape == (1, 8)
    assert score.shape == (1,)
    assert jnp.all(jnp.isfinite(hidden))
    assert jnp.all(jnp.isfinite(score))
    np.testing.assert_allclose(jnp.linalg.norm(encoded, axis=-1), 1.0, atol=1e-6)

    def objective(pixel_values):
        values = eqx.tree_at(lambda item: item.pixel_values, batch, pixel_values)
        return jnp.sum(model.encode(values, route=Route.DOCUMENT))

    assert batch.pixel_values is not None
    gradients = jax.grad(objective)(batch.pixel_values)
    assert gradients.shape == batch.pixel_values.shape
    assert jnp.all(jnp.isfinite(gradients))


def test_last_token_pooling_accepts_left_and_right_padding():
    masks = jnp.asarray(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 0, 0],
        ],
        dtype=jnp.int32,
    )
    np.testing.assert_array_equal(last_valid_token_indices(masks), [2, 4, 2])


def test_text_and_vision_depth_each_lower_to_one_scan():
    model = Qwen3VLEncoder.init(tiny_config(), key=jax.random.key(1))
    program = jax.make_jaxpr(lambda candidate: candidate.hidden_states(tiny_batch()))(
        model
    )

    def scans(jaxpr):
        found = []
        for equation in jaxpr.eqns:
            if equation.primitive.name == "scan":
                found.append(equation)
            for value in equation.params.values():
                nested = getattr(value, "jaxpr", None)
                if nested is not None:
                    found.extend(scans(nested))
        return found

    lengths = sorted(equation.params["length"] for equation in scans(program.jaxpr))
    assert lengths.count(2) >= 2


def test_checkpoint_tensor_inventory_round_trips_every_native_leaf():
    config = tiny_config()
    adapter = Qwen3VLCheckpointAdapter(rematerialization="selective")
    model = Qwen3VLEncoder.init(
        config,
        key=jax.random.key(7),
        rematerialization="none",
    )
    state = adapter.state_dict(model)
    assert frozenset(state) == qwen3_vl_weight_names(config)

    restored = adapter.from_state_dict(
        config,
        state,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/qwen3-vl",
        revision="fixture",
    )
    assert restored.rematerialization == "selective"
    actual = adapter.state_dict(restored)
    for name, expected in state.items():
        np.testing.assert_array_equal(actual[name], expected)


def test_processor_output_becomes_one_padded_finite_shape_batch():
    config = tiny_config()
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
        sequence_length_buckets=(6, 8),
        patch_count_buckets=(8,),
    )
    assert batch.input_ids.shape == (1, 6)
    assert batch.pixel_values is not None
    assert batch.pixel_values.shape == (8, config.vision.patch_dimension)
    assert batch.patch_valid is not None
    np.testing.assert_array_equal(batch.patch_valid, [True] * 4 + [False] * 4)
    assert batch.visual_token_valid is not None
    np.testing.assert_array_equal(batch.visual_token_valid, [True, False])
    assert batch.visual_token_indices is not None
    np.testing.assert_array_equal(batch.visual_token_indices, [2, 0])


def test_reranking_conversation_accepts_generic_query_document_pairs():
    conversation = _reranking_conversation(
        ("which harbor?", "a quiet harbor"),
        default_instruction="retrieve relevant passages",
    )

    content = conversation[1]["content"]
    assert [part["text"] for part in content if part["type"] == "text"] == [
        "<Instruct>: retrieve relevant passages",
        "<Query>:",
        "which harbor?",
        "\n<Document>:",
        "a quiet harbor",
    ]


def test_reranking_processor_accepts_pointwise_collator_pairs(monkeypatch):
    from representax.models.qwen3_vl import processing

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, checkpoint, **options):
            del checkpoint, options
            return cls()

        def apply_chat_template(self, conversations, **options):
            del options
            assert len(conversations) == 1
            return ["rendered pair"]

        def __call__(self, **options):
            assert options["images"] is None
            assert options["videos"] is None
            return {
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int32),
                "attention_mask": np.ones((1, 3), dtype=np.int32),
            }

    monkeypatch.setattr(
        processing,
        "import_module",
        lambda name: type("Transformers", (), {"Qwen3VLProcessor": FakeProcessor}),
    )
    processor = make_qwen3_vl_processor(
        "fixture",
        tiny_config(),
        mode="reranking",
        sequence_length_buckets=(8,),
        patch_count_buckets=(8,),
    )

    batch = processor(
        (("which harbor?", "a quiet harbor"),),
        route=Route.GENERIC,
    )

    assert batch.input_ids.shape == (1, 8)
    assert int(batch.attention_mask.sum()) == 3


def test_vision_layout_uses_qwen_merge_major_patch_order():
    layout = vision_layout([(1, 4, 4)], tiny_config().vision)
    np.testing.assert_array_equal(
        layout["vision_position_ids"],
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [2, 0],
            [2, 1],
            [3, 0],
            [3, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ],
    )


def test_video_mrope_splits_temporal_grid_into_frame_regions():
    positions = multimodal_position_ids(
        np.asarray([[1, 30, 30, 2, 30, 30, 3]], dtype=np.int32),
        np.ones((1, 7), dtype=np.int32),
        np.asarray([[0, 2, 2, 0, 2, 2, 0]], dtype=np.int32),
        (),
        ((2, 4, 2),),
        spatial_merge_size=2,
    )
    np.testing.assert_array_equal(
        positions[:, 0],
        [
            [0, 1, 1, 3, 4, 4, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 1, 3, 4, 4, 6],
        ],
    )
