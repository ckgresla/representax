"""Pinned same-tensor Qwen2.5-Omni parity against Transformers 5.6."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.integrations.huggingface import load_hf_config
from representax.models.qwen2_5_omni import (
    Qwen2_5OmniCheckpointAdapter,
    Qwen2_5OmniConfig,
    batch_from_processor_output,
    make_qwen2_5_omni_processor,
    qwen2_5_omni_weight_names,
    vision_layout,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


def _upstream_python() -> str:
    executable = os.environ.get("REPRESENTAX_QWEN2_5_OMNI_TRANSFORMERS_PYTHON")
    if executable is None:
        pytest.skip("set REPRESENTAX_QWEN2_5_OMNI_TRANSFORMERS_PYTHON for parity")
    assert executable is not None
    return executable


def _real_checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_QWEN2_5_OMNI_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_QWEN2_5_OMNI_CHECKPOINT for real preprocessing")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen2.5-omni-oracle")
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen2_5_omni.transformers_oracle",
            "--output-directory",
            str(directory),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    return directory


def _batch(reference, model):
    return batch_from_processor_output(
        {
            "input_ids": reference["input_ids"],
            "attention_mask": reference["attention_mask"],
            "pixel_values": reference["pixel_values"],
            "image_grid_thw": reference["image_grid_thw"],
            "input_features": reference["input_features"],
            "feature_attention_mask": reference["feature_attention_mask"],
        },
        model.config,
        sequence_length_buckets=(12,),
        patch_count_buckets=(16,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(2,),
    )


def test_forward_and_media_gradient_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    model = Qwen2_5OmniCheckpointAdapter(rematerialization="full").load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/qwen2.5-omni",
        revision="transformers-5.6.0",
    )
    batch = _batch(reference, model)
    np.testing.assert_array_equal(batch.position_ids, reference["position_ids"])
    assert batch.pixel_values is not None
    assert batch.input_features is not None
    vector = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)

    @eqx.filter_jit
    def parity_outputs(candidate, values, objective):
        def loss(pixel_values, input_features):
            replaced = eqx.tree_at(
                lambda item: (item.pixel_values, item.input_features),
                values,
                (pixel_values, input_features),
            )
            return jnp.sum(candidate.hidden_states(replaced) * objective)

        gradients = jax.grad(loss, argnums=(0, 1))(
            values.pixel_values, values.input_features
        )
        return candidate.hidden_states(values), gradients

    with jax.default_matmul_precision("highest"):
        hidden, (pixel_gradient, audio_gradient) = parity_outputs(model, batch, vector)
    tolerance = NumericalTolerance(absolute=4e-5, relative=4e-5, cosine=0.99999)
    results = {}
    for name, actual in (
        ("hidden", hidden),
        ("pixel_gradient", pixel_gradient),
        ("audio_gradient", audio_gradient),
    ):
        result = assert_numerically_equivalent(
            np.asarray(actual), reference[name], tolerance
        )
        results[name] = {
            "max_absolute": result.max_absolute,
            "relative_l2": result.relative_l2,
            "cosine": result.cosine,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


def test_native_tower_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    model = Qwen2_5OmniCheckpointAdapter(rematerialization="full").load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    batch = _batch(reference, model)
    assert batch.pixel_values is not None
    assert batch.patch_valid is not None
    assert batch.vision_full_segment_ids is not None
    assert batch.vision_window_segment_ids is not None
    assert batch.vision_position_ids is not None
    assert batch.reverse_merged_indices is not None
    assert batch.input_features is not None
    assert batch.audio_feature_valid is not None
    assert batch.audio_after_cnn_valid is not None
    assert batch.audio_pool_indices is not None
    assert batch.audio_token_valid is not None

    with jax.default_matmul_precision("highest"):
        image_features = model.vision(
            batch.pixel_values,
            batch.patch_valid,
            batch.vision_full_segment_ids,
            batch.vision_window_segment_ids,
            batch.vision_position_ids,
            batch.reverse_merged_indices,
            compute_dtype=jnp.float32,
            attention_implementation="xla",
            rematerialization="full",
        )
        audio_embeddings = model.audio(
            batch.input_features,
            batch.audio_feature_valid,
            batch.audio_after_cnn_valid,
            batch.audio_pool_indices,
            batch.audio_token_valid,
            compute_dtype=jnp.float32,
            attention_implementation="xla",
            rematerialization="full",
        )
    tolerance = NumericalTolerance(absolute=4e-5, relative=4e-5, cosine=0.99999)
    for name, actual in (
        ("image_features", image_features),
        ("audio_embeddings", audio_embeddings),
    ):
        result = numerical_result(np.asarray(actual), reference[name])
        print(name, result)
        assert_numerically_equivalent(np.asarray(actual), reference[name], tolerance)


def test_parameter_gradient_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = Qwen2_5OmniCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    batch = _batch(reference, model)
    vector = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)

    def objective(candidate):
        return jnp.sum(candidate.hidden_states(batch) * vector)

    with jax.default_matmul_precision("highest"):
        loss, gradients = eqx.filter_value_and_grad(objective)(model)
    np.testing.assert_allclose(loss, reference["parameter_loss"], rtol=4e-5, atol=4e-5)
    tolerance = NumericalTolerance(absolute=8e-5, relative=8e-5, cosine=0.9999)
    results = {}
    native_names = qwen2_5_omni_weight_names(model.config)
    for name, actual in adapter.state_dict(gradients).items():
        if name not in native_names:
            continue
        reference_name = "parameter_gradient__" + name
        if reference_name not in reference.files:
            continue
        expected = reference[reference_name]
        if np.linalg.norm(expected) <= 1e-8:
            result = numerical_result(np.asarray(actual), expected)
            assert result.max_absolute <= tolerance.absolute, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(actual), expected, tolerance
            )
        results[name] = result.max_absolute
    print(json.dumps({"maximum_absolute": max(results.values())}, indent=2))


def test_three_adamw_updates_match_transformers(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = Qwen2_5OmniCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    batch = _batch(reference, model)
    vector = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)
    optimizer = optax.adamw(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )
    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer_state = optimizer.init(parameters)
    losses = []

    @eqx.filter_jit
    def update(candidate, state):
        def objective(value):
            return jnp.sum(value.hidden_states(batch) * vector)

        loss, gradients = eqx.filter_value_and_grad(objective)(candidate)
        updates, state = optimizer.update(gradients, state, candidate)
        return loss, eqx.apply_updates(candidate, updates), state

    with jax.default_matmul_precision("highest"):
        for _ in range(3):
            loss, model, optimizer_state = update(model, optimizer_state)
            losses.append(float(loss))
    np.testing.assert_allclose(
        losses,
        reference["training_losses"],
        rtol=1e-4,
        atol=1e-4,
    )
    state = adapter.state_dict(model)
    results = {}
    for reference_name in reference.files:
        prefix = "updated_parameter__"
        if not reference_name.startswith(prefix):
            continue
        name = reference_name.removeprefix(prefix)
        actual = state[name]
        expected = reference[reference_name]
        result = assert_numerically_equivalent(
            np.asarray(actual),
            expected,
            NumericalTolerance(
                absolute=2e-5,
                relative=3e-3,
                cosine=0.99999,
            ),
        )
        results[name] = result.max_absolute
    print(
        json.dumps(
            {
                "losses": losses,
                "maximum_parameter_absolute": max(results.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_native_export_reloads_in_pinned_transformers(oracle_checkpoint, tmp_path):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = Qwen2_5OmniCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    export = adapter.save(model, tmp_path / "export")
    output_path = tmp_path / "reload.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen2_5_omni.transformers_reload",
            "--checkpoint",
            str(export),
            "--inputs",
            str(oracle_checkpoint / "oracle.npz"),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    reloaded = np.load(output_path)
    assert_numerically_equivalent(
        reloaded["hidden"],
        reference["hidden"],
        NumericalTolerance(absolute=1e-6, relative=1e-6, cosine=0.9999999),
    )


def test_real_multimodal_preprocessing_matches_transformers(tmp_path):
    checkpoint = _real_checkpoint()
    oracle_path = tmp_path / "preprocessing.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen2_5_omni.transformers_preprocessing_oracle",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(oracle_path),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    reference = np.load(oracle_path)
    config = Qwen2_5OmniConfig.from_hf_config(load_hf_config(checkpoint))
    processor = make_qwen2_5_omni_processor(
        checkpoint,
        config,
        sequence_length_buckets=(512,),
        patch_count_buckets=(4096,),
        audio_chunk_count_buckets=(256,),
        audio_token_count_buckets=(4096,),
    )
    from PIL import Image

    pixels = (np.arange(56 * 56 * 3, dtype=np.uint32).reshape(56, 56, 3) % 251).astype(
        np.uint8
    )
    audio = np.sin(np.linspace(0, 200, 16_000, dtype=np.float32))
    video = np.stack([np.roll(pixels, index, axis=1) for index in range(4)])
    batch = processor(
        [
            {
                "image": Image.fromarray(pixels),
                "audio": audio,
                "video": {"frames": video, "fps": 2.0},
                "text": "A deterministic multimodal sample.",
            }
        ]
    )

    valid_tokens = int(np.asarray(batch.attention_mask).sum())
    np.testing.assert_array_equal(
        np.asarray(batch.input_ids)[:, :valid_tokens], reference["input_ids"]
    )
    assert batch.pixel_values is not None
    assert batch.patch_valid is not None
    grids = [
        *reference["image_grid_thw"].tolist(),
        *reference["video_grid_thw"].tolist(),
    ]
    layout = vision_layout(grids, config.vision, patch_bucket=4096)
    raw_pixels = np.empty_like(np.asarray(batch.pixel_values))
    raw_pixels[layout["patch_order"]] = np.asarray(batch.pixel_values)
    expected_pixels = np.concatenate(
        (reference["pixel_values"], reference["pixel_values_videos"])
    )
    actual_pixels = raw_pixels[: len(expected_pixels)]
    pixel_result = numerical_result(actual_pixels, expected_pixels)
    assert pixel_result.max_absolute < 0.016
    assert pixel_result.relative_l2 < 6e-5

    assert batch.input_features is not None
    assert batch.audio_feature_valid is not None
    chunks = np.asarray(batch.input_features)
    valid = np.asarray(batch.audio_feature_valid)
    packed_audio = np.concatenate(
        [
            chunks[index, :, : int(mask.sum())]
            for index, mask in enumerate(valid)
            if mask.any()
        ],
        axis=1,
    )
    np.testing.assert_array_equal(
        packed_audio,
        reference["input_features"][0, :, : packed_audio.shape[1]],
    )
    print(
        json.dumps(
            {
                "valid_tokens": valid_tokens,
                "valid_patches": int(np.asarray(batch.patch_valid).sum()),
                "pixel_relative_l2": pixel_result.relative_l2,
                "valid_audio_frames": int(valid.sum()),
            },
            indent=2,
            sort_keys=True,
        )
    )
