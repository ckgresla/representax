"""Pinned same-tensor Qwen3-VL parity against Transformers 5.3."""

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

from representax.models.qwen3_vl import (
    EAGER_EMBED_V1_MODEL_ID,
    EAGER_EMBED_V1_REVISION,
    Qwen3VLCheckpointAdapter,
    Qwen3VLConfig,
    batch_from_processor_output,
    make_qwen3_vl_processor,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


def _upstream_python() -> str:
    executable = os.environ.get("REPRESENTAX_QWEN3_VL_TRANSFORMERS_PYTHON")
    if executable is None:
        pytest.skip("set REPRESENTAX_QWEN3_VL_TRANSFORMERS_PYTHON for parity")
    assert executable is not None
    return executable


def _eager_checkpoint() -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            EAGER_EMBED_V1_MODEL_ID,
            revision=EAGER_EMBED_V1_REVISION,
            allow_patterns=[
                "config.json",
                "chat_template.jinja",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "video_preprocessor_config.json",
                "special_tokens_map.json",
                "merges.txt",
                "vocab.json",
            ],
        )
    )


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("qwen3-vl-oracle")
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen3_vl.transformers_oracle",
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
            "mm_token_type_ids": reference["mm_token_type_ids"],
            "pixel_values": reference["pixel_values"],
            "image_grid_thw": reference["image_grid_thw"],
        },
        model.config,
        sequence_length_buckets=(5,),
        patch_count_buckets=(4,),
    )


def test_forward_and_pixel_gradient_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    model = Qwen3VLCheckpointAdapter(rematerialization="full").load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/qwen3-vl",
        revision="transformers-5.3.0",
    )
    batch = _batch(reference, model)
    assert batch.pixel_values is not None
    vector = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)

    @eqx.filter_jit
    def parity_outputs(candidate, values, objective):
        def loss(pixel_values):
            replaced = eqx.tree_at(
                lambda item: item.pixel_values,
                values,
                pixel_values,
            )
            return jnp.sum(candidate.hidden_states(replaced) * objective)

        return candidate.hidden_states(values), jax.grad(loss)(values.pixel_values)

    with jax.default_matmul_precision("highest"):
        hidden, pixel_gradient = parity_outputs(model, batch, vector)
    tolerance = NumericalTolerance(absolute=3e-5, relative=3e-5, cosine=0.999999)
    results = {}
    for name, actual in (
        ("hidden", hidden),
        ("pixel_gradient", pixel_gradient),
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


def test_parameter_gradients_and_adamw_update_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = Qwen3VLCheckpointAdapter(rematerialization="full")
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
    np.testing.assert_allclose(loss, reference["parameter_loss"], rtol=3e-5, atol=3e-5)
    gradient_results = {}
    tolerance = NumericalTolerance(absolute=5e-5, relative=5e-5, cosine=0.99999)
    for name, actual in adapter.state_dict(gradients).items():
        expected = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected) <= 1e-8:
            result = numerical_result(np.asarray(actual), expected)
            assert result.max_absolute <= tolerance.absolute, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(actual), expected, tolerance
            )
        gradient_results[name] = result.max_absolute

    optimizer = optax.adamw(
        learning_rate=1e-3,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.01,
    )
    parameters = eqx.filter(model, eqx.is_inexact_array)
    optimizer_state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, optimizer_state, parameters)
    updated = eqx.apply_updates(model, updates)
    update_results = {}
    for name, actual in adapter.state_dict(updated).items():
        expected = reference["updated_parameter__" + name]
        expected_gradient = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected_gradient) <= 1e-8:
            result = numerical_result(np.asarray(actual), expected)
            assert result.max_absolute <= 5e-6, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(actual),
                expected,
                NumericalTolerance(
                    absolute=3e-6,
                    relative=3e-5,
                    cosine=0.999999,
                ),
            )
        update_results[name] = result.max_absolute
    print(
        json.dumps(
            {
                "maximum_gradient_absolute": max(gradient_results.values()),
                "maximum_adamw_parameter_absolute": max(update_results.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_native_export_reloads_in_pinned_transformers(oracle_checkpoint, tmp_path):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = Qwen3VLCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        oracle_checkpoint,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    export = adapter.save(model, tmp_path / "export")
    reloaded_native = adapter.load(
        export,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    for name, value in adapter.state_dict(model).items():
        np.testing.assert_array_equal(adapter.state_dict(reloaded_native)[name], value)
    output = tmp_path / "reload.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen3_vl.transformers_reload",
            "--checkpoint",
            str(export),
            "--inputs",
            str(oracle_checkpoint / "oracle.npz"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    assert_numerically_equivalent(
        np.load(output)["hidden"],
        reference["hidden"],
        NumericalTolerance(absolute=1e-6, relative=1e-6, cosine=0.9999999),
    )


def test_eager_embed_preprocessing_matches_pinned_transformers(tmp_path):
    checkpoint = _eager_checkpoint()
    output = tmp_path / "eager-preprocessing.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.qwen3_vl.eager_processor_oracle",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    reference = np.load(output)
    config = Qwen3VLConfig.from_hf_config(
        json.loads((checkpoint / "config.json").read_text())
    )
    processor = make_qwen3_vl_processor(
        checkpoint,
        config,
        mode="eager_embedding",
        sequence_length_buckets=(512,),
        patch_count_buckets=(4096,),
    )
    from PIL import Image

    batch = processor(
        [
            {
                "image": Image.fromarray(reference["source_pixels"]),
                "text": "A deterministic document page.",
            }
        ]
    )
    token_count = reference["input_ids"].shape[1]
    np.testing.assert_array_equal(
        np.asarray(batch.input_ids)[:, -token_count:],
        reference["input_ids"],
    )
    np.testing.assert_array_equal(
        np.asarray(batch.attention_mask)[:, -token_count:],
        reference["attention_mask"],
    )
    assert batch.pixel_values is not None
    assert batch.patch_valid is not None
    patch_count = reference["pixel_values"].shape[0]
    np.testing.assert_array_equal(
        np.asarray(batch.pixel_values)[:patch_count],
        reference["pixel_values"],
    )
    assert int(np.asarray(batch.patch_valid).sum()) == patch_count
