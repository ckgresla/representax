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
    Qwen3VLCheckpointAdapter,
    batch_from_processor_output,
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
