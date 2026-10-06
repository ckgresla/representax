"""Same-tensor Qwen2/Qwen2.5-VL parity against Transformers 5.6."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models.qwen2_vl import (
    Qwen2VLCheckpointAdapter,
    batch_from_processor_output,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module", params=("qwen2_vl", "qwen2_5_vl"))
def oracle_checkpoint(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(request.param)
    executable = os.environ.get(
        "REPRESENTAX_QWEN2_VL_TRANSFORMERS_PYTHON", sys.executable
    )
    subprocess.run(
        [
            executable,
            "-m",
            "tests.models.qwen2_vl.transformers_oracle",
            "--generation",
            request.param,
            "--output-directory",
            str(directory),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    return directory


def _model_and_batch(directory):
    reference = np.load(directory / "oracle.npz")
    adapter = Qwen2VLCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        directory,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/qwen2-vl",
        revision="transformers-5.6.0",
    )
    batch = batch_from_processor_output(
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
        padding_side="left",
    )
    return reference, adapter, model, batch


def test_forward_and_pixel_gradient_parity(oracle_checkpoint):
    reference, _, model, batch = _model_and_batch(oracle_checkpoint)
    assert batch.pixel_values is not None
    objective = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)

    @eqx.filter_jit
    def outputs(candidate, values):
        def loss(pixels):
            changed = eqx.tree_at(lambda item: item.pixel_values, values, pixels)
            return jnp.sum(candidate.hidden_states(changed) * objective)

        return candidate.hidden_states(values), jax.grad(loss)(values.pixel_values)

    with jax.default_matmul_precision("highest"):
        hidden, pixel_gradient = outputs(model, batch)
    tolerance = NumericalTolerance(absolute=4e-5, relative=4e-5, cosine=0.999999)
    assert_numerically_equivalent(hidden, reference["hidden"], tolerance)
    assert_numerically_equivalent(
        pixel_gradient, reference["pixel_gradient"], tolerance
    )


def test_parameter_gradients_and_adamw_update_parity(oracle_checkpoint):
    reference, adapter, model, batch = _model_and_batch(oracle_checkpoint)
    objective = jnp.linspace(-0.5, 0.5, model.config.text.hidden_size)
    with jax.default_matmul_precision("highest"):
        loss, gradients = eqx.filter_value_and_grad(
            lambda candidate: jnp.sum(candidate.hidden_states(batch) * objective)
        )(model)
    np.testing.assert_allclose(loss, reference["parameter_loss"], rtol=4e-5, atol=4e-5)
    tolerance = NumericalTolerance(absolute=7e-5, relative=7e-5, cosine=0.99998)
    for name, actual in adapter.state_dict(gradients).items():
        expected = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected) <= 1e-8:
            assert numerical_result(actual, expected).max_absolute <= tolerance.absolute
        else:
            assert_numerically_equivalent(actual, expected, tolerance)

    optimizer = optax.adamw(
        learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01
    )
    parameters = eqx.filter(model, eqx.is_inexact_array)
    state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, state, parameters)
    updated = eqx.apply_updates(model, updates)
    for name, actual in adapter.state_dict(updated).items():
        expected = reference["updated_parameter__" + name]
        assert_numerically_equivalent(
            actual,
            expected,
            NumericalTolerance(absolute=5e-6, relative=5e-5, cosine=0.999999),
        )


def test_native_export_reloads_in_transformers(oracle_checkpoint, tmp_path):
    reference, adapter, model, batch = _model_and_batch(oracle_checkpoint)
    generation = model.config.generation
    export = adapter.save(model, tmp_path / generation)
    reloaded = adapter.load(
        export,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    for name, value in adapter.state_dict(model).items():
        np.testing.assert_array_equal(adapter.state_dict(reloaded)[name], value)
    output = tmp_path / f"{generation}-reload.npz"
    executable = os.environ.get(
        "REPRESENTAX_QWEN2_VL_TRANSFORMERS_PYTHON", sys.executable
    )
    subprocess.run(
        [
            executable,
            "-m",
            "tests.models.qwen2_vl.transformers_reload",
            "--generation",
            generation,
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
    np.testing.assert_allclose(
        np.load(output)["hidden"], reference["hidden"], rtol=1e-6, atol=1e-6
    )
