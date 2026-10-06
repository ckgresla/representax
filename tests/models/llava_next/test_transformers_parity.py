"""Same-tensor LLaVA-NeXT parity against Transformers 5.6."""

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

from representax.models.llava_next import (
    LlavaNextBatch,
    LlavaNextCheckpointAdapter,
    image_pack_indices,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("llava_next")
    executable = os.environ.get(
        "REPRESENTAX_LLAVA_NEXT_TRANSFORMERS_PYTHON", sys.executable
    )
    subprocess.run(
        [
            executable,
            "-m",
            "tests.models.llava_next.transformers_oracle",
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
    adapter = LlavaNextCheckpointAdapter(rematerialization="full")
    model = adapter.load(
        directory,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        model_id="test/llava-next",
        revision="transformers-5.6.0",
    )
    sources, tile_valid = image_pack_indices(
        reference["image_sizes"], model.config, image_bucket=1, tile_bucket=2
    )
    visual_positions = np.flatnonzero(
        reference["input_ids"].reshape(-1) == model.config.image_token_id
    ).astype(np.int32)
    capacity = reference["input_ids"].size
    pack_indices = np.zeros((capacity,), dtype=np.int32)
    pack_valid = np.zeros((capacity,), dtype=bool)
    token_indices = np.zeros((capacity,), dtype=np.int32)
    pack_indices[: sources.size] = sources
    pack_valid[: sources.size] = True
    token_indices[: sources.size] = visual_positions
    batch = LlavaNextBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
        pixel_values=jnp.asarray(reference["pixel_values"]),
        tile_valid=jnp.asarray(tile_valid),
        pack_indices=jnp.asarray(pack_indices),
        pack_valid=jnp.asarray(pack_valid),
        visual_token_indices=jnp.asarray(token_indices),
    )
    return reference, adapter, model, batch


def test_forward_and_pixel_gradient_parity(oracle_checkpoint):
    reference, _, model, batch = _model_and_batch(oracle_checkpoint)
    assert batch.pixel_values is not None
    objective = jnp.linspace(-0.5, 0.5, batch.input_ids.size * 8).reshape(
        1, batch.input_ids.shape[1], 8
    )

    @eqx.filter_jit
    def outputs(candidate, values):
        def loss(pixels):
            changed = eqx.tree_at(lambda item: item.pixel_values, values, pixels)
            return jnp.sum(candidate.hidden_states(changed) * objective)

        return candidate.hidden_states(values), jax.grad(loss)(values.pixel_values)

    with jax.default_matmul_precision("highest"):
        hidden, pixel_gradient = outputs(model, batch)
    tolerance = NumericalTolerance(absolute=5e-5, relative=5e-5, cosine=0.999999)
    assert_numerically_equivalent(hidden, reference["hidden"], tolerance)
    assert_numerically_equivalent(
        pixel_gradient, reference["pixel_gradient"], tolerance
    )


def test_parameter_gradients_and_adamw_update_parity(oracle_checkpoint):
    reference, adapter, model, batch = _model_and_batch(oracle_checkpoint)
    objective = jnp.linspace(-0.5, 0.5, batch.input_ids.size * 8).reshape(
        1, batch.input_ids.shape[1], 8
    )
    with jax.default_matmul_precision("highest"):
        loss, gradients = eqx.filter_value_and_grad(
            lambda candidate: jnp.sum(candidate.hidden_states(batch) * objective)
        )(model)
    np.testing.assert_allclose(loss, reference["parameter_loss"], rtol=5e-5, atol=5e-5)
    state_gradients = adapter.state_dict(gradients)
    tolerance = NumericalTolerance(absolute=1e-3, relative=1e-5, cosine=0.99999)
    names = {
        name.removeprefix("parameter_gradient__")
        for name in reference.files
        if name.startswith("parameter_gradient__")
    }
    for name in names:
        expected = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected) <= 1e-7:
            assert numerical_result(state_gradients[name], expected).max_absolute <= (
                tolerance.absolute
            )
        else:
            assert_numerically_equivalent(state_gradients[name], expected, tolerance)

    optimizer = optax.adamw(
        learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01
    )
    parameters = eqx.filter(model, eqx.is_inexact_array)
    state = optimizer.init(parameters)
    updates, _ = optimizer.update(gradients, state, parameters)
    updated = adapter.state_dict(eqx.apply_updates(model, updates))
    for name in names:
        if np.linalg.norm(reference["parameter_gradient__" + name]) <= 1e-7:
            # Adam's first step normalizes even roundoff-scale gradients. The
            # two mathematically zero CLIP bias cotangents may therefore take
            # opposite milliscale steps despite forward/gradient parity.
            continue
        assert_numerically_equivalent(
            updated[name],
            reference["updated_parameter__" + name],
            NumericalTolerance(absolute=7e-6, relative=7e-5, cosine=0.999999),
        )


def test_native_export_reloads_in_transformers(oracle_checkpoint, tmp_path):
    reference, adapter, model, _ = _model_and_batch(oracle_checkpoint)
    export = adapter.save(model, tmp_path / "export")
    reloaded = adapter.load(export)
    for name, value in adapter.state_dict(model).items():
        np.testing.assert_array_equal(adapter.state_dict(reloaded)[name], value)
    output = tmp_path / "reload.npz"
    executable = os.environ.get(
        "REPRESENTAX_LLAVA_NEXT_TRANSFORMERS_PYTHON", sys.executable
    )
    subprocess.run(
        [
            executable,
            "-m",
            "tests.models.llava_next.transformers_reload",
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
