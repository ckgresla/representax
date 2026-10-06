"""Pinned same-tensor CLIP parity against Transformers 5.6."""

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

from representax.core import Route
from representax.models.clip import CLIPBatch, CLIPCheckpointAdapter
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


def _upstream_python() -> str:
    executable = os.environ.get("REPRESENTAX_CLIP_TRANSFORMERS_PYTHON")
    if executable is None:
        pytest.skip("set REPRESENTAX_CLIP_TRANSFORMERS_PYTHON for CLIP parity")
    assert executable is not None
    return executable


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("clip-oracle")
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.clip.transformers_oracle",
            "--output-directory",
            str(directory),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    return directory


def _batch(reference) -> CLIPBatch:
    return CLIPBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
        text_valid=jnp.ones((2,), dtype=bool),
        pixel_values=jnp.asarray(reference["pixel_values"]),
        image_valid=jnp.ones((2,), dtype=bool),
    )


def test_forward_and_pixel_gradient_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    model = CLIPCheckpointAdapter(rematerialization="full", normalize_output=True).load(
        oracle_checkpoint
    )
    batch = _batch(reference)
    objective = jnp.linspace(-0.5, 0.5, model.config.projection_dimension)

    @eqx.filter_jit
    def outputs(candidate, values):
        def loss(pixel_values):
            replaced = eqx.tree_at(
                lambda item: item.pixel_values,
                values,
                pixel_values,
            )
            return jnp.sum(candidate.encode(replaced, route=Route.GENERIC) * objective)

        assert values.pixel_values is not None
        return (
            candidate.text_features(values.input_ids, values.attention_mask),
            candidate.image_features(values.pixel_values),
            candidate.encode(values, route=Route.GENERIC),
            jax.grad(loss)(values.pixel_values),
        )

    with jax.default_matmul_precision("highest"):
        raw_text, raw_image, composed, pixel_gradient = outputs(model, batch)
    text = raw_text / jnp.linalg.norm(raw_text, axis=-1, keepdims=True)
    image = raw_image / jnp.linalg.norm(raw_image, axis=-1, keepdims=True)
    results = {}
    for name, actual in (
        ("text", text),
        ("image", image),
        ("composed", composed),
        ("pixel_gradient", pixel_gradient),
    ):
        result = assert_numerically_equivalent(
            np.asarray(actual),
            reference[name],
            NumericalTolerance(absolute=3e-5, relative=3e-5, cosine=0.99999),
        )
        results[name] = result.max_absolute
    print(json.dumps(results, indent=2, sort_keys=True))


def test_parameter_gradients_and_three_adamw_updates(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = CLIPCheckpointAdapter(rematerialization="full", normalize_output=True)
    model = adapter.load(oracle_checkpoint)
    batch = _batch(reference)
    objective = jnp.linspace(-0.5, 0.5, model.config.projection_dimension)

    def loss(candidate):
        return jnp.sum(candidate.encode(batch, route=Route.GENERIC) * objective)

    with jax.default_matmul_precision("highest"):
        initial_loss, gradients = eqx.filter_value_and_grad(loss)(model)
    np.testing.assert_allclose(initial_loss, reference["parameter_loss"], atol=3e-5)
    gradient_results = {}
    state = adapter.state_dict(gradients)
    for reference_name in reference.files:
        prefix = "parameter_gradient__"
        if not reference_name.startswith(prefix):
            continue
        name = reference_name.removeprefix(prefix)
        expected = reference[reference_name]
        if np.linalg.norm(expected) <= 1e-8:
            result = numerical_result(np.asarray(state[name]), expected)
            assert result.max_absolute <= 6e-5, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(state[name]),
                expected,
                NumericalTolerance(absolute=6e-5, relative=8e-5, cosine=0.9999),
            )
        gradient_results[name] = result.max_absolute

    optimizer = optax.adamw(1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(candidate, optimizer_state):
        value, gradients = eqx.filter_value_and_grad(loss)(candidate)
        updates, optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            candidate,
        )
        return value, eqx.apply_updates(candidate, updates), optimizer_state

    losses = []
    with jax.default_matmul_precision("highest"):
        for _ in range(3):
            value, model, optimizer_state = update(model, optimizer_state)
            losses.append(float(value))
    np.testing.assert_allclose(
        losses, reference["training_losses"], rtol=2e-4, atol=2e-4
    )
    updated = adapter.state_dict(model)
    update_results = {}
    for reference_name in reference.files:
        prefix = "updated_parameter__"
        if not reference_name.startswith(prefix):
            continue
        name = reference_name.removeprefix(prefix)
        if np.linalg.norm(reference["parameter_gradient__" + name]) <= 1e-7:
            continue
        result = assert_numerically_equivalent(
            np.asarray(updated[name]),
            reference[reference_name],
            NumericalTolerance(absolute=2e-5, relative=3e-3, cosine=0.99999),
        )
        update_results[name] = result.max_absolute
    print(
        json.dumps(
            {
                "losses": losses,
                "maximum_gradient_absolute": max(gradient_results.values()),
                "maximum_update_absolute": max(update_results.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_native_export_reloads_in_transformers(oracle_checkpoint, tmp_path):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = CLIPCheckpointAdapter(rematerialization="full", normalize_output=True)
    model = adapter.load(oracle_checkpoint)
    export = adapter.save(model, tmp_path / "export")
    output = tmp_path / "reload.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.clip.transformers_reload",
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
    reloaded = np.load(output)
    assert_numerically_equivalent(
        reloaded["composed"],
        reference["composed"],
        NumericalTolerance(absolute=1e-6, relative=1e-6, cosine=0.9999999),
    )
