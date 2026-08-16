"""Pinned numerical parity for the native MPNet encoder."""

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
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

from representax.models.mpnet import MPNetBatch, MPNetCheckpointAdapter

pytestmark = pytest.mark.parity


def _upstream_python() -> str:
    executable = os.environ.get("REPRESENTAX_MPNET_TRANSFORMERS_PYTHON")
    if executable is None:
        pytest.skip("set REPRESENTAX_MPNET_TRANSFORMERS_PYTHON for MPNet parity")
    return executable


def _generate_oracle(directory: Path) -> None:
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.mpnet.transformers_oracle",
            "--output-directory",
            str(directory),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mpnet-oracle")
    _generate_oracle(directory)
    return directory


def test_pinned_transformers_forward_pooler_and_input_gradient_parity(
    oracle_checkpoint,
):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    model = MPNetCheckpointAdapter(rematerialization="full").load(oracle_checkpoint)
    batch = MPNetBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
    )
    embedded_batch = MPNetBatch(
        inputs_embeds=jnp.asarray(reference["inputs_embeds"]),
        attention_mask=batch.attention_mask,
    )
    objective = jnp.linspace(-0.5, 0.5, model.metadata.output_dimension)

    @eqx.filter_jit
    def parity_outputs(candidate, token_batch, embeddings_batch, vector):
        hidden = candidate.hidden_states(token_batch)
        pooler = candidate.pooler_output(token_batch)
        embedded_hidden = candidate.hidden_states(embeddings_batch)

        def embedded_objective(values):
            values_batch = MPNetBatch(
                inputs_embeds=values,
                attention_mask=embeddings_batch.attention_mask,
            )
            return jnp.sum(candidate.hidden_states(values_batch) * vector)

        gradient = jax.grad(embedded_objective)(embeddings_batch.inputs_embeds)
        return hidden, pooler, embedded_hidden, gradient

    with jax.default_matmul_precision("highest"):
        actual = parity_outputs(model, batch, embedded_batch, objective)
    tolerance = NumericalTolerance(
        absolute=2e-5,
        relative=2e-5,
        cosine=0.999999,
    )
    results = {}
    for value, name in zip(
        actual,
        ("hidden", "pooler", "embedded_hidden", "input_gradient"),
        strict=True,
    ):
        result = assert_numerically_equivalent(
            np.asarray(value), reference[name], tolerance
        )
        results[name] = {
            "max_absolute": result.max_absolute,
            "relative_l2": result.relative_l2,
            "cosine": result.cosine,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


def test_pinned_parameter_gradients_and_adamw_update_parity(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = MPNetCheckpointAdapter(rematerialization="full")
    model = adapter.load(oracle_checkpoint)
    batch = MPNetBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
    )
    objective_vector = jnp.linspace(-0.5, 0.5, model.metadata.output_dimension)

    def objective(candidate):
        hidden = candidate.hidden_states(batch)
        pooler = candidate.tower.pool(hidden)
        return jnp.sum(hidden * objective_vector) + jnp.sum(pooler * objective_vector)

    with jax.default_matmul_precision("highest"):
        loss, gradients = eqx.filter_value_and_grad(objective)(model)
    np.testing.assert_allclose(
        loss,
        reference["parameter_loss"],
        rtol=2e-6,
        atol=2e-6,
    )
    gradient_state = adapter.state_dict(gradients)
    gradient_results = {}
    tolerance = NumericalTolerance(
        absolute=2e-4,
        relative=1e-6,
        cosine=0.999999,
    )
    for name, value in gradient_state.items():
        expected = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected) <= 1e-7:
            result = numerical_result(np.asarray(value), expected)
            assert result.max_absolute <= tolerance.absolute, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(value),
                expected,
                tolerance,
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
    for name, value in adapter.state_dict(updated).items():
        expected = reference["updated_parameter__" + name]
        expected_gradient = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected_gradient) <= 1e-7:
            result = numerical_result(np.asarray(value), expected)
            assert result.max_absolute <= 5e-6, (name, result)
        else:
            result = assert_numerically_equivalent(
                np.asarray(value),
                expected,
                NumericalTolerance(
                    absolute=2e-6,
                    relative=2e-5,
                    cosine=0.999999,
                ),
            )
        update_results[name] = result.max_absolute
    print(
        json.dumps(
            {
                "loss_absolute": abs(float(loss) - float(reference["parameter_loss"])),
                "maximum_parameter_gradient_absolute": max(gradient_results.values()),
                "maximum_adamw_parameter_absolute": max(update_results.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


def test_native_export_reloads_in_pinned_transformers(oracle_checkpoint, tmp_path):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = MPNetCheckpointAdapter(rematerialization="full")
    model = adapter.load(oracle_checkpoint)
    export = adapter.save(model, tmp_path / "export")
    output_path = tmp_path / "reload.npz"
    subprocess.run(
        [
            _upstream_python(),
            "-m",
            "tests.models.mpnet.transformers_reload",
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
    tolerance = NumericalTolerance(
        absolute=1e-6,
        relative=1e-6,
        cosine=0.9999999,
    )
    assert_numerically_equivalent(reloaded["hidden"], reference["hidden"], tolerance)
    assert_numerically_equivalent(reloaded["pooler"], reference["pooler"], tolerance)
