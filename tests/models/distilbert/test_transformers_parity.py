"""Pinned Transformers and Sentence Transformers parity for DistilBERT."""

from __future__ import annotations

import json
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

from representax.integrations import load_sentence_transformer
from representax.models.distilbert import (
    DistilBertBatch,
    DistilBertCheckpointAdapter,
)
from tests.models.acceptance import (
    NumericalTolerance,
    assert_numerically_equivalent,
    numerical_result,
)

pytestmark = pytest.mark.parity


def _python() -> str:
    return os.environ.get("REPRESENTAX_DISTILBERT_TRANSFORMERS_PYTHON", sys.executable)


@pytest.fixture(scope="module")
def oracle_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("distilbert-oracle")
    subprocess.run(
        [
            _python(),
            "-m",
            "tests.models.distilbert.transformers_oracle",
            "--output-directory",
            str(directory),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    return directory


def _model_and_batch(oracle_checkpoint):
    reference = np.load(oracle_checkpoint / "oracle.npz")
    adapter = DistilBertCheckpointAdapter(rematerialization="full")
    model = adapter.load(oracle_checkpoint)
    batch = DistilBertBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
    )
    return reference, adapter, model, batch


def test_forward_and_input_gradient_match_transformers(oracle_checkpoint) -> None:
    reference, _, model, batch = _model_and_batch(oracle_checkpoint)
    embedded = DistilBertBatch(
        inputs_embeds=jnp.asarray(reference["inputs_embeds"]),
        attention_mask=batch.attention_mask,
    )
    objective = jnp.linspace(-0.5, 0.5, model.metadata.output_dimension)

    def embedded_loss(values):
        inputs = DistilBertBatch(
            inputs_embeds=values,
            attention_mask=embedded.attention_mask,
        )
        return jnp.sum(model.hidden_states(inputs) * objective)

    with jax.default_matmul_precision("highest"):
        hidden = model.hidden_states(batch)
        embedded_hidden = model.hidden_states(embedded)
        input_gradient = jax.grad(embedded_loss)(embedded.inputs_embeds)
    tolerance = NumericalTolerance(absolute=2e-5, relative=2e-5, cosine=0.999999)
    for actual, name in (
        (hidden, "hidden"),
        (embedded_hidden, "embedded_hidden"),
        (input_gradient, "input_gradient"),
    ):
        assert_numerically_equivalent(actual, reference[name], tolerance)


def test_parameter_gradients_and_adamw_match_transformers(oracle_checkpoint) -> None:
    reference, adapter, model, batch = _model_and_batch(oracle_checkpoint)
    objective = jnp.linspace(-0.5, 0.5, model.metadata.output_dimension)
    with jax.default_matmul_precision("highest"):
        loss, gradients = eqx.filter_value_and_grad(
            lambda candidate: jnp.sum(candidate.hidden_states(batch) * objective)
        )(model)
    np.testing.assert_allclose(loss, reference["parameter_loss"], rtol=2e-6, atol=2e-6)
    tolerance = NumericalTolerance(absolute=3e-5, relative=3e-5, cosine=0.99999)
    for name, actual in adapter.state_dict(gradients).items():
        expected = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected) <= 1e-7:
            assert numerical_result(actual, expected).max_absolute <= tolerance.absolute
        else:
            assert_numerically_equivalent(actual, expected, tolerance)

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
    for name, actual in adapter.state_dict(updated).items():
        expected = reference["updated_parameter__" + name]
        expected_gradient = reference["parameter_gradient__" + name]
        if np.linalg.norm(expected_gradient) <= 1e-7:
            # This branch is an analytically null update. Comparing two nearly
            # cancelling FP32 reductions is better expressed as an absolute
            # bound than a relative or cosine requirement.
            assert numerical_result(actual, expected).max_absolute <= 7e-6
        else:
            assert_numerically_equivalent(
                actual,
                expected,
                NumericalTolerance(
                    absolute=2e-6,
                    relative=2e-5,
                    cosine=0.999999,
                ),
            )


def test_native_export_reloads_in_transformers(oracle_checkpoint, tmp_path) -> None:
    reference, adapter, model, _ = _model_and_batch(oracle_checkpoint)
    export = adapter.save(model, tmp_path / "export")
    output = tmp_path / "reload.npz"
    subprocess.run(
        [
            _python(),
            "-m",
            "tests.models.distilbert.transformers_reload",
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


def test_real_multilingual_clip_projection_matches_sentence_transformers(
    tmp_path,
) -> None:
    checkpoint_value = os.environ.get("REPRESENTAX_CLIP_MULTILINGUAL_CHECKPOINT")
    if checkpoint_value is None:
        pytest.skip("set REPRESENTAX_CLIP_MULTILINGUAL_CHECKPOINT for real parity")
    checkpoint = Path(checkpoint_value)
    texts = (
        "A red geometric shape.",
        "Ein rotes geometrisches Objekt.",
        "Un objet géométrique rouge.",
    )
    texts_path = tmp_path / "texts.json"
    upstream_path = tmp_path / "upstream.npy"
    metadata_path = tmp_path / "upstream.json"
    texts_path.write_text(json.dumps(texts))
    subprocess.run(
        [
            _python(),
            "-m",
            "tests.models.sentence_transformers.transformers_oracle",
            "--checkpoint",
            str(checkpoint),
            "--texts",
            str(texts_path),
            "--output",
            str(upstream_path),
            "--metadata",
            str(metadata_path),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    model = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
        sequence_length_buckets=(32,),
    )
    with jax.default_matmul_precision("highest"):
        actual = model.embed(texts, batch_size=len(texts))
    result = assert_numerically_equivalent(
        actual,
        np.load(upstream_path),
        NumericalTolerance(absolute=2e-6, relative=2e-6, cosine=0.999999),
    )
    print(result)

    export = model.model.save_to_hf(
        tmp_path / "sentence-transformer-export",
        source_checkpoint=checkpoint,
    )
    source_config = json.loads((checkpoint / "config.json").read_text())
    exported_config = json.loads((export / "config.json").read_text())
    assert (
        exported_config["transformers_version"] == source_config["transformers_version"]
    )
    assert "_name_or_path" not in exported_config
    native_reload = load_sentence_transformer(
        export,
        local_files_only=True,
        sequence_length_buckets=(32,),
    )
    source_batch = model.preprocess(texts)
    reloaded_batch = native_reload.preprocess(texts)
    np.testing.assert_array_equal(
        reloaded_batch.input_ids,
        source_batch.input_ids,
    )
    np.testing.assert_array_equal(
        reloaded_batch.attention_mask,
        source_batch.attention_mask,
    )
    source_state = DistilBertCheckpointAdapter().state_dict(model.model.backbone)
    reloaded_state = DistilBertCheckpointAdapter().state_dict(
        native_reload.model.backbone
    )
    assert source_state.keys() == reloaded_state.keys()
    for name in source_state:
        np.testing.assert_array_equal(reloaded_state[name], source_state[name])
    with jax.default_matmul_precision("highest"):
        reloaded_native = native_reload.embed(texts, batch_size=len(texts))
    assert_numerically_equivalent(
        reloaded_native,
        actual,
        NumericalTolerance(absolute=2e-6, relative=2e-6, cosine=0.999999),
    )
    reloaded_upstream_path = tmp_path / "reloaded-upstream.npy"
    subprocess.run(
        [
            _python(),
            "-m",
            "tests.models.sentence_transformers.transformers_oracle",
            "--checkpoint",
            str(export),
            "--texts",
            str(texts_path),
            "--output",
            str(reloaded_upstream_path),
            "--metadata",
            str(tmp_path / "reloaded-upstream.json"),
        ],
        check=True,
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": str(Path.cwd())},
    )
    assert_numerically_equivalent(
        reloaded_native,
        np.load(reloaded_upstream_path),
        NumericalTolerance(absolute=2e-6, relative=2e-6, cosine=0.999999),
    )
