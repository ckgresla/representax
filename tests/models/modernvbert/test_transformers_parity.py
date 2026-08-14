"""Pinned numerical parity against Transformers ModernVBERT."""

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.core import Route
from representax.models.modernvbert import (
    ModernVBERTBatch,
    ModernVBERTCheckpointAdapter,
    ModernVBERTTextBatch,
    ModernVBERTTextCheckpointAdapter,
)
from tests.models.acceptance import NumericalTolerance, assert_numerically_equivalent

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    "REPRESENTAX_MODERNVBERT_CHECKPOINT" not in os.environ
    or "REPRESENTAX_MODERNVBERT_ORACLE" not in os.environ,
    reason=(
        "set REPRESENTAX_MODERNVBERT_CHECKPOINT and "
        "REPRESENTAX_MODERNVBERT_ORACLE for the pinned model gate"
    ),
)
@pytest.mark.parametrize("rematerialization", ["none", "selective", "full"])
def test_pinned_transformers_forward_and_input_gradient_parity(rematerialization):
    reference = np.load(Path(os.environ["REPRESENTAX_MODERNVBERT_ORACLE"]))
    model = ModernVBERTTextCheckpointAdapter().load(
        Path(os.environ["REPRESENTAX_MODERNVBERT_CHECKPOINT"]),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        rematerialization=rematerialization,
    )
    batch = ModernVBERTTextBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
    )
    raw_embeddings = jnp.asarray(reference["inputs_embeds"], dtype=jnp.float32)
    objective_vector = jnp.linspace(-0.5, 0.5, model.metadata.output_dimension)

    @eqx.filter_jit
    def parity_outputs(candidate, token_batch, embeddings, vector):
        hidden = candidate.hidden_states(token_batch)
        pooled = candidate.encode(token_batch, route=Route.QUERY)

        def objective(values):
            embedded_batch = ModernVBERTTextBatch(
                inputs_embeds=values,
                attention_mask=token_batch.attention_mask,
            )
            return jnp.sum(candidate.encode(embedded_batch, route=Route.QUERY) * vector)

        return hidden, pooled, jax.grad(objective)(embeddings)

    with jax.default_matmul_precision("highest"):
        hidden, pooled, input_gradients = parity_outputs(
            model, batch, raw_embeddings, objective_vector
        )
    hidden = np.asarray(hidden, dtype=np.float32)
    pooled = np.asarray(pooled, dtype=np.float32)
    input_gradients = np.asarray(input_gradients, dtype=np.float32)

    assert_numerically_equivalent(
        hidden,
        reference["hidden"],
        NumericalTolerance(absolute=1e-3, relative=5e-4, cosine=0.9999998),
    )
    assert_numerically_equivalent(
        pooled,
        reference["pooled"],
        NumericalTolerance(absolute=5e-6, relative=1e-5, cosine=0.9999998),
    )
    assert_numerically_equivalent(
        input_gradients,
        reference["input_grads"],
        NumericalTolerance(absolute=3e-3, relative=5e-4, cosine=0.9999998),
    )


@pytest.mark.skipif(
    "REPRESENTAX_MODERNVBERT_CHECKPOINT" not in os.environ
    or "REPRESENTAX_MODERNVBERT_MULTIMODAL_ORACLE" not in os.environ,
    reason=(
        "set REPRESENTAX_MODERNVBERT_CHECKPOINT and "
        "REPRESENTAX_MODERNVBERT_MULTIMODAL_ORACLE for the pinned image gate"
    ),
)
def test_pinned_transformers_multimodal_forward_and_pixel_gradient_parity():
    reference = np.load(Path(os.environ["REPRESENTAX_MODERNVBERT_MULTIMODAL_ORACLE"]))
    model = ModernVBERTCheckpointAdapter().load(
        Path(os.environ["REPRESENTAX_MODERNVBERT_CHECKPOINT"]),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    input_ids = jnp.asarray(reference["input_ids"])
    attention_mask = jnp.asarray(reference["attention_mask"])
    pixel_attention_mask = jnp.asarray(reference["pixel_attention_mask"])
    pixels = jnp.asarray(reference["pixel_values"], dtype=jnp.float32)
    objective_vector = jnp.linspace(-1.0, 1.0, model.metadata.output_dimension)[None]

    @eqx.filter_jit
    def parity_outputs(candidate, pixel_values, vector):
        def objective(values):
            batch = ModernVBERTBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=values,
                pixel_attention_mask=pixel_attention_mask,
            )
            features = candidate.image_features(values)
            pooled = candidate.encode(batch, route=Route.GENERIC)
            return jnp.sum(pooled * vector), (features, pooled)

        (_, (features, pooled)), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(pixel_values)
        return features, pooled, gradients

    with jax.default_matmul_precision("highest"):
        image_features, pooled, pixel_gradients = parity_outputs(
            model,
            pixels,
            objective_vector,
        )
    image_features = np.asarray(image_features, dtype=np.float32).reshape(
        reference["image_features"].shape
    )

    assert_numerically_equivalent(
        image_features,
        reference["image_features"],
        NumericalTolerance(absolute=2e-4, relative=1e-5, cosine=0.9999998),
    )
    assert_numerically_equivalent(
        np.asarray(pooled, dtype=np.float32),
        reference["pooled"],
        NumericalTolerance(absolute=5e-6, relative=1e-5, cosine=0.9999998),
    )
    assert_numerically_equivalent(
        np.asarray(pixel_gradients, dtype=np.float32),
        reference["pixel_grad"],
        NumericalTolerance(absolute=3e-4, relative=3e-3, cosine=0.9999),
    )


@pytest.mark.skipif(
    "REPRESENTAX_MODERNVBERT_CHECKPOINT" not in os.environ
    or "REPRESENTAX_MODERNVBERT_MULTICROP_ORACLE" not in os.environ,
    reason=(
        "set REPRESENTAX_MODERNVBERT_CHECKPOINT and "
        "REPRESENTAX_MODERNVBERT_MULTICROP_ORACLE for the full crop gate"
    ),
)
def test_pinned_transformers_multicrop_forward_parity():
    reference = np.load(Path(os.environ["REPRESENTAX_MODERNVBERT_MULTICROP_ORACLE"]))
    model = ModernVBERTCheckpointAdapter().load(
        Path(os.environ["REPRESENTAX_MODERNVBERT_CHECKPOINT"]),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    batch = ModernVBERTBatch(
        input_ids=jnp.asarray(reference["input_ids"]),
        attention_mask=jnp.asarray(reference["attention_mask"]),
        pixel_values=jnp.asarray(reference["pixel_values"], dtype=jnp.float32),
        pixel_attention_mask=jnp.asarray(reference["pixel_attention_mask"]),
    )

    @eqx.filter_jit
    def parity_outputs(candidate, inputs):
        return candidate.image_features(inputs.pixel_values), candidate.encode(
            inputs, route=Route.GENERIC
        )

    with jax.default_matmul_precision("highest"):
        image_features, pooled = parity_outputs(model, batch)
    image_features = np.asarray(image_features, dtype=np.float32).reshape(
        reference["image_features"].shape
    )

    assert_numerically_equivalent(
        image_features,
        reference["image_features"],
        NumericalTolerance(absolute=2e-4, relative=1e-5, cosine=0.9999998),
    )
    assert_numerically_equivalent(
        np.asarray(pooled, dtype=np.float32),
        reference["pooled"],
        NumericalTolerance(absolute=5e-6, relative=1e-5, cosine=0.9999998),
    )
