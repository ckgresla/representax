"""Exact native late-interaction representation and training contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import (
    BUILTIN_MODALITIES,
    EncoderMetadata,
    LateInteractionRepresentation,
    Route,
    encode_late_interaction,
)
from representax.tasks import build_task
from representax.tasks.late_interaction import (
    LateInteractionConfig,
    LateInteractionContrastiveConfig,
    LateInteractionTask,
    late_interaction_loss_terms,
    maxsim_scores,
)
from representax.tasks.retrieval import retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state

_PYLATE_ORACLE = (
    Path(__file__).parents[1] / "fixtures" / "late_interaction" / "pylate-1.6.0"
)


class _TokenBatch(eqx.Module):
    values: jax.Array
    valid: jax.Array


class _TokenEncoder(eqx.Module):
    projection: jax.Array
    metadata: EncoderMetadata

    def __init__(self, *, key: jax.Array) -> None:
        self.projection = jax.random.normal(key, (4, 3))
        self.metadata = EncoderMetadata(
            model_id="representax/test-late-interaction",
            revision="1",
            output_dimension=3,
            routes=frozenset(Route),
            modalities=BUILTIN_MODALITIES,
        )

    def encode_late_interaction(
        self,
        inputs: _TokenBatch,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> LateInteractionRepresentation:
        del route, key
        return LateInteractionRepresentation(
            values=jnp.matmul(
                inputs.values,
                self.projection,
                precision=jax.lax.Precision.HIGHEST,
            ),
            valid=inputs.valid,
        )


def _representation(values: Any, valid: Any) -> LateInteractionRepresentation:
    return LateInteractionRepresentation(
        values=jnp.asarray(values, dtype=jnp.float32),
        valid=jnp.asarray(valid, dtype=jnp.bool_),
    )


def _assert_array_trees_close(actual: Any, expected: Any) -> None:
    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=3e-5, atol=3e-6)


def test_shared_encoder_boundary_normalizes_and_zeros_invalid_tokens():
    model = _TokenEncoder(key=jax.random.key(0))
    inputs = _TokenBatch(
        values=jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4),
        valid=jnp.asarray([[True, True, False], [True, False, False]]),
    )

    encoded = encode_late_interaction(model, inputs, route=Route.QUERY)

    norms = jnp.linalg.norm(encoded.values, axis=-1)
    np.testing.assert_allclose(norms[encoded.valid], 1.0, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(encoded.values[~encoded.valid], 0.0)


def test_maxsim_matches_manual_masked_token_reduction():
    queries = _representation(
        [[[1, 0], [0, 1]], [[0, 1], [1, 0]]],
        [[True, True], [True, False]],
    )
    documents = _representation(
        [[[1, 0], [-1, 0]], [[0, 1], [1, 0]], [[1, 1], [0, 0]]],
        [[True, True], [True, True], [False, False]],
    )

    scores = maxsim_scores(queries, documents)

    expected = jnp.asarray([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_array_equal(scores, expected)


def test_masked_document_tokens_cannot_win_a_negative_maximum():
    queries = _representation([[[-1.0, 0.0]]], [[True]])
    documents = _representation([[[1.0, 0.0], [-1.0, 0.0]]], [[True, False]])

    scores = maxsim_scores(queries, documents)

    np.testing.assert_array_equal(scores, jnp.asarray([[-1.0]]))


def test_tiled_maxsim_matches_full_values_gradients_and_jit():
    query_values = jax.random.normal(jax.random.key(1), (3, 4, 5))
    document_values = jax.random.normal(jax.random.key(2), (5, 6, 5))
    query_valid = jnp.asarray(
        [[True, True, True, False], [True, True, False, False], [True] * 4]
    )
    document_valid = jnp.asarray(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [True] * 6,
            [True, True, True, False, False, False],
            [True, False, False, False, False, False],
        ]
    )

    def objective(q_values, d_values, chunk_size):
        return jnp.sum(
            maxsim_scores(
                _representation(q_values, query_valid),
                _representation(d_values, document_valid),
                document_chunk_size=chunk_size,
            )
        )

    full_value, full_gradients = jax.value_and_grad(
        lambda q, d: objective(q, d, None), argnums=(0, 1)
    )(query_values, document_values)
    tiled_value, tiled_gradients = jax.value_and_grad(
        lambda q, d: objective(q, d, 2), argnums=(0, 1)
    )(query_values, document_values)
    compiled = jax.jit(lambda q, d: objective(q, d, 2))(query_values, document_values)

    np.testing.assert_allclose(tiled_value, full_value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(compiled, full_value, rtol=2e-6, atol=2e-6)
    for tiled, full in zip(tiled_gradients, full_gradients, strict=True):
        np.testing.assert_allclose(tiled, full, rtol=3e-5, atol=3e-6)


def test_late_interaction_contrastive_prefers_aligned_pairs_and_builds_from_config():
    queries = _representation(
        [[[1, 0]], [[0, 1]]],
        [[True], [True]],
    )
    aligned = _representation(
        [[[1, 0]], [[0, 1]]],
        [[True], [True]],
    )
    reversed_documents = _representation(
        [[[0, 1]], [[1, 0]]],
        [[True], [True]],
    )
    positives = jnp.eye(2, dtype=jnp.bool_)

    good = late_interaction_loss_terms(
        queries,
        aligned,
        positives,
        temperature=0.1,
    )
    bad = late_interaction_loss_terms(
        queries,
        reversed_documents,
        positives,
        temperature=0.1,
    )
    task = build_task(
        LateInteractionConfig(),
        LateInteractionContrastiveConfig(temperature=0.1),
    )

    assert good.loss < bad.loss
    assert isinstance(task, LateInteractionTask)
    assert task.temperature == 0.1


def _training_batch() -> Any:
    query_values = jax.random.normal(jax.random.key(3), (5, 4, 4))
    document_values = query_values + 0.1 * jax.random.normal(
        jax.random.key(4), query_values.shape
    )
    query_valid = jnp.asarray(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
        ]
    )
    document_valid = jnp.asarray(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    return retrieval_batch(
        query=_TokenBatch(query_values, query_valid),
        document=_TokenBatch(document_values, document_valid),
        positive_mask=jnp.eye(5, dtype=jnp.bool_),
    )


def test_late_interaction_grad_cache_matches_direct_optimizer_update():
    model = _TokenEncoder(key=jax.random.key(5))
    task = LateInteractionTask(temperature=0.2, symmetric=True)
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _training_batch()
    direct = build_train_step(task, optimizer, max_grad_norm=0.8, donate_state=False)
    cached = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.8,
        execution=GradCache(
            query_chunk_size=2,
            document_chunk_size=3,
            loss_row_chunk_size=2,
        ),
        donate_state=False,
    )

    direct_result = direct(state, batch, jax.random.key(6))
    cached_result = cached(state, batch, jax.random.key(6))

    _assert_array_trees_close(cached_result.metrics, direct_result.metrics)
    _assert_array_trees_close(cached_result.state, direct_result.state)


@pytest.mark.parity
def test_maxsim_loss_and_representation_gradients_match_pylate():
    metadata = json.loads((_PYLATE_ORACLE / "metadata.json").read_text())
    assert metadata["pylate_version"] == "1.6.0"
    assert metadata["backend"] == "torch"
    reference = np.load(_PYLATE_ORACLE / "oracle.npz")
    query_values = reference["query_values"]
    document_values = reference["document_values"]
    query_valid = reference["query_valid"]
    document_valid = reference["document_valid"]
    positives = jnp.asarray(reference["positive_mask"])
    temperature = float(reference["temperature"])

    def native_objective(query, document):
        return late_interaction_loss_terms(
            _representation(query, query_valid),
            _representation(document, document_valid),
            positives,
            temperature=temperature,
            document_chunk_size=2,
        ).loss

    native_value, native_gradients = jax.value_and_grad(
        native_objective,
        argnums=(0, 1),
    )(jnp.asarray(query_values), jnp.asarray(document_values))

    native_scores = maxsim_scores(
        _representation(query_values, query_valid),
        _representation(document_values, document_valid),
        document_chunk_size=2,
    )

    np.testing.assert_allclose(
        native_value,
        reference["loss"],
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        native_gradients[0],
        reference["query_gradient"],
        rtol=5e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        native_gradients[1],
        reference["document_gradient"],
        rtol=5e-5,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        native_scores,
        reference["scores"],
        rtol=2e-5,
        atol=2e-6,
    )
