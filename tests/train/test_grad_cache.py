"""Exact native-JAX GradCache execution tests."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from representax.core import EncoderMetadata, Modality, Route, encode
from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, make_train_state


def _assert_array_trees_close(actual: Any, expected: Any) -> None:
    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [
        leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)
    ]
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        assert jnp.allclose(actual_leaf, expected_leaf, rtol=2e-5, atol=2e-6)


def _nontrivial_batch():
    query = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    document = jnp.asarray(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 0.0, 1.0, 0.2],
            [0.2, 0.0, 0.8, 0.0],
            [0.2, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    positives = jnp.asarray(
        [
            [True, True, False, False, False, False, False],
            [False, False, True, False, False, False, False],
            [False, False, False, True, True, False, False],
            [False, False, False, False, False, True, False],
            [False, False, False, False, False, False, True],
        ]
    )
    weights = jnp.where(positives, 1.0, 0.0).at[0, 0].set(3.0)
    return retrieval_batch(
        query=query,
        document=document,
        positive_mask=positives,
        positive_weights=weights,
        query_valid=jnp.asarray([True, True, True, True, False]),
        document_valid=jnp.asarray([True, True, True, True, True, True, False]),
    )


@pytest.mark.runtime
@pytest.mark.parametrize("symmetric", [False, True])
def test_grad_cache_matches_direct_full_optimizer_update(symmetric: bool):
    model = DenseEncoder(4, 3, key=jax.random.key(0), normalize=False)
    task = MNRTask(
        scale=7.0,
        symmetric=symmetric,
        dimensions=(2, 3),
        dimension_weights=(1.0, 2.0),
    )
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-2)
    state = make_train_state(model, optimizer)
    batch = _nontrivial_batch()
    direct = build_train_step(task, optimizer, max_grad_norm=0.7)
    cached = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=GradCache(query_chunk_size=2, document_chunk_size=3),
    )

    direct_result = direct(state, batch, jax.random.key(17))
    cached_result = cached(state, batch, jax.random.key(17))

    _assert_array_trees_close(cached_result.metrics, direct_result.metrics)
    _assert_array_trees_close(cached_result.state, direct_result.state)


class _StochasticEncoder(eqx.Module):
    weight: jax.Array
    metadata: EncoderMetadata
    keep_probability: float = eqx.field(static=True)

    def __init__(self, *, key: jax.Array, keep_probability: float = 0.75) -> None:
        self.weight = jax.random.normal(key, (4, 3))
        self.metadata = EncoderMetadata(
            model_id="representax/test-stochastic",
            revision="1",
            output_dimension=3,
            routes=frozenset(Route),
            modalities=frozenset(Modality),
        )
        self.keep_probability = keep_probability

    def encode(
        self,
        inputs: jax.Array,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del route
        if key is None:
            raise ValueError("the stochastic test encoder requires a key")
        values = inputs @ self.weight
        mask = jax.random.bernoulli(key, self.keep_probability, values.shape)
        return jnp.where(mask, values / self.keep_probability, 0.0)


def _explicit_chunked_encode(
    model: _StochasticEncoder,
    inputs: jax.Array,
    *,
    route: Route,
    chunk_size: int,
    key: jax.Array,
) -> jax.Array:
    batch_size = inputs.shape[0]
    chunk_count = (batch_size + chunk_size - 1) // chunk_size
    padding = chunk_count * chunk_size - batch_size
    chunks = jnp.pad(inputs, ((0, padding), (0, 0))).reshape(
        chunk_count, chunk_size, inputs.shape[1]
    )
    keys = jax.random.split(key, chunk_count)
    embeddings = [
        encode(model, chunk, route=route, key=chunk_key)
        for chunk, chunk_key in zip(chunks, keys, strict=True)
    ]
    return jnp.concatenate(embeddings, axis=0)[:batch_size]


def test_grad_cache_rematerialization_replays_stochastic_chunks_exactly():
    model = _StochasticEncoder(key=jax.random.key(1))
    batch = _nontrivial_batch()
    task = MNRTask(scale=3.0, symmetric=True)
    execution = GradCache(query_chunk_size=2, document_chunk_size=3)
    key = jax.random.key(29)
    query_key, document_key = jax.random.split(key)

    def explicit_loss(candidate: _StochasticEncoder) -> jax.Array:
        queries = _explicit_chunked_encode(
            candidate,
            batch.query,
            route=Route.QUERY,
            chunk_size=2,
            key=query_key,
        )
        documents = _explicit_chunked_encode(
            candidate,
            batch.document,
            route=Route.DOCUMENT,
            chunk_size=3,
            key=document_key,
        )
        return task.loss_from_embeddings(queries, documents, batch).loss

    def cached_loss(candidate: _StochasticEncoder) -> jax.Array:
        return execution.evaluate(task, candidate, batch, key=key).loss

    explicit_value, explicit_gradients = eqx.filter_value_and_grad(explicit_loss)(
        model
    )
    cached_value, cached_gradients = eqx.filter_value_and_grad(cached_loss)(model)

    assert jnp.array_equal(cached_value, explicit_value)
    _assert_array_trees_close(cached_gradients, explicit_gradients)
    assert "remat" in str(jax.make_jaxpr(cached_loss)(model))


@pytest.mark.parametrize(
    ("query_chunk_size", "document_chunk_size"),
    [(0, None), (-1, 2), (2, 0), (2, -1)],
)
def test_grad_cache_rejects_nonpositive_chunk_sizes(
    query_chunk_size: int,
    document_chunk_size: int | None,
):
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        GradCache(
            query_chunk_size=query_chunk_size,
            document_chunk_size=document_chunk_size,
        )
