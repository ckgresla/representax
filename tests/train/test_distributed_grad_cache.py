"""Global-negative GradCache acceptance over named data-parallel meshes."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route, encode
from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask, mnr_loss_terms, retrieval_batch
from representax.train import (
    DataParallel,
    GradCache,
    build_data_parallel_train_step,
    build_train_step,
    init_train_state,
)


def _assert_array_trees_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 4e-5,
    atol: float = 4e-6,
) -> None:
    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


def _global_batch():
    query = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    document = jnp.asarray(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.1, 0.9],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.8, 0.0, 0.0, 0.2],
        ],
        dtype=jnp.float32,
    )
    positive_mask = jnp.eye(8, dtype=jnp.bool_)
    positive_weights = jnp.eye(8, dtype=jnp.float32).at[2, 2].set(2.0)
    return retrieval_batch(
        query=query,
        document=document,
        positive_mask=positive_mask,
        positive_weights=positive_weights,
        query_valid=jnp.asarray([True, True, True, True, True, True, True, False]),
        document_valid=jnp.asarray([True, True, True, True, True, True, False, True]),
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
def test_data_parallel_grad_cache_matches_one_device_global_update(world_size: int):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = DenseEncoder(4, 3, key=jax.random.key(5), normalize=False)
    task = MNRTask(
        scale=9.0,
        symmetric=True,
        dimensions=(2, 3),
        dimension_weights=(1.0, 2.0),
    )
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _global_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
    plan = DataParallel.from_devices(devices[:world_size])
    distributed_step = build_data_parallel_train_step(
        task,
        optimizer,
        plan,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )

    with jax.default_matmul_precision("highest"):
        reference = reference_step(state, batch, jax.random.key(17))
        distributed = distributed_step(
            plan.place_replicated(state),
            plan.place_batch(batch),
            plan.place_replicated(jax.random.key(17)),
        )
        jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)

    local_count = batch.query.shape[0] // world_size
    queries = encode(model, batch.query, route=Route.QUERY)
    documents = encode(model, batch.document, route=Route.DOCUMENT)
    global_terms = mnr_loss_terms(
        queries,
        documents,
        batch.positive_mask,
        positive_weights=batch.positive_weights,
        query_valid=batch.query_valid,
        document_valid=batch.document_valid,
        scale=task.scale,
    )
    local_terms = mnr_loss_terms(
        queries[:1],
        documents[:local_count],
        batch.positive_mask[:1, :local_count],
        positive_weights=batch.positive_weights[:1, :local_count],
        query_valid=batch.query_valid[:1],
        document_valid=batch.document_valid[:local_count],
        scale=task.scale,
    )
    assert float(global_terms.row_losses[0]) > float(local_terms.row_losses[0])
