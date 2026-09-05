"""Device-local MNR semantics on data-parallel meshes."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType

from representax.core import LossOutput
from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import (
    GradCache,
    ShardingPlan,
    build_train_step,
    init_train_state,
)


def _assert_array_trees_close(actual: Any, expected: Any) -> None:
    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=4e-5,
            atol=4e-6,
        )


def _batch():
    query = jnp.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0, 0.0),
            (0.0, 1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ),
        dtype=jnp.float32,
    )
    document = jnp.asarray(
        (
            (0.8, 0.2, 0.0, 0.0),
            (0.1, 0.9, 0.0, 0.0),
            (0.0, 0.1, 0.9, 0.0),
            (0.0, 0.0, 0.1, 0.9),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.8, 0.2, 0.0),
            (0.0, 0.0, 0.8, 0.2),
            (0.8, 0.0, 0.0, 0.2),
        ),
        dtype=jnp.float32,
    )
    return retrieval_batch(
        query=query,
        document=document,
        positive_mask=jnp.eye(8, dtype=jnp.bool_),
        positive_weights=jnp.eye(8, dtype=jnp.float32).at[2, 2].set(2.0),
        query_valid=jnp.asarray([True, True, True, True, True, True, True, False]),
        document_valid=jnp.asarray([True, True, True, True, True, True, False, True]),
    )


class _ExplicitGroupedMNR(MNRTask):
    groups: int = eqx.field(static=True, default=2)

    def loss_from_embeddings(
        self,
        query_embeddings: jax.Array,
        document_embeddings: jax.Array,
        batch: Any,
        *,
        row_chunk_size: int | None = None,
    ) -> LossOutput:
        query_size = query_embeddings.shape[0] // self.groups
        document_size = document_embeddings.shape[0] // self.groups
        task = MNRTask(
            scale=self.scale,
            symmetric=self.symmetric,
            negative_scope="global",
        )
        outputs = []
        for index in range(self.groups):
            query_slice = slice(index * query_size, (index + 1) * query_size)
            document_slice = slice(
                index * document_size,
                (index + 1) * document_size,
            )
            local_batch = retrieval_batch(
                query=batch.query[query_slice],
                document=batch.document[document_slice],
                positive_mask=batch.positive_mask[query_slice, document_slice],
                positive_weights=batch.positive_weights[query_slice, document_slice],
                query_valid=batch.query_valid[query_slice],
                document_valid=batch.document_valid[document_slice],
            )
            outputs.append(
                task.loss_from_embeddings(
                    query_embeddings[query_slice],
                    document_embeddings[document_slice],
                    local_batch,
                    row_chunk_size=row_chunk_size,
                )
            )
        return LossOutput(
            loss=jnp.mean(jnp.stack([output.loss for output in outputs])),
            metrics={
                name: jnp.mean(jnp.stack([output.metrics[name] for output in outputs]))
                for name in outputs[0].metrics
            },
        )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("axis_type", [AxisType.Explicit, AxisType.Auto])
def test_ddp_grad_cache_matches_explicit_device_local_update(
    world_size: int,
    axis_type: AxisType,
) -> None:
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = DenseEncoder(4, 3, key=jax.random.key(5), normalize=False)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        _ExplicitGroupedMNR(
            scale=9.0,
            symmetric=True,
            negative_scope="local",
            groups=world_size,
        ),
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        devices=devices[:world_size],
        axis_types=(axis_type,),
    )
    plan = ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
    distributed_step = build_train_step(
        MNRTask(scale=9.0, symmetric=True, negative_scope="local"),
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )

    reference = reference_step(state, batch, jax.random.key(17))
    distributed = distributed_step(
        plan.place_state(state),
        plan.place_batch(batch),
        jax.device_put(jax.random.key(17), plan.replicated_sharding),
    )
    jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)
