"""Named execution sharding and process-local input assembly."""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

from representax.models import DenseEncoder
from representax.tasks.retrieval import (
    ProcessLocalRetrievalBatch,
    place_process_local_retrieval_batch,
    process_local_retrieval_batch,
)
from representax.train import (
    ShardingPlan,
    init_train_state,
    parameter_specs_from_rules,
)


def test_process_local_retrieval_batch_keeps_global_relation_columns():
    batch = process_local_retrieval_batch(
        query=jnp.ones((2, 3)),
        document=jnp.ones((2, 3)),
        positive_mask=jnp.asarray(
            [[True, False, False, False], [False, False, True, False]]
        ),
    )

    assert isinstance(batch, ProcessLocalRetrievalBatch)
    assert batch.query.shape == (2, 3)
    assert batch.document.shape == (2, 3)
    assert batch.positive_mask.shape == (2, 4)
    np.testing.assert_array_equal(batch.query_valid, [True, True])
    np.testing.assert_array_equal(batch.document_valid, [True, True])


def test_process_local_retrieval_batch_rejects_short_global_document_axis():
    with pytest.raises(ValueError, match="at least the local document rows"):
        process_local_retrieval_batch(
            query=jnp.ones((2, 3)),
            document=jnp.ones((3, 3)),
            positive_mask=jnp.ones((2, 2), dtype=jnp.bool_),
        )


def test_data_parallel_assembles_process_local_rows_on_one_process():
    device = jax.devices("cpu")[0]
    mesh = jax.make_mesh((1,), ("data",), devices=[device])
    sharding = jax.sharding.NamedSharding(mesh, P("data"))
    local_batch = process_local_retrieval_batch(
        query=jnp.arange(6).reshape(2, 3),
        document=jnp.arange(6, 12).reshape(2, 3),
        positive_mask=jnp.eye(2, dtype=jnp.bool_),
    )

    global_batch = place_process_local_retrieval_batch(local_batch, sharding)

    np.testing.assert_array_equal(global_batch.query, local_batch.query)
    np.testing.assert_array_equal(global_batch.document, local_batch.document)
    np.testing.assert_array_equal(
        global_batch.positive_mask,
        local_batch.positive_mask,
    )


def test_custom_parameter_rules_build_model_shaped_specs():
    model = DenseEncoder(4, 4, key=jax.random.key(3))

    specs = cast(
        DenseEncoder,
        parameter_specs_from_rules(
            model,
            ((r"\.projection\.weight$", P("model", None)),),
        ),
    )

    assert specs.projection.weight == P("model", None)
    assert specs.projection.bias == P()


def test_fsdp_rejects_specs_that_do_not_match_parameter_divisibility():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires two JAX devices")
    model = DenseEncoder(3, 3, key=jax.random.key(3))
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    mesh = jax.make_mesh((2,), ("model",), devices=devices[:2])
    specs = cast(
        DenseEncoder,
        parameter_specs_from_rules(
            model,
            ((r"\.projection\.weight$", P("model", None)),),
        ),
    )

    with pytest.raises(ValueError, match="not divisible"):
        ShardingPlan.custom(
            state,
            optimizer,
            mesh,
            specs,
            parameter_axis_names=("model",),
            data_axis_name=None,
        )
