"""Named execution sharding and process-local input assembly."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.tasks.retrieval import (
    ProcessLocalRetrievalBatch,
    process_local_retrieval_batch,
)
from representax.train import DataParallel


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
    plan = DataParallel.from_devices([jax.devices("cpu")[0]])
    local_batch = process_local_retrieval_batch(
        query=jnp.arange(6).reshape(2, 3),
        document=jnp.arange(6, 12).reshape(2, 3),
        positive_mask=jnp.eye(2, dtype=jnp.bool_),
    )

    global_batch = plan.place_process_local_batch(local_batch)

    np.testing.assert_array_equal(global_batch.query, local_batch.query)
    np.testing.assert_array_equal(global_batch.document, local_batch.document)
    np.testing.assert_array_equal(
        global_batch.positive_mask,
        local_batch.positive_mask,
    )
