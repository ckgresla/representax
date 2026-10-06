"""Native multiple-negatives ranking tests."""

import jax
import jax.numpy as jnp
import numpy as np

from representax.models.mpnet import MPNetBatch
from representax.tasks.retrieval import MNRTask, mnr_loss_terms, retrieval_batch


def test_mnr_prefers_aligned_pairs():
    aligned = jnp.eye(3, dtype=jnp.float32)
    positives = jnp.eye(3, dtype=jnp.bool_)

    good = mnr_loss_terms(aligned, aligned, positives, scale=10.0)
    bad = mnr_loss_terms(aligned, aligned[::-1], positives, scale=10.0)

    assert good.loss < bad.loss
    assert good.forward_loss == good.loss
    assert good.reverse_loss == 0


def test_symmetric_mnr_reports_both_directions():
    queries = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    documents = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    positives = jnp.eye(2, dtype=jnp.bool_)

    terms = mnr_loss_terms(
        queries,
        documents,
        positives,
        scale=5.0,
        symmetric=True,
    )

    assert jnp.allclose(terms.forward_loss, terms.reverse_loss)
    assert jnp.allclose(terms.loss, terms.forward_loss)


def test_graded_mnr_weights_multiple_positives():
    queries = jnp.asarray([[1.0, 0.0]])
    documents = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    positives = jnp.asarray([[True, True]])

    uniform = mnr_loss_terms(queries, documents, positives, scale=5.0)
    graded = mnr_loss_terms(
        queries,
        documents,
        positives,
        positive_weights=jnp.asarray([[9.0, 1.0]]),
        scale=5.0,
    )

    assert graded.loss < uniform.loss


def test_tiled_mnr_matches_full_values_and_representation_gradients():
    queries = jnp.asarray(
        [[0.8, -0.2, 0.3], [0.1, 0.9, -0.4], [-0.5, 0.2, 0.7]],
        dtype=jnp.float32,
    )
    documents = jnp.asarray(
        [
            [0.7, -0.1, 0.2],
            [0.2, 0.8, -0.3],
            [-0.4, 0.3, 0.8],
            [0.5, 0.5, 0.0],
            [-0.3, -0.6, 0.4],
        ],
        dtype=jnp.float32,
    )
    positives = jnp.asarray(
        [
            [True, False, False, True, False],
            [False, True, False, False, False],
            [False, False, True, False, True],
        ]
    )
    weights = jnp.where(positives, 1.0, 0.0).at[0, 0].set(2.0)
    options = {
        "positive_weights": weights,
        "query_valid": jnp.asarray([True, True, False]),
        "document_valid": jnp.asarray([True, True, True, True, False]),
        "scale": 7.0,
        "symmetric": True,
    }
    batch = retrieval_batch(
        query=queries,
        document=documents,
        positive_mask=positives,
        positive_weights=weights,
        query_valid=options["query_valid"],
        document_valid=options["document_valid"],
    )
    task = MNRTask(scale=options["scale"], symmetric=options["symmetric"])

    def objective(query, document, row_chunk_size):
        return task.loss_from_embeddings(
            query,
            document,
            batch,
            row_chunk_size=row_chunk_size,
        ).loss

    full_value, full_gradients = jax.value_and_grad(
        lambda query, document: objective(query, document, None),
        argnums=(0, 1),
    )(queries, documents)
    tiled_value, tiled_gradients = jax.value_and_grad(
        lambda query, document: objective(query, document, 2),
        argnums=(0, 1),
    )(queries, documents)

    np.testing.assert_allclose(tiled_value, full_value, rtol=2e-6, atol=2e-7)
    for tiled, full in zip(tiled_gradients, full_gradients, strict=True):
        np.testing.assert_allclose(tiled, full, rtol=3e-5, atol=3e-6)


def test_retrieval_batch_accepts_packed_payload_logical_counts():
    packed = MPNetBatch(
        input_ids=jnp.asarray([[0, 4, 2, 0, 5, 2, 1]]),
        attention_mask=jnp.asarray([[1, 1, 1, 1, 1, 1, 0]]),
        position_ids=jnp.asarray([[2, 3, 4, 2, 3, 4, 1]]),
        segment_ids=jnp.asarray([[0, 0, 0, 1, 1, 1, -1]]),
        logical_batch_size=2,
    )
    batch = retrieval_batch(
        query=packed,
        document=packed,
        positive_mask=np.eye(2, dtype=np.bool_),
    )
    assert batch.positive_mask.shape == (2, 2)
