"""Native multiple-negatives ranking tests."""

import jax.numpy as jnp

from representax.tasks.retrieval import mnr_loss_terms


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
