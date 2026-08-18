"""Embedding-similarity evaluator contracts."""

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import Route, encode
from representax.evaluation import (
    EmbeddingSimilarityEvaluator,
    embedding_similarity_metrics,
)
from representax.models import DenseEncoder
from representax.tasks.pairwise import pairwise_batch
from representax.train import EvaluationRunner


def _batches():
    yield pairwise_batch(
        left=jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        right=jnp.asarray([[0.9, 0.1], [0.3, 0.7], [-1.0, -1.0]]),
        labels=jnp.asarray([0.95, 0.8, 0.0]),
        valid=jnp.asarray([True, True, False]),
    )
    yield pairwise_batch(
        left=jnp.asarray([[1.0, -1.0], [-1.0, 0.0]]),
        right=jnp.asarray([[0.8, -0.7], [1.0, 0.0]]),
        labels=jnp.asarray([0.9, 0.1]),
    )


def test_embedding_similarity_runner_reduces_over_the_complete_corpus():
    model = DenseEncoder(2, 2, key=jax.random.key(17))
    evaluator = EmbeddingSimilarityEvaluator(
        name="sts",
        similarity_functions=("cosine", "dot"),
        main_similarity="cosine",
    )

    result = EvaluationRunner(evaluator).run(model, _batches())

    all_batches = list(_batches())
    left = np.concatenate(
        [
            np.asarray(encode(model, batch.left, route=Route.GENERIC))
            for batch in all_batches
        ]
    )
    right = np.concatenate(
        [
            np.asarray(encode(model, batch.right, route=Route.GENERIC))
            for batch in all_batches
        ]
    )
    labels = np.concatenate([np.asarray(batch.labels) for batch in all_batches])
    valid = np.concatenate([np.asarray(batch.valid) for batch in all_batches])
    expected = embedding_similarity_metrics(
        left[valid],
        right[valid],
        labels[valid],
        similarity_functions=("cosine", "dot"),
    )

    assert result.batches == 2
    assert result.examples == 5
    assert evaluator.primary_metric == "valid/sts/spearman_cosine"
    assert result.metrics.keys() == {f"valid/sts/{name}" for name in expected}
    for name, value in expected.items():
        np.testing.assert_allclose(result.metrics[f"valid/sts/{name}"], value)


def test_embedding_similarity_rejects_less_than_two_valid_pairs():
    evaluator = EmbeddingSimilarityEvaluator()
    model = DenseEncoder(2, 2, key=jax.random.key(19))
    batch = pairwise_batch(
        left=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        right=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        labels=jnp.asarray([1.0, 1.0]),
        valid=jnp.asarray([True, False]),
    )

    with np.testing.assert_raises_regex(ValueError, "at least two valid pairs"):
        EvaluationRunner(evaluator).run(model, [batch])
