"""Information-retrieval evaluator contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from representax.evaluation import (
    InformationRetrievalEvaluator,
    information_retrieval_metrics,
    retrieval_evaluation_batch,
)
from representax.models import DenseEncoder
from representax.train import EvaluationRunner


def test_information_retrieval_metrics_cover_distinct_ranking_geometries():
    metrics = information_retrieval_metrics(
        np.asarray(
            [
                [20, 21, 22, 23],
                [23, 22, 21, 20],
            ]
        ),
        np.asarray([10, 11]),
        {
            10: frozenset({20, 23}),
            11: frozenset({21}),
        },
        accuracy_at_k=(1, 2),
        precision_recall_at_k=(1, 2),
        mrr_at_k=(4,),
        ndcg_at_k=(4,),
        map_at_k=(4,),
    )

    np.testing.assert_allclose(metrics["accuracy@1"], 0.5)
    np.testing.assert_allclose(metrics["accuracy@2"], 0.5)
    np.testing.assert_allclose(metrics["precision@1"], 0.5)
    np.testing.assert_allclose(metrics["precision@2"], 0.25)
    np.testing.assert_allclose(metrics["recall@1"], 0.25)
    np.testing.assert_allclose(metrics["recall@2"], 0.25)
    np.testing.assert_allclose(metrics["mrr@4"], 2 / 3)


def _retrieval_batches():
    yield retrieval_evaluation_batch(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
        jnp.asarray([10, 11]),
        kind="query",
    )
    yield retrieval_evaluation_batch(
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        jnp.asarray([20, 21, -1]),
        kind="document",
        valid=jnp.asarray([True, True, False]),
    )
    yield retrieval_evaluation_batch(
        jnp.asarray([[-1.0, 0.0], [0.1, 0.9]]),
        jnp.asarray([22, 23]),
        kind="document",
    )


def test_information_retrieval_runner_streams_bounded_rankings_over_corpus_batches():
    model = DenseEncoder(2, 2, key=jax.random.key(31))
    evaluator = InformationRetrievalEvaluator(
        relevant_documents={10: frozenset({20}), 11: frozenset({21, 23})},
        name="toy",
        score_functions=("cosine", "dot"),
        main_score_function="cosine",
        accuracy_at_k=(1, 2),
        precision_recall_at_k=(1, 2),
        mrr_at_k=(2,),
        ndcg_at_k=(2,),
        map_at_k=(2,),
    )

    result = EvaluationRunner(evaluator).run(model, _retrieval_batches())

    assert result.batches == 3
    assert result.examples == 7
    assert evaluator.primary_metric == "valid/toy/cosine_ndcg@2"
    assert set(result.metrics) == {
        f"valid/toy/{function}_{metric}@{k}"
        for function in ("cosine", "dot")
        for metric, ks in (
            ("accuracy", (1, 2)),
            ("precision", (1, 2)),
            ("recall", (1, 2)),
            ("mrr", (2,)),
            ("ndcg", (2,)),
            ("map", (2,)),
        )
        for k in ks
    }
    assert result.metrics["valid/toy/cosine_accuracy@1"] == 1.0
    assert result.metrics["valid/toy/cosine_recall@1"] == 0.75


def test_information_retrieval_rejects_query_batches_after_corpus_batches():
    model = DenseEncoder(2, 2, key=jax.random.key(37))
    evaluator = InformationRetrievalEvaluator(
        relevant_documents={10: frozenset({20})},
        accuracy_at_k=(1,),
        precision_recall_at_k=(1,),
        mrr_at_k=(1,),
        ndcg_at_k=(1,),
        map_at_k=(1,),
    )
    query = retrieval_evaluation_batch(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([10]),
        kind="query",
    )
    document = retrieval_evaluation_batch(
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([20]),
        kind="document",
    )

    with pytest.raises(ValueError, match="query batches must precede"):
        EvaluationRunner(evaluator).run(model, [query, document, query])
