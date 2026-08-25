"""Same-tensor evaluator parity with Sentence Transformers 5.6.1."""

import numpy as np
import pytest

from representax.evaluation import (
    information_retrieval_metrics,
    similarity_metrics,
)


@pytest.mark.parity
def test_embedding_similarity_matches_sentence_transformers_5_6_1():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    scipy_stats = pytest.importorskip("scipy.stats")
    upstream_util = pytest.importorskip("sentence_transformers.util")
    if sentence_transformers.__version__ != "5.6.1":
        pytest.fail("embedding-similarity parity requires sentence-transformers==5.6.1")

    rng = np.random.default_rng(711)
    left = rng.normal(size=(37, 19)).astype(np.float32)
    right = rng.normal(size=(37, 19)).astype(np.float32)
    labels = rng.uniform(0.0, 1.0, size=37).astype(np.float32)
    # Exercise SciPy's average-rank tie semantics instead of testing only unique data.
    labels[[2, 9, 21]] = 0.5

    actual = similarity_metrics(left, right, labels)
    tensors = (
        __import__("torch").from_numpy(left),
        __import__("torch").from_numpy(right),
    )
    functions = {
        "cosine": upstream_util.pairwise_cos_sim,
        "dot": upstream_util.pairwise_dot_score,
        "euclidean": upstream_util.pairwise_euclidean_sim,
        "manhattan": upstream_util.pairwise_manhattan_sim,
    }
    expected = {}
    for name, function in functions.items():
        scores = function(*tensors).detach().cpu().numpy()
        expected[f"pearson_{name}"] = float(
            scipy_stats.pearsonr(labels, scores).statistic
        )
        expected[f"spearman_{name}"] = float(
            scipy_stats.spearmanr(labels, scores).statistic
        )
    expected["pearson_max"] = max(expected[f"pearson_{name}"] for name in functions)
    expected["spearman_max"] = max(expected[f"spearman_{name}"] for name in functions)

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=2e-6, atol=2e-7)


@pytest.mark.parity
def test_information_retrieval_matches_sentence_transformers_5_6_1():
    sentence_transformers = pytest.importorskip("sentence_transformers")
    evaluation = pytest.importorskip(
        "sentence_transformers.sentence_transformer.evaluation"
    )
    if sentence_transformers.__version__ != "5.6.1":
        pytest.fail("retrieval parity requires sentence-transformers==5.6.1")

    rng = np.random.default_rng(719)
    query_embeddings = rng.normal(size=(7, 13)).astype(np.float32)
    document_embeddings = rng.normal(size=(23, 13)).astype(np.float32)
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    document_embeddings /= np.linalg.norm(
        document_embeddings,
        axis=1,
        keepdims=True,
    )
    scores = query_embeddings @ document_embeddings.T
    ranked_indices = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    query_ids = np.arange(100, 107, dtype=np.int32)
    document_ids = np.arange(200, 223, dtype=np.int32)
    rankings = document_ids[ranked_indices]
    relevant_documents = {
        int(query_id): frozenset(
            {
                int(rankings[index, 0]),
                int(rankings[index, 3]),
                int(document_ids[(index * 5 + 11) % len(document_ids)]),
            }
        )
        for index, query_id in enumerate(query_ids)
    }
    settings = {
        "accuracy_at_k": (1, 3, 5, 10),
        "precision_recall_at_k": (1, 3, 5, 10),
        "mrr_at_k": (10,),
        "ndcg_at_k": (10,),
        "map_at_k": (10,),
    }

    actual = information_retrieval_metrics(
        rankings,
        query_ids,
        relevant_documents,
        **settings,
    )
    upstream = evaluation.InformationRetrievalEvaluator(
        queries={str(value): "query" for value in query_ids},
        corpus={str(value): "document" for value in document_ids},
        relevant_docs={
            str(query_id): {str(document_id) for document_id in documents}
            for query_id, documents in relevant_documents.items()
        },
        accuracy_at_k=list(settings["accuracy_at_k"]),
        precision_recall_at_k=list(settings["precision_recall_at_k"]),
        mrr_at_k=list(settings["mrr_at_k"]),
        ndcg_at_k=list(settings["ndcg_at_k"]),
        map_at_k=list(settings["map_at_k"]),
        write_csv=False,
    )
    upstream_rankings = [
        [
            {
                "corpus_id": str(document_id),
                "score": float(scores[query_index, document_id - 200]),
            }
            for document_id in row
        ]
        for query_index, row in enumerate(rankings)
    ]
    nested = upstream.compute_metrics(upstream_rankings)
    expected = {
        f"{metric.replace('@k', '')}@{k}": float(value)
        for metric, values in nested.items()
        for k, value in values.items()
    }

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, rtol=0.0, atol=1e-12)
