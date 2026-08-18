"""Same-tensor evaluator parity with Sentence Transformers 5.6.1."""

import numpy as np
import pytest

from representax.evaluation import embedding_similarity_metrics


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

    actual = embedding_similarity_metrics(left, right, labels)
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
