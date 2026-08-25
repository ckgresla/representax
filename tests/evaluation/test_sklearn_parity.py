"""Exact host-reducer parity with the pinned scikit-learn algorithms."""

import numpy as np
import pytest

from representax.evaluation import clustering_metrics, linear_probe_metrics


@pytest.mark.parity
def test_linear_probe_matches_direct_scikit_learn() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(733)
    embeddings = rng.normal(size=(90, 11))
    labels = np.repeat(np.arange(3), 30)
    embeddings[:, :3] += np.eye(3)[labels] * 2.5
    splits = np.tile(
        np.repeat(np.asarray((0, 1, 2)), 10),
        3,
    )
    candidates = (0.1, 1.0, 10.0)
    actual = linear_probe_metrics(
        embeddings,
        labels,
        splits,
        inverse_regularization=candidates,
        normalization="none",
        max_iterations=500,
        seed=17,
    )

    train = splits == 0
    validation = splits == 1
    test = splits == 2
    validation_scores = []
    for value in candidates:
        probe = linear_model.LogisticRegression(
            C=value,
            max_iter=500,
            random_state=17,
            solver="lbfgs",
        ).fit(embeddings[train], labels[train])
        validation_scores.append(
            probe.score(embeddings[validation], labels[validation])
        )
    selected = candidates[int(np.argmax(validation_scores))]
    fit = train | validation
    reference = linear_model.LogisticRegression(
        C=selected,
        max_iter=500,
        random_state=17,
        solver="lbfgs",
    ).fit(embeddings[fit], labels[fit])
    predictions = reference.predict(embeddings[test])
    expected = {
        "accuracy": metrics.accuracy_score(labels[test], predictions),
        "f1_macro": metrics.f1_score(labels[test], predictions, average="macro"),
        "validation_accuracy": max(validation_scores),
        "selected_inverse_regularization": selected,
    }
    assert actual == expected


@pytest.mark.parity
def test_clustering_matches_direct_scikit_learn() -> None:
    cluster = pytest.importorskip("sklearn.cluster")
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(739)
    labels = np.repeat(np.arange(4), 25)
    centers = np.eye(4, 7) * 4.0
    embeddings = centers[labels] + rng.normal(scale=0.1, size=(100, 7))
    actual = clustering_metrics(
        embeddings,
        labels,
        normalization="none",
        batch_size=32,
        max_iterations=80,
        n_init=5,
        seed=19,
    )
    predicted = cluster.MiniBatchKMeans(
        n_clusters=4,
        batch_size=32,
        max_iter=80,
        n_init=5,
        random_state=19,
    ).fit_predict(embeddings)
    expected = {
        "v_measure": metrics.v_measure_score(labels, predicted),
        "homogeneity": metrics.homogeneity_score(labels, predicted),
        "completeness": metrics.completeness_score(labels, predicted),
        "adjusted_rand": metrics.adjusted_rand_score(labels, predicted),
        "normalized_mutual_info": metrics.normalized_mutual_info_score(
            labels, predicted
        ),
    }
    assert actual == expected
