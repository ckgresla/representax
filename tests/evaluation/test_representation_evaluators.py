"""Canonical frozen-representation evaluator contracts."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.evaluation import (
    ClassificationProbeEvaluator,
    ClusteringEvaluator,
    EvaluationSplit,
    JEPARepresentationEvaluator,
    PairClassificationEvaluator,
    clustering_metrics,
    labeled_evaluation_batch,
    pair_classification_metrics,
)
from representax.tasks.classification import pair_classification_batch
from representax.tasks.triplet import labeled_examples_batch
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class IdentityEncoder(eqx.Module):
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route=None, key=None):
        del route, key
        return inputs.values


def encoder(dimension: int = 3) -> IdentityEncoder:
    return IdentityEncoder(
        EncoderMetadata(
            model_id="identity",
            revision="test",
            output_dimension=dimension,
            routes=frozenset({Route.GENERIC}),
            modalities=frozenset({Modality.TEXT}),
        )
    )


def test_pair_classification_selects_similarity_thresholds() -> None:
    left = np.asarray(
        ((1.0, 0.0), (0.9, 0.1), (1.0, 0.0), (0.0, 1.0)),
        dtype=np.float32,
    )
    right = np.asarray(
        ((1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        dtype=np.float32,
    )
    labels = np.asarray((1, 1, 0, 0), dtype=np.int32)
    metrics = pair_classification_metrics(
        left, right, labels, similarity_functions=("cosine", "euclidean")
    )
    assert metrics["average_precision_max"] == 1.0
    assert metrics["accuracy_max"] == 1.0
    assert metrics["f1_max"] == 1.0

    batch = pair_classification_batch(
        left=Inputs(jnp.asarray(left)),
        right=Inputs(jnp.asarray(right)),
        labels=jnp.asarray(labels),
    )
    result = EvaluationRunner(
        PairClassificationEvaluator(similarity_functions=("cosine",))
    ).run(encoder(2), (batch,))
    assert result.metrics["valid/pair_classification/average_precision_max"] == 1.0


def _probe_batches() -> tuple:
    train = labeled_evaluation_batch(
        examples=Inputs(
            jnp.asarray(
                ((2.0, 0.0), (1.5, 0.1), (0.0, 2.0), (0.1, 1.5)),
                dtype=jnp.float32,
            )
        ),
        labels=jnp.asarray((0, 0, 1, 1)),
        split=EvaluationSplit.TRAIN,
    )
    validation = labeled_evaluation_batch(
        examples=Inputs(jnp.asarray(((1.8, 0.1), (0.1, 1.8)))),
        labels=jnp.asarray((0, 1)),
        split=EvaluationSplit.VALIDATION,
    )
    test = labeled_evaluation_batch(
        examples=Inputs(jnp.asarray(((1.7, 0.2), (0.2, 1.7)))),
        labels=jnp.asarray((0, 1)),
        split=EvaluationSplit.TEST,
    )
    return train, validation, test


def test_classification_probe_fits_only_host_side_probe() -> None:
    evaluator = ClassificationProbeEvaluator(
        inverse_regularization=(0.1, 1.0),
        max_iterations=200,
    )
    result = EvaluationRunner(evaluator).run(encoder(2), _probe_batches())
    assert result.metrics["valid/classification_probe/accuracy"] == 1.0
    assert result.metrics["valid/classification_probe/f1_macro"] == 1.0
    assert result.metrics[
        "valid/classification_probe/selected_inverse_regularization"
    ] in (0.1, 1.0)


def test_clustering_reports_permutation_invariant_geometry() -> None:
    values = np.asarray(
        (
            (2.0, 0.0),
            (1.9, 0.1),
            (2.1, -0.1),
            (0.0, 2.0),
            (0.1, 1.9),
            (-0.1, 2.1),
        ),
        dtype=np.float32,
    )
    labels = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int32)
    metrics = clustering_metrics(
        values,
        labels,
        batch_size=6,
        n_init=3,
        seed=7,
    )
    assert metrics["v_measure"] == 1.0
    batch = labeled_examples_batch(
        examples=Inputs(jnp.asarray(values)), labels=jnp.asarray(labels)
    )
    result = EvaluationRunner(ClusteringEvaluator(batch_size=6, n_init=3, seed=7)).run(
        encoder(2), (batch,)
    )
    assert result.metrics["valid/clustering/v_measure"] == 1.0


def test_jepa_representation_combines_transfer_knn_and_health() -> None:
    evaluator = JEPARepresentationEvaluator(
        inverse_regularization=(0.1, 1.0),
        max_iterations=200,
        neighbors=1,
    )
    result = EvaluationRunner(evaluator).run(encoder(2), _probe_batches())
    assert result.metrics["valid/jepa_representation/linear_probe_accuracy"] == 1.0
    assert result.metrics["valid/jepa_representation/knn_accuracy"] == 1.0
    assert result.metrics["valid/jepa_representation/effective_rank"] > 1.0
    assert result.metrics["valid/jepa_representation/feature_std_min"] > 0.0
