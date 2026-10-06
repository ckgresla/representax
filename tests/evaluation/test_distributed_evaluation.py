"""Annotation-only evaluator execution over arbitrary single-host data meshes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from representax.core import EncoderMetadata, Modality, Route
from representax.evaluation import (
    ClassificationEvaluator,
    ClassificationProbeEvaluator,
    ClusteringEvaluator,
    EvaluationSplit,
    JEPARepresentationEvaluator,
    PairClassificationEvaluator,
    labeled_evaluation_batch,
)
from representax.tasks.classification import pair_classification_batch
from representax.tasks.cross_encoder import PointwiseBatch
from representax.tasks.triplet import labeled_examples_batch
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class Scorer(eqx.Module):
    weight: jax.Array

    def logits(self, inputs: Inputs, *, key=None):
        del key
        return inputs.values @ self.weight


class Encoder(eqx.Module):
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route=None, key=None):
        del route, key
        return inputs.values


def encoder() -> Encoder:
    return Encoder(
        EncoderMetadata(
            model_id="distributed-evaluation",
            revision="test",
            output_dimension=2,
            routes=frozenset({Route.GENERIC}),
            modalities=frozenset({Modality.TEXT}),
        )
    )


@pytest.mark.distributed
def test_evaluation_is_exact_over_one_two_and_four_device_data_meshes() -> None:
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four physical or virtual JAX devices")
    model = Scorer(jnp.asarray((1.0, -1.0), dtype=jnp.float32))
    batch = PointwiseBatch(
        inputs=Inputs(
            jnp.asarray(
                tuple((2.0, 0.0) if index % 2 else (0.0, 2.0) for index in range(8))
            )
        ),
        labels=jnp.asarray(tuple(index % 2 for index in range(8)), dtype=jnp.int32),
        valid=jnp.ones((8,), dtype=jnp.bool_),
    )
    results = []
    for device_count in (1, 2, 4):
        mesh = jax.make_mesh(
            (device_count,),
            ("data",),
            devices=devices[:device_count],
        )
        sharding = NamedSharding(mesh, P("data"))
        placements = []

        def place_batch(tree, *, sharding=sharding, placements=placements):
            placed = jax.tree.map(
                lambda value: (
                    jax.device_put(value, sharding)
                    if isinstance(value, jax.Array)
                    else value
                ),
                tree,
            )
            placements.append(placed.inputs.values.sharding)
            return placed

        result = EvaluationRunner(ClassificationEvaluator()).run(
            model,
            (batch,),
            place_batch=place_batch,
        )
        results.append(result.metrics)
        assert placements == [sharding]
    assert results[0] == results[1] == results[2]
    assert results[0]["valid/classification/accuracy"] == 1.0
    np.testing.assert_allclose(
        results[0]["valid/classification/f1_macro"],
        1.0,
    )


@pytest.mark.distributed
def test_representation_evaluators_are_exact_over_data_meshes() -> None:
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires four physical or virtual JAX devices")
    values = jnp.asarray(
        ((2.0, 0.0), (1.8, 0.1), (0.0, 2.0), (0.1, 1.8)),
        dtype=jnp.float32,
    )
    labels = jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
    cases = (
        (
            PairClassificationEvaluator(similarity_functions=("cosine",)),
            (
                pair_classification_batch(
                    left=Inputs(values),
                    right=Inputs(values[jnp.asarray((0, 1, 0, 1))]),
                    labels=jnp.asarray((1, 1, 0, 0), dtype=jnp.int32),
                ),
            ),
        ),
        (
            ClassificationProbeEvaluator(
                inverse_regularization=(1.0,), max_iterations=100
            ),
            tuple(
                labeled_evaluation_batch(
                    examples=Inputs(values),
                    labels=labels,
                    split=split,
                )
                for split in EvaluationSplit
            ),
        ),
        (
            ClusteringEvaluator(batch_size=4, n_init=1),
            (labeled_examples_batch(examples=Inputs(values), labels=labels),),
        ),
        (
            JEPARepresentationEvaluator(
                inverse_regularization=(1.0,),
                max_iterations=100,
                neighbors=1,
                query_batch_size=1,
            ),
            tuple(
                labeled_evaluation_batch(
                    examples=Inputs(values),
                    labels=labels,
                    split=split,
                )
                for split in EvaluationSplit
            ),
        ),
    )
    for evaluator, batches in cases:
        results = []
        for device_count in (1, 2, 4):
            mesh = jax.make_mesh(
                (device_count,),
                ("data",),
                devices=devices[:device_count],
            )
            sharding = NamedSharding(mesh, P("data"))

            def place_batch(tree, *, sharding=sharding):
                return jax.tree.map(
                    lambda value: (
                        jax.device_put(value, sharding)
                        if isinstance(value, jax.Array)
                        else value
                    ),
                    tree,
                )

            results.append(
                EvaluationRunner(evaluator)
                .run(
                    encoder(),
                    batches,
                    place_batch=place_batch,
                )
                .metrics
            )
        assert results[0] == results[1] == results[2]
