"""Annotation-only evaluator execution over arbitrary single-host data meshes."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from representax.evaluation import ClassificationEvaluator
from representax.tasks.cross_encoder import PointwiseBatch
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class Scorer(eqx.Module):
    weight: jax.Array

    def logits(self, inputs: Inputs, *, key=None):
        del key
        return inputs.values @ self.weight


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
