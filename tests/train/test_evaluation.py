"""Shared offline and in-training evaluation tests."""

import jax
import jax.numpy as jnp
import numpy as np

from representax.models import DenseEncoder
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import EvaluationRunner, evaluate


def _batches():
    return [
        pairwise_batch(
            left=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
            right=jnp.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=jnp.float32),
            labels=jnp.asarray([0.9, 0.8], dtype=jnp.float32),
        ),
        pairwise_batch(
            left=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
            right=jnp.asarray([[0.8, 0.9]], dtype=jnp.float32),
            labels=jnp.asarray([0.95], dtype=jnp.float32),
        ),
    ]


def test_offline_evaluation_uses_example_weighted_valid_metrics():
    model = DenseEncoder(2, 2, key=jax.random.key(5))
    task = CosineRegressionTask()
    expected = [float(task.loss(model, batch).loss) for batch in _batches()]

    result = evaluate(model, task, _batches())

    assert result.batches == 2
    assert result.examples == 3
    np.testing.assert_allclose(
        result.metrics["valid/loss"],
        (2 * expected[0] + expected[1]) / 3,
        rtol=1e-6,
    )


def test_evaluation_runner_reuses_the_compiled_shape_signature():
    model = DenseEncoder(2, 2, key=jax.random.key(7))
    runner = EvaluationRunner(CosineRegressionTask())

    first = runner.run(model, _batches(), max_batches=1)
    second = runner.run(model, _batches(), max_batches=1)

    assert first.compilation_seconds > 0
    assert second.compilation_seconds == 0
    assert second.metrics == first.metrics
