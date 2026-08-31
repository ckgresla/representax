"""Shared offline and in-training evaluation tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.evaluation import LossEvaluator
from representax.models import DenseEncoder
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import EvaluationRunner, evaluate
from representax.train.evaluation import _batch_size


class _ModelNativeBatch(eqx.Module):
    inputs: jax.Array
    auxiliary: jax.Array
    valid: jax.Array


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


def test_offline_evaluation_uses_example_weighted_eval_metrics():
    model = DenseEncoder(2, 2, key=jax.random.key(5))
    task = CosineRegressionTask()
    expected = [float(task.loss(model, batch).loss) for batch in _batches()]

    result = evaluate(model, task, _batches())

    assert result.batches == 2
    assert result.examples == 3
    np.testing.assert_allclose(
        result.metrics["eval/loss"],
        (2 * expected[0] + expected[1]) / 3,
        rtol=1e-6,
    )


def test_evaluation_runner_reuses_the_compiled_shape_signature():
    model = DenseEncoder(2, 2, key=jax.random.key(7))
    runner = EvaluationRunner(LossEvaluator(CosineRegressionTask()))

    first = runner.run(model, _batches(), max_batches=1)
    second = runner.run(model, _batches(), max_batches=1)

    assert first.compilation_seconds > 0
    assert second.compilation_seconds == 0
    assert second.metrics == first.metrics


def test_evaluation_batch_size_uses_the_explicit_task_example_axis():
    batch = _ModelNativeBatch(
        inputs=jnp.zeros((2, 4)),
        auxiliary=jnp.zeros((7, 3)),
        valid=jnp.ones((2,), dtype=jnp.bool_),
    )

    assert _batch_size(batch) == 2
