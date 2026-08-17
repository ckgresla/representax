"""Generic compiled training-step tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models import DenseEncoder
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import build_train_step, init_train_state


@pytest.mark.runtime
def test_compiled_retrieval_training_reduces_loss():
    model = DenseEncoder(4, 3, key=jax.random.key(0))
    task = MNRTask(scale=5.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=0.03, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    step = build_train_step(task, optimizer, max_grad_norm=1.0)
    batch = retrieval_batch(
        query=jnp.eye(4, dtype=jnp.float32),
        document=jnp.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        positive_mask=jnp.eye(4, dtype=jnp.bool_),
    )

    first = step(state, batch, jax.random.key(1))
    result = first
    for index in range(1, 40):
        result = step(result.state, batch, jax.random.key(index + 1))

    assert result.metrics.loss < first.metrics.loss
    assert bool(result.metrics.numeric_finite)
    assert int(result.state.step) == 40


@pytest.mark.runtime
def test_nonfinite_forward_keeps_model_and_optimizer_state():
    model = DenseEncoder(2, 2, key=jax.random.key(7), normalize=False)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = init_train_state(model, optimizer)
    step = build_train_step(
        MNRTask(scale=5.0),
        optimizer,
        max_grad_norm=None,
        donate_state=False,
    )
    batch = retrieval_batch(
        query=jnp.asarray([[jnp.nan, 0.0]], dtype=jnp.float32),
        document=jnp.asarray([[1.0, 0.0]], dtype=jnp.float32),
        positive_mask=jnp.ones((1, 1), dtype=jnp.bool_),
    )

    result = step(state, batch, jax.random.key(8))

    assert not bool(result.metrics.numeric_finite)
    assert bool(result.metrics.skipped_update)
    assert float(result.metrics.update_global_norm) == 0.0
    assert int(result.state.step) == 0
    for actual, expected in zip(
        jax.tree.leaves(result.state.model),
        jax.tree.leaves(state.model),
        strict=True,
    ):
        if eqx.is_array(actual):
            np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        jax.tree.leaves(result.state.optimizer_state),
        jax.tree.leaves(state.optimizer_state),
        strict=True,
    ):
        if eqx.is_array(actual):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.runtime
def test_compiled_gradient_accumulation_matches_one_full_batch_update():
    model = DenseEncoder(4, 3, key=jax.random.key(19))
    task = CosineRegressionTask()
    optimizer = optax.adamw(learning_rate=3e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    batch = pairwise_batch(
        left=jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
        right=jnp.asarray(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.1, 0.8, 0.1, 0.0],
                [0.0, 0.1, 0.8, 0.1],
                [0.0, 0.0, 0.1, 0.9],
            ],
            dtype=jnp.float32,
        ),
        labels=jnp.asarray([0.9, 0.8, 0.7, 0.6], dtype=jnp.float32),
        valid=jnp.asarray([True, False, True, True]),
    )

    direct = build_train_step(task, optimizer, max_grad_norm=None)(state, batch, None)
    accumulated = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        gradient_accumulation_steps=2,
    )(state, batch, None)

    np.testing.assert_allclose(
        accumulated.metrics.loss,
        direct.metrics.loss,
        rtol=1e-6,
        atol=1e-7,
    )
    for name in direct.metrics.task:
        np.testing.assert_allclose(
            accumulated.metrics.task[name],
            direct.metrics.task[name],
            rtol=1e-6,
            atol=1e-7,
        )
    assert int(accumulated.state.step) == 1
    for actual, expected in zip(
        jax.tree.leaves(accumulated.state),
        jax.tree.leaves(direct.state),
        strict=True,
    ):
        if eqx.is_array(actual):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_gradient_accumulation_requires_divisible_array_batches():
    model = DenseEncoder(4, 3, key=jax.random.key(29))
    task = CosineRegressionTask()
    optimizer = optax.sgd(1e-3)
    state = init_train_state(model, optimizer)
    batch = pairwise_batch(
        left=jnp.ones((3, 4), dtype=jnp.float32),
        right=jnp.ones((3, 4), dtype=jnp.float32),
        labels=jnp.ones((3,), dtype=jnp.float32),
    )

    step = build_train_step(task, optimizer, gradient_accumulation_steps=2)

    with pytest.raises(ValueError, match="divisible"):
        step(state, batch, None)
