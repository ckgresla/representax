"""Generic compiled training-step tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models import DenseEncoder
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
