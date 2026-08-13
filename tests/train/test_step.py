"""Generic compiled training-step tests."""

import jax
import jax.numpy as jnp
import optax
import pytest

import representax as rx
from representax.tasks.retrieval import MNRTask, retrieval_batch


@pytest.mark.runtime
def test_compiled_retrieval_training_reduces_loss():
    model = rx.models.DenseEncoder(4, 3, key=jax.random.key(0))
    task = MNRTask(scale=5.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=0.03, weight_decay=0.0)
    state = rx.train.make_train_state(model, optimizer)
    step = rx.train.build_train_step(task, optimizer, max_grad_norm=1.0)
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
