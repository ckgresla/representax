"""Compile and steady-state measurements for the generic train step."""

import time

import jax
import jax.numpy as jnp
import optax
import pytest

from representax.models import DenseEncoder
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import build_train_step, make_train_state


@pytest.mark.performance
def test_train_step_reports_compile_and_steady_state(capsys):
    model = DenseEncoder(32, 16, key=jax.random.key(0))
    task = MNRTask(scale=10.0)
    optimizer = optax.adamw(learning_rate=1e-3)
    state = make_train_state(model, optimizer)
    step = build_train_step(task, optimizer)
    values = jax.random.normal(jax.random.key(1), (32, 32))
    batch = retrieval_batch(
        query=values,
        document=values,
        positive_mask=jnp.eye(32, dtype=jnp.bool_),
    )

    started = time.perf_counter()
    result = step(state, batch, jax.random.key(2))
    result.metrics.loss.block_until_ready()
    compile_seconds = time.perf_counter() - started

    started = time.perf_counter()
    iterations = 10
    for index in range(iterations):
        result = step(result.state, batch, jax.random.key(index + 3))
    result.metrics.loss.block_until_ready()
    steady_seconds = time.perf_counter() - started

    print(
        {
            "compile_plus_first_step_seconds": compile_seconds,
            "steady_state_step_seconds": steady_seconds / iterations,
        }
    )
    assert compile_seconds > 0
    assert steady_seconds > 0
    assert "compile_plus_first_step_seconds" in capsys.readouterr().out
