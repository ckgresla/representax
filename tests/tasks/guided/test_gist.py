from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.models import DenseEncoder
from representax.tasks.guided import GISTTask, gist_batch, gist_loss_terms
from representax.tasks.modifiers import MatryoshkaTask
from representax.train import GradCache, build_train_step, init_train_state


def _arrays():
    keys = jax.random.split(jax.random.key(7), 6)
    student = tuple(jax.random.normal(key, (7, 9)) for key in keys[:3])
    guide = tuple(jax.random.normal(key, (7, 5)) for key in keys[3:])
    return student, guide


def test_score_row_chunking_preserves_gist_values_and_gradients():
    student, guide = _arrays()

    def objective(values, row_chunk_size):
        return gist_loss_terms(
            values,
            guide,
            temperature=0.07,
            margin_strategy="relative",
            margin=0.1,
            row_chunk_size=row_chunk_size,
        ).loss

    with jax.default_matmul_precision("highest"):
        direct, direct_gradients = jax.value_and_grad(
            lambda values: objective(values, None)
        )(student)
        cached, cached_gradients = jax.value_and_grad(
            lambda values: objective(values, 3)
        )(student)
    np.testing.assert_allclose(cached, direct, rtol=2e-6, atol=2e-6)
    for actual, expected in zip(cached_gradients, direct_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


def test_gist_task_runs_through_the_generic_compiled_step():
    student, guide = _arrays()
    batch = gist_batch(
        anchor=student[0],
        positive=student[1],
        negatives=(student[2],),
        guide_anchor=guide[0],
        guide_positive=guide[1],
        guide_negatives=(guide[2],),
    )
    model = DenseEncoder(9, 8, key=jax.random.key(8))
    task = GISTTask(temperature=0.07, margin=0.1)
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)

    result = build_train_step(task, optimizer)(state, batch, jax.random.key(9))

    assert bool(result.metrics.numeric_finite)
    assert int(result.state.step) == 1
    updated_model = cast(DenseEncoder, result.state.model)
    assert not jnp.array_equal(
        updated_model.projection.weight,
        model.projection.weight,
    )

    cached = build_train_step(
        task,
        optimizer,
        execution=GradCache(
            query_chunk_size=3,
            document_chunk_size=2,
            loss_row_chunk_size=3,
        ),
    )(state, batch, jax.random.key(9))
    assert bool(cached.metrics.numeric_finite)
    np.testing.assert_allclose(
        cached.metrics.loss,
        result.metrics.loss,
        rtol=1e-4,
        atol=1e-4,
    )

    matryoshka = MatryoshkaTask(task, (8, 4), weights=(1.0, 0.5))
    direct_matryoshka = build_train_step(matryoshka, optimizer)(
        state,
        batch,
        jax.random.key(10),
    )
    cached_matryoshka = build_train_step(
        matryoshka,
        optimizer,
        execution=GradCache(
            query_chunk_size=3,
            document_chunk_size=2,
            loss_row_chunk_size=3,
        ),
    )(state, batch, jax.random.key(10))
    np.testing.assert_allclose(
        cached_matryoshka.metrics.loss,
        direct_matryoshka.metrics.loss,
        rtol=1e-4,
        atol=1e-4,
    )
