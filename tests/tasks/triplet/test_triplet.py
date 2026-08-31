"""Native explicit and in-batch-mined triplet contracts."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.models import DenseEncoder
from representax.tasks import build_task
from representax.tasks.triplet import (
    BatchHardSoftMarginLossConfig,
    BatchTripletLossConfig,
    BatchTripletTask,
    ExplicitTripletConfig,
    ExplicitTripletLossConfig,
    ExplicitTripletTask,
    LabeledExamplesConfig,
    batch_all_triplet_loss_terms,
    batch_hard_triplet_loss_terms,
    batch_semi_hard_triplet_loss_terms,
    explicit_triplet_batch,
    explicit_triplet_loss_terms,
    labeled_examples_batch,
    pairwise_triplet_distances,
    triplet_masks,
)
from representax.train import build_train_step, init_train_state


def _examples():
    return jnp.asarray(
        [
            [1.0, 0.0, 0.1, 0.0],
            [0.9, 0.1, 0.0, 0.1],
            [0.0, 1.0, 0.1, 0.0],
            [0.1, 0.9, 0.0, 0.1],
            [-0.8, -0.2, 0.1, 0.0],
            [-0.9, -0.1, 0.0, 0.1],
        ],
        dtype=jnp.float32,
    )


def _labels():
    return jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)


def test_triplet_batches_validate_their_distinct_data_contracts():
    examples = _examples()
    explicit = explicit_triplet_batch(
        anchor=examples[:2],
        positive=examples[2:4],
        negative=examples[4:6],
    )
    labeled = labeled_examples_batch(examples=examples, labels=_labels())

    np.testing.assert_array_equal(explicit.valid, np.ones((2,), dtype=np.bool_))
    assert labeled.labels.dtype == jnp.int32

    with pytest.raises(ValueError, match="one row per triplet"):
        explicit_triplet_batch(
            anchor=examples[:2],
            positive=examples[:3],
            negative=examples[:2],
        )
    with pytest.raises(TypeError, match="integer dtype"):
        labeled_examples_batch(examples=examples, labels=jnp.ones((6,)))


@pytest.mark.parametrize("metric", ("cosine", "euclidean", "manhattan"))
def test_explicit_triplet_loss_prefers_a_closer_positive(metric):
    anchor = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    close = jnp.asarray([[0.9, 0.1], [0.1, 0.9]])
    far = jnp.asarray([[-1.0, 0.0], [0.0, -1.0]])

    good = explicit_triplet_loss_terms(
        anchor,
        close,
        far,
        metric=metric,
        margin=0.5,
    )
    bad = explicit_triplet_loss_terms(
        anchor,
        far,
        close,
        metric=metric,
        margin=0.5,
    )

    assert good.loss < bad.loss


def test_explicit_triplet_loss_excludes_padding_rows_from_reduction():
    examples = _examples()
    terms = explicit_triplet_loss_terms(
        examples[:2],
        examples[2:4],
        examples[4:6],
        valid=jnp.asarray([True, False]),
        margin=0.5,
    )

    np.testing.assert_allclose(terms.loss, terms.row_losses[0], rtol=1e-6)


@pytest.mark.parametrize("metric", ("cosine", "euclidean", "squared_euclidean"))
def test_pairwise_triplet_distance_matrix_is_symmetric_with_zero_diagonal(metric):
    distances = pairwise_triplet_distances(_examples(), metric=metric)

    np.testing.assert_allclose(distances, distances.T, atol=1e-6)
    np.testing.assert_allclose(jnp.diag(distances), 0.0, atol=1e-6)


def test_triplet_masks_require_distinct_same_label_positives_and_other_labels():
    positive, negative, triplets = triplet_masks(_labels())

    assert positive[0, 1]
    assert not positive[0, 0]
    assert negative[0, 2]
    assert not negative[0, 1]
    assert triplets[0, 1, 2]
    assert not triplets[0, 1, 0]


@pytest.mark.parametrize("mining", ("all", "hard", "hard_soft_margin", "semi_hard"))
def test_every_batch_mining_policy_is_jittable_and_finite(mining):
    examples = _examples()
    labels = _labels()

    if mining == "all":
        terms = jax.jit(
            lambda value: batch_all_triplet_loss_terms(value, labels, margin=5.0)
        )(examples)
    elif mining == "semi_hard":
        terms = jax.jit(
            lambda value: batch_semi_hard_triplet_loss_terms(value, labels, margin=5.0)
        )(examples)
    else:
        terms = jax.jit(
            lambda value: batch_hard_triplet_loss_terms(
                value,
                labels,
                margin=5.0,
                soft_margin=mining == "hard_soft_margin",
            )
        )(examples)

    assert jnp.isfinite(terms.loss)
    assert jnp.any(terms.selected)


def test_batch_mining_ignores_invalid_examples():
    examples = _examples()
    labels = _labels()
    valid = jnp.asarray([True, True, True, True, False, False])

    padded = batch_all_triplet_loss_terms(
        examples,
        labels,
        valid=valid,
        margin=0.5,
    )
    unpadded = batch_all_triplet_loss_terms(
        examples[:4],
        labels[:4],
        margin=0.5,
    )

    np.testing.assert_allclose(padded.loss, unpadded.loss, rtol=1e-6, atol=1e-7)


def test_registry_builds_explicit_and_mined_triplets_with_task_owned_routes():
    explicit = build_task(
        ExplicitTripletConfig(
            anchor_route=Route.QUERY,
            positive_route=Route.DOCUMENT,
            negative_route=Route.DOCUMENT,
        ),
        ExplicitTripletLossConfig(distance="cosine", margin=0.5),
    )
    mined = build_task(
        LabeledExamplesConfig(route=Route.QUERY),
        BatchTripletLossConfig(mining="semi_hard", margin=0.5),
    )
    soft = build_task(
        LabeledExamplesConfig(),
        BatchHardSoftMarginLossConfig(),
    )

    assert isinstance(explicit, ExplicitTripletTask)
    assert explicit.anchor_route is Route.QUERY
    assert explicit.positive_route is Route.DOCUMENT
    assert isinstance(mined, BatchTripletTask)
    assert mined.mining == "semi_hard"
    assert mined.route is Route.QUERY
    assert soft.mining == "hard_soft_margin"
    assert soft.margin is None


@pytest.mark.parametrize("task_kind", ("explicit", "mined"))
def test_triplet_tasks_run_through_the_generic_compiled_train_step(task_kind):
    examples = _examples()
    model = DenseEncoder(4, 5, key=jax.random.key(3))
    optimizer = optax.adamw(1e-2)
    state = init_train_state(model, optimizer)
    if task_kind == "explicit":
        task = ExplicitTripletTask(distance="cosine", margin=0.5)
        batch = explicit_triplet_batch(
            anchor=examples[:2],
            positive=examples[2:4],
            negative=examples[4:6],
        )
    else:
        task = BatchTripletTask(mining="hard", margin=0.5)
        batch = labeled_examples_batch(examples=examples, labels=_labels())

    result = build_train_step(task, optimizer)(state, batch, None)

    assert result.state.step == 1
    assert result.metrics.numeric_finite
    assert not result.metrics.skipped_update


def test_explicit_triplet_accumulation_matches_the_full_batch() -> None:
    examples = _examples()[:4]
    batch = explicit_triplet_batch(
        anchor=examples,
        positive=examples + 0.1,
        negative=examples[::-1],
    )
    task = ExplicitTripletTask(distance="cosine", margin=0.5)
    model = DenseEncoder(4, 5, key=jax.random.key(23))
    optimizer = optax.sgd(1e-2)
    state = init_train_state(model, optimizer)

    direct = build_train_step(task, optimizer, max_grad_norm=None)(state, batch, None)
    accumulated = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        gradient_accumulation_steps=2,
    )(state, batch, None)

    np.testing.assert_allclose(accumulated.metrics.loss, direct.metrics.loss, rtol=1e-6)
    np.testing.assert_allclose(
        cast(DenseEncoder, accumulated.state.model).projection.weight,
        cast(DenseEncoder, direct.state.model).projection.weight,
        rtol=2e-5,
        atol=2e-6,
    )
