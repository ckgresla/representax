"""Native labeled-pair loss and task contracts."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.models import DenseEncoder
from representax.tasks import build_task
from representax.tasks.pairwise import (
    AngleConfig,
    AnglETask,
    ContrastiveConfig,
    ContrastiveTask,
    CoSENTConfig,
    CoSENTTask,
    CosineRegressionConfig,
    CosineRegressionTask,
    PairwiseConfig,
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
    pairwise_angle_similarity,
    pairwise_batch,
)
from representax.train import build_train_step, init_train_state


def _embeddings():
    left = jnp.asarray(
        [[1.0, 0.0, 0.2], [0.0, 1.0, -0.1], [0.6, 0.4, 0.3], [-0.2, 0.7, 0.5]],
        dtype=jnp.float32,
    )
    right = jnp.asarray(
        [[0.9, 0.1, 0.1], [0.7, 0.3, 0.2], [0.5, 0.5, 0.2], [0.3, -0.4, 0.8]],
        dtype=jnp.float32,
    )
    return left, right


def test_pairwise_batch_validates_aligned_payload_rows():
    left, right = _embeddings()
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=jnp.asarray([1.0, 0.0, 0.7, 0.2]),
    )

    assert batch.labels.dtype == jnp.float32
    np.testing.assert_array_equal(batch.valid, np.ones((4,), dtype=np.bool_))

    with pytest.raises(ValueError, match="one row per pair"):
        pairwise_batch(left=left[:2], right=right, labels=batch.labels)


def test_cosine_regression_masks_padding_rows_without_changing_the_mean():
    left, right = _embeddings()
    labels = jnp.asarray([0.8, 0.1, 0.7, 99.0])
    valid = jnp.asarray([True, True, True, False])

    terms = cosine_regression_loss_terms(left, right, labels, valid=valid)
    expected = jnp.mean(jnp.square(terms.scores[:3] - labels[:3]))

    np.testing.assert_allclose(terms.loss, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("metric", ("cosine", "euclidean", "manhattan"))
def test_contrastive_loss_prefers_close_positives_and_distant_negatives(metric):
    left, right = _embeddings()
    labels = jnp.asarray([1.0, 0.0, 1.0, 0.0])
    ordinary = contrastive_loss_terms(
        left,
        right,
        labels,
        metric=metric,
        margin=0.5,
    )
    swapped = contrastive_loss_terms(
        left,
        right[::-1],
        labels,
        metric=metric,
        margin=0.5,
    )

    assert jnp.isfinite(ordinary.loss)
    assert ordinary.row_losses.shape == labels.shape
    assert ordinary.loss != swapped.loss


def test_online_contrastive_selects_a_static_mask_of_hard_pairs():
    left = jnp.tile(jnp.asarray([[1.0, 0.0]], dtype=jnp.float32), (4, 1))
    right = jnp.asarray(
        [
            [0.1, 0.9949874],
            [0.8, 0.6],
            [0.9, 0.4358899],
            [0.0, 1.0],
        ]
    )
    labels = jnp.asarray([1.0, 0.0, 1.0, 0.0])

    terms = jax.jit(online_contrastive_loss_terms)(left, right, labels)

    assert terms.selected.shape == labels.shape
    assert jnp.any(terms.selected)
    assert jnp.isfinite(terms.loss)


def test_cosent_orders_pair_scores_and_ignores_label_ties():
    left = jnp.tile(jnp.asarray([[1.0, 0.0]]), (3, 1))
    good = jnp.asarray([[1.0, 0.0], [0.5, 0.8660254], [0.0, 1.0]])
    bad = good[::-1]
    labels = jnp.asarray([1.0, 0.5, 0.0])

    good_terms = pair_ranking_loss_terms(left, good, labels, scale=10.0)
    bad_terms = pair_ranking_loss_terms(left, bad, labels, scale=10.0)

    assert good_terms.loss < bad_terms.loss
    assert not jnp.any(jnp.diag(good_terms.ordered_pairs))


def test_angle_similarity_supports_odd_representation_dimensions():
    left, right = _embeddings()
    values = pairwise_angle_similarity(left, right)

    assert values.shape == (4,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.parametrize(
    ("loss_config", "task_type"),
    (
        (CosineRegressionConfig(), CosineRegressionTask),
        (ContrastiveConfig(mining="online"), ContrastiveTask),
        (CoSENTConfig(scale=7.0), CoSENTTask),
        (AngleConfig(scale=9.0), AnglETask),
    ),
)
def test_registry_builds_pairwise_losses_with_task_owned_routes(
    loss_config,
    task_type,
):
    task = build_task(
        PairwiseConfig(left_route=Route.QUERY, right_route=Route.DOCUMENT),
        loss_config,
    )

    assert isinstance(task, task_type)
    assert task.left_route is Route.QUERY
    assert task.right_route is Route.DOCUMENT


def test_pairwise_task_runs_through_the_generic_compiled_train_step():
    model = DenseEncoder(3, 4, key=jax.random.key(3))
    optimizer = optax.adamw(1e-2)
    task = CoSENTTask(scale=5.0)
    state = init_train_state(model, optimizer)
    left, right = _embeddings()
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=jnp.asarray([1.0, 0.0, 0.8, 0.2]),
    )

    result = build_train_step(task, optimizer)(state, batch, None)

    assert result.state.step == 1
    assert result.metrics.numeric_finite
    assert not result.metrics.skipped_update


def test_standard_contrastive_accumulation_matches_the_full_batch() -> None:
    model = DenseEncoder(3, 4, key=jax.random.key(13))
    optimizer = optax.sgd(1e-2)
    state = init_train_state(model, optimizer)
    left, right = _embeddings()
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=jnp.asarray([1.0, 0.0, 1.0, 0.0]),
    )
    task = ContrastiveTask()

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


def test_online_contrastive_rejects_inexact_accumulation() -> None:
    with pytest.raises(TypeError, match="does not support exact"):
        build_train_step(
            ContrastiveTask(online=True),
            optax.sgd(1e-3),
            gradient_accumulation_steps=2,
        )
