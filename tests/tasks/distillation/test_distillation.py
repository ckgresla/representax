"""Native embedding, margin, and distribution distillation contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.models import DenseEncoder
from representax.tasks import build_task
from representax.tasks.distillation import (
    DistributionDistillationConfig,
    DistributionDistillationTask,
    DistributionKLLossConfig,
    EmbeddingDistillationConfig,
    EmbeddingDistillationLossConfig,
    EmbeddingDistillationTask,
    MarginDistillationConfig,
    MarginDistillationTask,
    MarginMSELossConfig,
    distribution_distillation_batch,
    distribution_kl_loss_terms,
    embedding_distillation_batch,
    embedding_distillation_loss_terms,
    margin_distillation_batch,
    margin_mse_loss_terms,
)
from representax.train import build_train_step, init_train_state


def _examples():
    rng = np.random.default_rng(41)
    return jnp.asarray(rng.normal(size=(5, 4)), dtype=jnp.float32)


def test_embedding_batch_normalizes_broadcast_and_per_column_targets():
    first = _examples()
    second = first + 0.1
    broadcast_targets = jnp.ones((5, 3))
    per_column_targets = jnp.ones((5, 2, 3))

    broadcast = embedding_distillation_batch(
        inputs=(first, second),
        teacher_embeddings=broadcast_targets,
    )
    per_column = embedding_distillation_batch(
        inputs=(first, second),
        teacher_embeddings=per_column_targets,
    )

    assert broadcast.teacher_embeddings.shape == (2, 5, 3)
    assert per_column.teacher_embeddings.shape == (2, 5, 3)
    with pytest.raises(ValueError, match="batch, column"):
        embedding_distillation_batch(
            inputs=(first, second),
            teacher_embeddings=jnp.ones((5, 3, 3)),
        )


def test_margin_batch_accepts_margins_or_canonical_teacher_scores():
    examples = _examples()
    margins = jnp.asarray([0.3, -0.1, 0.2, 0.4, 0.0])
    scores = jnp.stack(
        (jnp.asarray([0.5, 0.4, 0.3, 0.2, 0.1]), jnp.zeros((5,))),
        axis=1,
    )
    direct = margin_distillation_batch(
        query=examples,
        positive=examples + 0.1,
        negatives=(examples - 0.2,),
        teacher_margins=margins,
    )
    canonical = margin_distillation_batch(
        query=examples,
        positive=examples + 0.1,
        negatives=(examples - 0.2,),
        teacher_scores=scores,
    )

    assert direct.teacher_margins.shape == (5, 1)
    np.testing.assert_allclose(canonical.teacher_margins[:, 0], scores[:, 0])


def test_distribution_batch_requires_one_score_per_candidate():
    examples = _examples()
    batch = distribution_distillation_batch(
        query=examples,
        candidates=(examples + 0.1, examples - 0.1),
        teacher_scores=jnp.ones((5, 2)),
    )

    assert batch.teacher_scores.shape == (5, 2)
    with pytest.raises(ValueError, match="one column per candidate"):
        distribution_distillation_batch(
            query=examples,
            candidates=(examples + 0.1, examples - 0.1),
            teacher_scores=jnp.ones((5, 3)),
        )


@pytest.mark.parametrize("distance", ("mse", "l2", "cosine"))
def test_embedding_distillation_is_zero_for_matching_targets(distance):
    students = jnp.stack((_examples(), _examples() + 0.2))
    terms = embedding_distillation_loss_terms(
        students,
        students,
        distance=distance,
    )

    np.testing.assert_allclose(terms.loss, 0.0, atol=1e-6)


def test_embedding_distillation_masks_padding_rows_and_averages_columns():
    students = jnp.stack((_examples(), _examples() + 0.2))
    teachers = jnp.zeros_like(students)
    valid = jnp.asarray([True, True, True, True, False])
    terms = embedding_distillation_loss_terms(
        students,
        teachers,
        valid=valid,
        distance="mse",
    )
    expected = jnp.mean(jnp.square(students[:, :4]))

    np.testing.assert_allclose(terms.loss, expected, rtol=1e-6)


@pytest.mark.parametrize("similarity", ("dot", "cosine"))
def test_margin_mse_matches_direct_positive_minus_negative_scores(similarity):
    query = _examples()
    positive = query + 0.1
    negatives = jnp.stack((query - 0.2, query[::-1]))
    zero_targets = jnp.zeros((5, 2))
    terms = margin_mse_loss_terms(
        query,
        positive,
        negatives,
        zero_targets,
        similarity=similarity,
    )

    np.testing.assert_allclose(
        terms.loss,
        jnp.mean(jnp.square(terms.predicted_margins)),
        rtol=1e-6,
    )


def test_distribution_kl_is_zero_when_teacher_scores_match_student_scores():
    query = _examples()
    candidates = jnp.stack((query + 0.1, query - 0.2, query[::-1]))
    student_scores = jnp.stack(
        tuple(jnp.sum(query * candidate, axis=1) for candidate in candidates),
        axis=1,
    )
    terms = distribution_kl_loss_terms(
        query,
        candidates,
        student_scores,
        temperature=2.0,
    )

    np.testing.assert_allclose(terms.loss, 0.0, atol=2e-7)


def test_registry_builds_distillation_tasks_with_task_owned_routes():
    embedding = build_task(
        EmbeddingDistillationConfig(routes=(Route.QUERY, Route.DOCUMENT)),
        EmbeddingDistillationLossConfig(distance="l2"),
    )
    margin = build_task(
        MarginDistillationConfig(
            query_route=Route.QUERY,
            document_route=Route.DOCUMENT,
        ),
        MarginMSELossConfig(similarity="cosine"),
    )
    distribution = build_task(
        DistributionDistillationConfig(),
        DistributionKLLossConfig(temperature=2.0),
    )

    assert isinstance(embedding, EmbeddingDistillationTask)
    assert embedding.routes == (Route.QUERY, Route.DOCUMENT)
    assert embedding.distance == "l2"
    assert isinstance(margin, MarginDistillationTask)
    assert margin.similarity == "cosine"
    assert isinstance(distribution, DistributionDistillationTask)
    assert distribution.temperature == 2.0


@pytest.mark.parametrize("task_kind", ("embedding", "margin", "distribution"))
def test_distillation_tasks_run_through_the_generic_compiled_train_step(task_kind):
    examples = _examples()
    model = DenseEncoder(4, 4, key=jax.random.key(7), normalize=False)
    optimizer = optax.adamw(1e-2)
    state = init_train_state(model, optimizer)
    if task_kind == "embedding":
        task = EmbeddingDistillationTask(distance="mse")
        batch = embedding_distillation_batch(
            inputs=(examples,),
            teacher_embeddings=np.asarray(examples),
        )
    elif task_kind == "margin":
        task = MarginDistillationTask()
        batch = margin_distillation_batch(
            query=examples,
            positive=examples + 0.1,
            negatives=(examples - 0.2, examples[::-1]),
            teacher_margins=jnp.zeros((5, 2)),
        )
    else:
        task = DistributionDistillationTask(temperature=2.0)
        batch = distribution_distillation_batch(
            query=examples,
            candidates=(examples + 0.1, examples - 0.2),
            teacher_scores=jnp.zeros((5, 2)),
        )

    result = build_train_step(task, optimizer)(state, batch, None)

    assert result.state.step == 1
    assert result.metrics.numeric_finite
    assert not result.metrics.skipped_update
