"""Fast contracts for scorer tasks and their registered construction."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array, Float

from representax.core import Scorer, score_logits
from representax.models.processing import Processor
from representax.tasks import build_task
from representax.tasks.cross_encoder import (
    BinaryCrossEntropyConfig,
    CrossEntropyConfig,
    CrossInBatchRankingConfig,
    CrossMNRCollator,
    CrossMNRConfig,
    CrossMNRTask,
    LambdaLossConfig,
    ListMLEConfig,
    ListNetConfig,
    ListwiseRankingBatch,
    ListwiseRankingCollator,
    ListwiseRankingConfig,
    ListwiseScoringTask,
    MarginMSEConfig,
    PairwiseRankingBatch,
    PairwiseRankingCollator,
    PairwiseRankingConfig,
    PointwiseBatch,
    PointwiseCollator,
    PointwiseScoringConfig,
    PointwiseScoringTask,
    RankNetConfig,
    ScoreMSEConfig,
)
from representax.train.grad_cache import GradCache
from representax.train.step import build_train_step, init_train_state


class _Inputs(eqx.Module):
    values: jax.Array


class _Scorer(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def logits(self, inputs: _Inputs, *, key=None):
        del key
        return inputs.values @ self.weight + self.bias


def _scorer(outputs: int = 1) -> _Scorer:
    return _Scorer(
        weight=jnp.arange(3 * outputs, dtype=jnp.float32).reshape(3, outputs) / 9,
        bias=jnp.linspace(-0.2, 0.2, outputs),
    )


def _processor() -> Processor:
    def process(artifacts, *, route, seed):
        del route, seed
        return _Inputs(
            jnp.asarray(
                [
                    (len(str(left)), len(str(right)), index)
                    for index, (left, right) in enumerate(artifacts)
                ],
                dtype=np.float32,
            )
        )

    return Processor(process=process, contract={"kind": "test-pair-processor"})


def test_scorer_contract_preserves_raw_output_rank() -> None:
    model = _scorer(2)
    assert isinstance(model, Scorer)
    actual = score_logits(model, _Inputs(jnp.ones((4, 3))))
    assert actual.shape == (4, 2)


@pytest.mark.parametrize(
    ("loss", "expected_type"),
    (
        (BinaryCrossEntropyConfig(), PointwiseScoringTask),
        (CrossEntropyConfig(), PointwiseScoringTask),
        (ScoreMSEConfig(), PointwiseScoringTask),
    ),
)
def test_pointwise_losses_build_from_the_registry(loss, expected_type) -> None:
    task = build_task(PointwiseScoringConfig(), loss)
    assert isinstance(task, expected_type)


@pytest.mark.parametrize(
    "loss",
    (RankNetConfig(), LambdaLossConfig(), ListNetConfig(), ListMLEConfig()),
)
def test_listwise_losses_build_from_the_registry(loss) -> None:
    task = build_task(ListwiseRankingConfig(), loss)
    assert isinstance(task, ListwiseScoringTask)


def test_margin_mse_builds_from_the_registry() -> None:
    task = build_task(PairwiseRankingConfig(), MarginMSEConfig())
    assert task.activation == "identity"


def test_cross_mnr_builds_for_direct_and_gradcache_execution() -> None:
    task = build_task(CrossInBatchRankingConfig(), CrossMNRConfig())
    assert isinstance(task, CrossMNRTask)


def test_task_collators_use_one_loaded_pair_processor() -> None:
    processor = _processor()
    pointwise = PointwiseCollator(processor=processor)(
        (
            {"query": "q0", "document": "d0", "label": 1.0},
            {"query": "q1", "document": "d1", "label": 0.0},
        )
    )
    assert pointwise.inputs.values.shape == (2, 3)

    pairwise = PairwiseRankingCollator(processor=processor)(
        (
            {
                "query": "q0",
                "positive": "p0",
                "negative": "n0",
                "margin": 0.7,
            },
        )
    )
    assert pairwise.positive.values.shape == pairwise.negative.values.shape == (1, 3)

    listwise = ListwiseRankingCollator(processor=processor, documents_per_query=3)(
        (
            {"query": "q0", "documents": ("a", "b"), "labels": (1.0, 0.0)},
            {
                "query": "q1",
                "documents": ("c", "d", "e"),
                "labels": (2.0, 1.0, 0.0),
            },
        )
    )
    assert listwise.inputs.values.shape == (2, 3, 3)
    np.testing.assert_array_equal(
        listwise.valid,
        ((True, True, False), (True, True, True)),
    )

    cross_mnr = CrossMNRCollator(processor=processor, hard_negatives_per_query=1)(
        (
            {"query": "q0", "positive": "p0", "negatives": ("n0",)},
            {"query": "q1", "positive": "p1", "negatives": ("n1",)},
        )
    )
    assert cross_mnr.inputs.values.shape == (2, 4, 3)
    np.testing.assert_array_equal(cross_mnr.positive_indices, (0, 2))


def test_cross_mnr_direct_and_cached_values_and_gradients_are_exact() -> None:
    batch = CrossMNRCollator(processor=_processor(), hard_negatives_per_query=1)(
        (
            {"query": "query-zero", "positive": "positive-0", "negatives": ("n0",)},
            {"query": "query-one", "positive": "positive-1", "negatives": ("n1",)},
        )
    )
    batch = jax.tree.map(jnp.asarray, batch)
    task = CrossMNRTask()
    model = _scorer()
    direct = eqx.filter_value_and_grad(
        lambda candidate: task.loss(candidate, batch).loss
    )
    cached = eqx.filter_value_and_grad(
        lambda candidate: (
            GradCache(query_chunk_size=3, score_chunk_size=3)
            .evaluate(task, candidate, batch, key=None)
            .loss
        )
    )
    direct_loss, direct_gradient = direct(model)
    cached_loss, cached_gradient = cached(model)
    np.testing.assert_allclose(cached_loss, direct_loss, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        cached_gradient.weight, direct_gradient.weight, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        cached_gradient.bias, direct_gradient.bias, rtol=1e-6, atol=1e-7
    )


def test_pointwise_tasks_are_jittable_and_differentiable() -> None:
    inputs = _Inputs(jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10)
    batch = PointwiseBatch(
        inputs=inputs,
        labels=jnp.asarray((1.0, 0.0, 1.0, 0.0, 1.0)),
        valid=jnp.asarray((True, True, True, True, False)),
    )
    task = PointwiseScoringTask(objective="binary_cross_entropy")
    model = _scorer()

    def objective(candidate: _Scorer) -> Float[Array, ""]:
        return task.loss(candidate, batch).loss

    loss, gradient = jax.jit(jax.value_and_grad(objective))(model)
    assert np.isfinite(loss)
    assert gradient.weight.shape == model.weight.shape


def test_pairwise_margin_matches_direct_difference() -> None:
    model = _scorer()
    positive = _Inputs(jnp.asarray(((1.0, 0.0, 1.0), (0.5, 0.2, 0.1))))
    negative = _Inputs(jnp.asarray(((0.0, 1.0, 0.0), (0.1, 0.2, 0.5))))
    margins = jnp.asarray((0.25, -0.1))
    batch = PairwiseRankingBatch(
        positive=positive,
        negative=negative,
        margins=margins,
        valid=jnp.asarray((True, True)),
    )
    task = build_task(PairwiseRankingConfig(), MarginMSEConfig())
    actual = task.loss(model, batch).loss
    predicted = model.logits(positive)[:, 0] - model.logits(negative)[:, 0]
    np.testing.assert_allclose(actual, jnp.mean(jnp.square(predicted - margins)))


@pytest.mark.parametrize(
    "task",
    (
        PointwiseScoringTask(objective="mse"),
        ListwiseScoringTask(objective="listnet"),
        ListwiseScoringTask(objective="list_mle"),
    ),
)
def test_exact_gradient_accumulation_matches_one_full_batch(task) -> None:
    model = _scorer()
    optimizer = optax.sgd(learning_rate=3e-3)
    state = init_train_state(model, optimizer)
    values = jnp.arange(4 * 3 * 3, dtype=jnp.float32).reshape(4, 3, 3) / 20
    if isinstance(task, PointwiseScoringTask):
        batch = PointwiseBatch(
            inputs=_Inputs(values[:, 0]),
            labels=jnp.asarray((0.2, -0.3, 0.7, 0.1)),
            valid=jnp.asarray((True, True, True, False)),
        )
    else:
        batch = ListwiseRankingBatch(
            inputs=_Inputs(values),
            labels=jnp.asarray(
                (
                    (3.0, 2.0, 1.0),
                    (2.0, 0.0, 1.0),
                    (1.0, 3.0, 2.0),
                    (2.0, 1.0, 0.0),
                )
            ),
            valid=jnp.ones((4, 3), dtype=jnp.bool_),
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
    for actual, expected in zip(
        jax.tree.leaves(accumulated.state),
        jax.tree.leaves(direct.state),
        strict=True,
    ):
        if eqx.is_array(actual):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("objective", ("ranknet", "lambda"))
def test_order_dependent_listwise_losses_reject_gradient_accumulation(
    objective,
) -> None:
    with pytest.raises(TypeError, match="does not support exact"):
        build_train_step(
            ListwiseScoringTask(objective=objective),
            optax.sgd(1e-3),
            gradient_accumulation_steps=2,
        )


@pytest.mark.parametrize(
    "task",
    (
        ListwiseScoringTask(objective="ranknet"),
        ListwiseScoringTask(objective="lambda"),
        ListwiseScoringTask(objective="listnet"),
        ListwiseScoringTask(objective="list_mle"),
        ListwiseScoringTask(objective="list_mle", position_aware=True),
    ),
)
def test_listwise_tasks_keep_query_as_the_batch_axis(
    task: ListwiseScoringTask,
) -> None:
    values = jnp.arange(2 * 4 * 3, dtype=jnp.float32).reshape(2, 4, 3) / 20
    batch = ListwiseRankingBatch(
        inputs=_Inputs(values),
        labels=jnp.asarray(((3.0, 1.0, 2.0, 0.0), (2.0, 0.0, 1.0, 0.0))),
        valid=jnp.asarray(((True, True, True, True), (True, True, True, False))),
    )
    model = _scorer()

    def objective(candidate: _Scorer) -> Float[Array, ""]:
        return task.loss(candidate, batch).loss

    loss, gradient = jax.jit(jax.value_and_grad(objective))(model)
    assert np.isfinite(loss)
    assert np.all(np.isfinite(gradient.weight))
