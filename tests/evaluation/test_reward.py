"""Reward evaluation stays distinct from reranking metrics."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from representax.evaluation import RewardEvaluator
from representax.tasks.reward_modeling import (
    ListwiseRewardBatch,
    PairwiseRewardBatch,
    PointwiseRewardBatch,
    ProcessRewardBatch,
)
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class Scorer(eqx.Module):
    weight: jax.Array

    def logits(self, inputs: Inputs, *, key=None):
        del key
        return inputs.values @ self.weight


def test_pairwise_reward_reports_preference_metrics_without_ir_metrics():
    model = Scorer(jnp.asarray(((1.0,), (0.0,))))
    batch = PairwiseRewardBatch(
        chosen=Inputs(jnp.asarray(((2.0, 0.0), (1.0, 0.0)))),
        rejected=Inputs(jnp.asarray(((0.0, 0.0), (-1.0, 0.0)))),
        margins=jnp.asarray((0.0, 1.0)),
        valid=jnp.ones((2,), dtype=jnp.bool_),
    )

    evaluator = RewardEvaluator(kind="pairwise")
    result = EvaluationRunner(evaluator).run(model, (batch,))

    assert evaluator.primary_metric == "valid/reward/pairwise_accuracy"
    assert result.metrics["valid/reward/pairwise_accuracy"] == 1.0
    assert result.metrics["valid/reward/margin_accuracy"] == 1.0
    assert result.metrics["valid/reward/score_margin_mean"] == 2.0
    assert not any(
        metric in name
        for name in result.metrics
        for metric in ("mrr", "ndcg", "map", "precision", "recall")
    )


def test_listwise_reward_uses_ranking_metrics_only_for_candidate_lists():
    model = Scorer(jnp.asarray(((1.0,), (0.0,))))
    batch = ListwiseRewardBatch(
        candidates=Inputs(jnp.asarray((((3.0, 0.0), (2.0, 0.0), (1.0, 0.0)),))),
        preferences=jnp.asarray(((2.0, 1.0, 0.0),)),
        valid=jnp.ones((1, 3), dtype=jnp.bool_),
    )

    result = EvaluationRunner(RewardEvaluator(kind="listwise", at_k=(1, 3))).run(
        model, (batch,)
    )

    assert result.metrics["valid/reward/ndcg@3"] == 1.0
    assert result.metrics["valid/reward/top1_accuracy"] == 1.0


def test_pointwise_and_process_reward_use_target_metrics():
    pointwise_model = Scorer(jnp.asarray(((1.0,), (0.0,))))
    pointwise = PointwiseRewardBatch(
        inputs=Inputs(jnp.asarray(((1.0, 0.0), (-1.0, 0.0)))),
        labels=jnp.asarray((1.0, -1.0)),
        valid=jnp.ones((2,), dtype=jnp.bool_),
    )
    pointwise_result = EvaluationRunner(RewardEvaluator(kind="pointwise")).run(
        pointwise_model, (pointwise,)
    )
    assert pointwise_result.metrics["valid/reward/mse"] == 0.0
    assert pointwise_result.metrics["valid/reward/mae"] == 0.0

    process_model = Scorer(jnp.asarray(((1.0, -1.0), (0.0, 0.0))))
    process = ProcessRewardBatch(
        inputs=Inputs(jnp.asarray(((1.0, 0.0), (-1.0, 0.0)))),
        labels=jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
        valid=jnp.ones((2, 2), dtype=jnp.bool_),
    )
    process_result = EvaluationRunner(RewardEvaluator(kind="process")).run(
        process_model, (process,)
    )
    np.testing.assert_allclose(process_result.metrics["valid/reward/accuracy"], 1.0)
    np.testing.assert_allclose(
        process_result.metrics["valid/reward/sequence_accuracy"], 1.0
    )
