"""Task-specific evaluator inventory and composition contracts."""

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np

from representax.config import DataConfig, EvaluationConfig
from representax.core import EncoderMetadata, Modality, Route
from representax.data import mix, source
from representax.evaluation import (
    ClassificationEvaluator,
    JEPAEvaluator,
    MiningEvaluationBatch,
    MSEEvaluator,
    ParaphraseMiningEvaluator,
    RerankingEvaluator,
    RewardEvaluator,
    SequentialEvaluator,
    TripletEvaluator,
    beir_evaluation,
    classification_metrics,
    ranking_metrics,
    regression_metrics,
)
from representax.models.processing import Processor
from representax.tasks.cross_encoder import (
    ListwiseRankingBatch,
    PointwiseBatch,
)
from representax.tasks.jepa import JEPABatch, LeJEPATask
from representax.tasks.reward_modeling import PairwiseRewardBatch
from representax.tasks.triplet import explicit_triplet_batch
from representax.train import EvaluationRunner


class Inputs(eqx.Module):
    values: jax.Array


class IdentityEncoder(eqx.Module):
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route=None, key=None):
        del route, key
        return inputs.values


class ScoreModel(eqx.Module):
    weight: jax.Array

    def logits(self, inputs: Inputs, *, key=None):
        del key
        return inputs.values @ self.weight


def encoder(dimension: int = 3) -> IdentityEncoder:
    return IdentityEncoder(
        EncoderMetadata(
            model_id="identity",
            revision="test",
            output_dimension=dimension,
            routes=frozenset({Route.GENERIC, Route.QUERY, Route.DOCUMENT}),
            modalities=frozenset({Modality.TEXT}),
        )
    )


def test_metric_reducers_cover_classification_regression_and_ranking() -> None:
    classification = classification_metrics(
        np.asarray(((4.0, -1.0), (-2.0, 3.0), (2.0, 1.0))),
        np.asarray((0, 1, 1)),
    )
    assert classification["accuracy"] == 2 / 3
    assert 0 <= classification["f1_macro"] <= 1
    regression = regression_metrics(np.asarray((1.0, 2.0)), np.asarray((1.5, 1.5)))
    assert {name: regression[name] for name in ("mse", "rmse", "mae")} == {
        "mse": 0.25,
        "rmse": 0.5,
        "mae": 0.5,
    }
    ranking = ranking_metrics(
        np.asarray(((3.0, 2.0, 1.0),)),
        np.asarray(((2.0, 1.0, 0.0),)),
        np.ones((1, 3), dtype=bool),
        at_k=(1, 3),
    )
    assert ranking["ndcg@3"] == 1.0
    assert ranking["mrr@1"] == 1.0
    padded = ranking_metrics(
        np.asarray(((3.0, 2.0, 1.0), (0.0, 0.0, 0.0))),
        np.asarray(((2.0, 1.0, 0.0), (0.0, 0.0, 0.0))),
        np.asarray(((True, True, True), (False, False, False))),
        at_k=(3,),
    )
    assert padded["ndcg@3"] == ranking["ndcg@3"]


def test_classification_mse_and_sequential_evaluation_compile() -> None:
    model = ScoreModel(jnp.asarray((1.0, -1.0, 0.5)))
    batch = PointwiseBatch(
        inputs=Inputs(jnp.asarray(((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)))),
        labels=jnp.asarray((1, 0), dtype=jnp.int32),
        valid=jnp.asarray((True, True)),
    )
    classification = ClassificationEvaluator(name="class")
    mse = MSEEvaluator(name="score")
    suite = SequentialEvaluator(
        (classification, mse),
        primary_metric="valid/class/accuracy",
    )
    result = EvaluationRunner(suite).run(model, (batch,))
    assert result.metrics["valid/class/accuracy"] == 1.0
    assert "valid/score/mse" in result.metrics


def test_triplet_and_paraphrase_evaluators() -> None:
    model = encoder(2)
    triplet = explicit_triplet_batch(
        anchor=Inputs(jnp.asarray(((1.0, 0.0), (0.0, 1.0)))),
        positive=Inputs(jnp.asarray(((0.9, 0.1), (0.1, 0.9)))),
        negative=Inputs(jnp.asarray(((0.0, 1.0), (1.0, 0.0)))),
    )
    triplet_result = EvaluationRunner(TripletEvaluator()).run(model, (triplet,))
    assert triplet_result.metrics["valid/triplet/accuracy"] == 1.0

    mining = MiningEvaluationBatch(
        inputs=Inputs(jnp.asarray(((1.0, 0.0), (0.99, 0.01), (0.0, 1.0)))),
        ids=jnp.asarray((10, 11, 12)),
        valid=jnp.ones((3,), dtype=jnp.bool_),
    )
    paraphrase = EvaluationRunner(
        ParaphraseMiningEvaluator(frozenset({(10, 11)}), max_pairs=3)
    ).run(model, (mining,))
    assert paraphrase.metrics["valid/paraphrase/average_precision"] == 1.0


def test_reranking_reward_and_jepa_evaluators() -> None:
    scorer = ScoreModel(jnp.asarray((1.0, 0.0)))
    listwise = ListwiseRankingBatch(
        inputs=Inputs(jnp.asarray((((3.0, 0.0), (2.0, 0.0), (1.0, 0.0)),))),
        labels=jnp.asarray(((2.0, 1.0, 0.0),)),
        valid=jnp.ones((1, 3), dtype=jnp.bool_),
    )
    reranking = EvaluationRunner(RerankingEvaluator(at_k=(1, 3))).run(
        scorer, (listwise,)
    )
    assert reranking.metrics["valid/reranking/ndcg@3"] == 1.0

    reward = PairwiseRewardBatch(
        chosen=Inputs(jnp.asarray(((2.0, 0.0), (1.0, 0.0)))),
        rejected=Inputs(jnp.asarray(((0.0, 0.0), (-1.0, 0.0)))),
        margins=jnp.zeros((2,)),
        valid=jnp.ones((2,), dtype=jnp.bool_),
    )
    reward_result = EvaluationRunner(RewardEvaluator()).run(scorer, (reward,))
    assert reward_result.metrics["valid/reward/pairwise_accuracy"] == 1.0
    assert "valid/reward/ndcg@1" not in reward_result.metrics
    assert "valid/reward/mrr@1" not in reward_result.metrics
    assert "valid/reward/map@1" not in reward_result.metrics

    jepa_batch = JEPABatch(
        views=Inputs(
            jnp.asarray(
                (
                    ((1.0, 0.0), (0.9, 0.1)),
                    ((0.0, 1.0), (0.1, 0.9)),
                    ((0.7, 0.7), (0.6, 0.8)),
                )
            )
        ),
        valid=jnp.ones((3, 2), dtype=jnp.bool_),
    )
    jepa = EvaluationRunner(JEPAEvaluator(LeJEPATask(slices=8))).run(
        model=encoder(2), batches=(jepa_batch,)
    )
    assert jepa.metrics["valid/jepa/effective_rank"] > 1.0


def _process(values, **_options):
    return Inputs(
        jnp.asarray(
            [
                [float(len(str(value))), float(str(value).count("a"))]
                for value in values
            ],
            dtype=jnp.float32,
        )
    )


def test_beir_adapter_and_serializable_evaluator_inventory() -> None:
    evaluator, batches = beir_evaluation(
        queries=grain.MapDataset.source([{"_id": "q1", "text": "aa"}]),
        corpus=grain.MapDataset.source(
            [
                {"_id": "d1", "text": "aa"},
                {"_id": "d2", "text": "bbbb"},
            ]
        ),
        qrels=grain.MapDataset.source(
            [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
        ),
        processor=Processor(_process, {"kind": "test"}),
        batch_size=2,
        name="retrieval",
    )
    batches = tuple(batches)
    result = EvaluationRunner(evaluator).run(encoder(2), batches)
    assert all(batch.ids.shape == (2,) for batch in batches)
    assert result.metrics["valid/retrieval/cosine_ndcg@10"] == 1.0

    configuration = EvaluationConfig(
        data=DataConfig(
            distribution=mix(
                source("file:///tmp/eval.jsonl", map="tests.data.identity")
            )
        ),
        batch_size=8,
        evaluators=(
            {"kind": "classification"},
            {"kind": "pair_classification"},
            {"kind": "classification_probe"},
            {"kind": "clustering"},
            {"kind": "jepa_representation"},
            {"kind": "mse"},
            {"kind": "triplet"},
            {"kind": "reranking"},
            {"kind": "reward"},
            {"kind": "jepa"},
            {
                "kind": "paraphrase_mining",
                "duplicate_pairs": ((1, 2),),
            },
            {
                "kind": "information_retrieval",
                "relevant_documents": {1: (2,)},
            },
        ),
        primary_metric="valid/classification/accuracy",
    )
    assert tuple(item.kind for item in configuration.evaluators) == (
        "classification",
        "pair_classification",
        "classification_probe",
        "clustering",
        "jepa_representation",
        "mse",
        "triplet",
        "reranking",
        "reward",
        "jepa",
        "paraphrase_mining",
        "information_retrieval",
    )
