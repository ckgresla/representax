"""Bounded W&B acceptance campaign over every built-in task and evaluator.

This intentionally uses small synthetic, fixed-shape batches. It validates the
registry, compiled optimizer path, evaluator reducers, sharding placement, and
the real asynchronous W&B reporter; it is not a quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from representax.config import JobConfig, WandbConfig
from representax.core import (
    BUILTIN_MODALITIES,
    EncoderMetadata,
    LateInteractionRepresentation,
    Route,
)
from representax.evaluation import (
    ClassificationEvaluator,
    JEPAEvaluator,
    LossEvaluator,
    MiningEvaluationBatch,
    MSEEvaluator,
    ParaphraseMiningEvaluator,
    RerankingEvaluator,
    RewardEvaluator,
    SimilarityEvaluator,
    TripletEvaluator,
    beir_evaluation,
)
from representax.models import (
    DenoisingAutoEncoder,
    DenseEncoder,
    EncoderPair,
    PairClassifier,
    TokenReconstructionDecoder,
)
from representax.models.processing import Processor
from representax.tasks import build_task
from representax.tasks.classification import (
    PairClassificationConfig,
    SoftmaxClassificationConfig,
    pair_classification_batch,
)
from representax.tasks.contrastive_tension import (
    ContrastiveTensionConfig,
    ContrastiveTensionExamplesConfig,
    ContrastiveTensionInBatchConfig,
    ContrastiveTensionPairsConfig,
    contrastive_tension_batch,
    contrastive_tension_examples,
)
from representax.tasks.cross_encoder import (
    BinaryCrossEntropyConfig,
    CrossInBatchRankingConfig,
    CrossMNRConfig,
    ListwiseRankingBatch,
    ListwiseRankingConfig,
    MarginMSEConfig,
    PairwiseRankingBatch,
    PairwiseRankingConfig,
    PointwiseBatch,
    PointwiseScoringConfig,
    RankNetConfig,
    cross_mnr_batch,
)
from representax.tasks.distillation import (
    DistributionDistillationConfig,
    DistributionKLLossConfig,
    EmbeddingDistillationConfig,
    EmbeddingDistillationLossConfig,
    MarginDistillationConfig,
    MarginMSELossConfig,
    distribution_distillation_batch,
    embedding_distillation_batch,
    margin_distillation_batch,
)
from representax.tasks.guided import GISTConfig, GuidedRetrievalConfig, gist_batch
from representax.tasks.jepa import JEPABatch, JEPAConfig, LeJEPAConfig
from representax.tasks.late_interaction import (
    LateInteractionConfig,
    LateInteractionContrastiveConfig,
)
from representax.tasks.mega_batch import (
    MegaBatchConfig,
    MegaBatchMarginConfig,
    mega_batch,
)
from representax.tasks.pairwise import (
    CosineRegressionConfig,
    PairwiseConfig,
    pairwise_batch,
)
from representax.tasks.reconstruction import (
    DenoisingAutoEncoderConfig,
    DenoisingConfig,
    denoising_batch,
)
from representax.tasks.registry import BUILTIN_TASKS
from representax.tasks.regularization import (
    GORConfig,
    RegularizationConfig,
    regularization_batch,
)
from representax.tasks.retrieval import MNRConfig, RetrievalConfig, retrieval_batch
from representax.tasks.reward_modeling import (
    BradleyTerryConfig,
    ListwiseRewardBatch,
    ListwiseRewardConfig,
    PairwiseRewardBatch,
    PairwiseRewardConfig,
    PlackettLuceConfig,
    PointwiseRewardBatch,
    PointwiseRewardConfig,
    PointwiseRewardLossConfig,
    ProcessRewardBatch,
    ProcessRewardConfig,
    ProcessRewardLossConfig,
)
from representax.tasks.triplet import (
    BatchTripletLossConfig,
    ExplicitTripletConfig,
    ExplicitTripletLossConfig,
    LabeledExamplesConfig,
    explicit_triplet_batch,
    labeled_examples_batch,
)
from representax.train import (
    EvaluationRunner,
    MetricRecord,
    RunLogger,
    ShardingPlan,
    WandbReporter,
    build_train_step,
    init_train_state,
)


class Inputs(eqx.Module):
    values: jax.Array


class Scorer(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def logits(self, inputs: Inputs, *, key: Any = None) -> jax.Array:
        del key
        return inputs.values @ self.weight + self.bias


class IdentityEncoder(eqx.Module):
    scale: jax.Array
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode(self, inputs: Inputs, *, route: Route, key: Any = None) -> jax.Array:
        del route, key
        return inputs.values * self.scale


class TokenBatch(eqx.Module):
    values: jax.Array
    valid: jax.Array


class TokenEncoder(eqx.Module):
    projection: jax.Array
    metadata: EncoderMetadata = eqx.field(static=True)

    def encode_late_interaction(
        self, inputs: TokenBatch, *, route: Route, key: Any = None
    ) -> LateInteractionRepresentation:
        del route, key
        return LateInteractionRepresentation(
            values=inputs.values @ self.projection,
            valid=inputs.valid,
        )


@dataclass(frozen=True)
class AcceptanceJob:
    name: str
    kind: str
    group: str
    execution: Mapping[str, Any]

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        if mode != "json":
            raise ValueError("acceptance metadata is JSON-only")
        return {
            "name": self.name,
            "acceptance": {"kind": self.kind, "group": self.group, "steps": 30},
            "execution": dict(self.execution),
        }


@dataclass(frozen=True)
class TrainingCase:
    kind: str
    model: eqx.Module
    task: Any
    batch: Any


@dataclass(frozen=True)
class EvaluationCase:
    kind: str
    model: eqx.Module
    evaluator: Any
    batches: tuple[Any, ...]


def _metadata(name: str, dimension: int) -> EncoderMetadata:
    return EncoderMetadata(
        model_id=f"representax/acceptance-{name}",
        revision="synthetic-v1",
        output_dimension=dimension,
        routes=frozenset(Route),
        modalities=BUILTIN_MODALITIES,
    )


def _arrays(rows: int = 8, dimension: int = 4) -> tuple[jax.Array, ...]:
    keys = jax.random.split(jax.random.key(20260825), 8)
    return tuple(jax.random.normal(key, (rows, dimension)) for key in keys)


def _scorer(outputs: int = 1, dimension: int = 4) -> Scorer:
    return Scorer(
        weight=jax.random.normal(jax.random.key(outputs + 20), (dimension, outputs))
        / 4,
        bias=jnp.zeros((outputs,), dtype=jnp.float32),
    )


def training_cases() -> dict[str, TrainingCase]:
    a, b, c, d, ga, gb, gc, _ = _arrays()

    def encoder(seed: int = 1) -> DenseEncoder:
        return DenseEncoder(4, 4, key=jax.random.key(seed), normalize=False)

    valid = jnp.ones((8,), dtype=jnp.bool_)
    cases: dict[str, TrainingCase] = {}

    cases["retrieval"] = TrainingCase(
        "retrieval",
        encoder(1),
        build_task(RetrievalConfig(), MNRConfig(scale=10.0, symmetric=True)),
        retrieval_batch(query=a, document=b, positive_mask=jnp.eye(8, dtype=jnp.bool_)),
    )
    token_model = TokenEncoder(
        jax.random.normal(jax.random.key(2), (4, 4)) / 4,
        _metadata("late-interaction", 4),
    )
    token_valid = jnp.ones((8, 3), dtype=jnp.bool_)
    cases["late_interaction"] = TrainingCase(
        "late_interaction",
        token_model,
        build_task(LateInteractionConfig(), LateInteractionContrastiveConfig()),
        retrieval_batch(
            query=TokenBatch(a[:, None, :] + jnp.zeros((8, 3, 4)), token_valid),
            document=TokenBatch(b[:, None, :] + jnp.zeros((8, 3, 4)), token_valid),
            positive_mask=jnp.eye(8, dtype=jnp.bool_),
        ),
    )
    cases["pointwise_scoring"] = TrainingCase(
        "pointwise_scoring",
        _scorer(),
        build_task(PointwiseScoringConfig(), BinaryCrossEntropyConfig()),
        PointwiseBatch(Inputs(a), (jnp.arange(8) % 2).astype(jnp.float32), valid),
    )
    grid = jnp.stack(tuple(a + 0.05 * index for index in range(8)), axis=1)
    cases["cross_in_batch_ranking"] = TrainingCase(
        "cross_in_batch_ranking",
        _scorer(),
        build_task(CrossInBatchRankingConfig(), CrossMNRConfig()),
        cross_mnr_batch(
            inputs=Inputs(grid),
            positive_indices=jnp.arange(8),
            valid=jnp.ones((8, 8), dtype=jnp.bool_),
        ),
    )
    cases["pairwise_ranking"] = TrainingCase(
        "pairwise_ranking",
        _scorer(),
        build_task(PairwiseRankingConfig(), MarginMSEConfig()),
        PairwiseRankingBatch(Inputs(a), Inputs(b), jnp.zeros((8,)), valid),
    )
    cases["listwise_ranking"] = TrainingCase(
        "listwise_ranking",
        _scorer(),
        build_task(ListwiseRankingConfig(), RankNetConfig()),
        ListwiseRankingBatch(
            Inputs(jnp.stack((a, b, c), axis=1)),
            jnp.tile(jnp.asarray((2.0, 1.0, 0.0)), (8, 1)),
            jnp.ones((8, 3), dtype=jnp.bool_),
        ),
    )
    cases["pairwise"] = TrainingCase(
        "pairwise",
        encoder(3),
        build_task(PairwiseConfig(), CosineRegressionConfig()),
        pairwise_batch(left=a, right=b, labels=jnp.linspace(-0.5, 0.5, 8)),
    )
    cases["pairwise_reward"] = TrainingCase(
        "pairwise_reward",
        _scorer(),
        build_task(PairwiseRewardConfig(), BradleyTerryConfig()),
        PairwiseRewardBatch(Inputs(a), Inputs(b), jnp.zeros((8,)), valid),
    )
    cases["listwise_reward"] = TrainingCase(
        "listwise_reward",
        _scorer(),
        build_task(ListwiseRewardConfig(), PlackettLuceConfig()),
        ListwiseRewardBatch(
            Inputs(jnp.stack((a, b, c), axis=1)),
            jnp.tile(jnp.asarray((3.0, 2.0, 1.0)), (8, 1)),
            jnp.ones((8, 3), dtype=jnp.bool_),
        ),
    )
    cases["pointwise_reward"] = TrainingCase(
        "pointwise_reward",
        _scorer(),
        build_task(PointwiseRewardConfig(), PointwiseRewardLossConfig()),
        PointwiseRewardBatch(Inputs(a), jnp.linspace(-1.0, 1.0, 8), valid),
    )
    cases["process_reward"] = TrainingCase(
        "process_reward",
        _scorer(outputs=3),
        build_task(ProcessRewardConfig(), ProcessRewardLossConfig()),
        ProcessRewardBatch(
            Inputs(a),
            jnp.tile(jnp.asarray((1.0, 0.0, 1.0)), (8, 1)),
            jnp.ones((8, 3), dtype=jnp.bool_),
        ),
    )
    cases["jepa"] = TrainingCase(
        "jepa",
        IdentityEncoder(jnp.asarray(1.0), _metadata("jepa", 4)),
        build_task(JEPAConfig(), LeJEPAConfig(slices=16, knots=9)),
        JEPABatch(
            Inputs(jnp.stack((a, a + 0.05, a - 0.05), axis=1)),
            jnp.ones((8, 3), dtype=jnp.bool_),
        ),
    )
    cases["explicit_triplet"] = TrainingCase(
        "explicit_triplet",
        encoder(4),
        build_task(ExplicitTripletConfig(), ExplicitTripletLossConfig()),
        explicit_triplet_batch(anchor=a, positive=a + 0.05, negative=b),
    )
    cases["labeled_examples"] = TrainingCase(
        "labeled_examples",
        encoder(5),
        build_task(LabeledExamplesConfig(), BatchTripletLossConfig()),
        labeled_examples_batch(examples=a, labels=jnp.arange(8) // 2),
    )
    cases["embedding_distillation"] = TrainingCase(
        "embedding_distillation",
        encoder(6),
        build_task(EmbeddingDistillationConfig(), EmbeddingDistillationLossConfig()),
        embedding_distillation_batch(
            inputs=(a, b), teacher_embeddings=jnp.stack((ga, gb), axis=1)
        ),
    )
    cases["margin_distillation"] = TrainingCase(
        "margin_distillation",
        encoder(7),
        build_task(MarginDistillationConfig(), MarginMSELossConfig()),
        margin_distillation_batch(
            query=a,
            positive=b,
            negatives=(c, d),
            teacher_margins=jnp.zeros((8, 2)),
        ),
    )
    cases["distribution_distillation"] = TrainingCase(
        "distribution_distillation",
        encoder(8),
        build_task(DistributionDistillationConfig(), DistributionKLLossConfig()),
        distribution_distillation_batch(
            query=a,
            candidates=(b, c, d),
            teacher_scores=jnp.zeros((8, 3)),
        ),
    )
    cases["guided_retrieval"] = TrainingCase(
        "guided_retrieval",
        encoder(9),
        build_task(GuidedRetrievalConfig(), GISTConfig()),
        gist_batch(
            anchor=a,
            positive=b,
            negatives=(c,),
            guide_anchor=ga,
            guide_positive=gb,
            guide_negatives=(gc,),
        ),
    )
    classifier_encoder = encoder(10)
    classifier = PairClassifier.init(
        classifier_encoder, feature_dimension=12, class_count=3, key=jax.random.key(11)
    )
    cases["pair_classification"] = TrainingCase(
        "pair_classification",
        classifier,
        build_task(PairClassificationConfig(), SoftmaxClassificationConfig()),
        pair_classification_batch(left=a, right=b, labels=jnp.arange(8) % 3),
    )
    cases["contrastive_tension_pairs"] = TrainingCase(
        "contrastive_tension_pairs",
        EncoderPair.from_encoder(encoder(12)),
        build_task(ContrastiveTensionPairsConfig(), ContrastiveTensionConfig()),
        contrastive_tension_batch(first=a, second=b, labels=jnp.arange(8) % 2),
    )
    cases["contrastive_tension_examples"] = TrainingCase(
        "contrastive_tension_examples",
        EncoderPair.from_encoder(encoder(13), scale=10.0),
        build_task(
            ContrastiveTensionExamplesConfig(), ContrastiveTensionInBatchConfig()
        ),
        contrastive_tension_examples(a),
    )
    cases["representation_regularization"] = TrainingCase(
        "representation_regularization",
        encoder(14),
        build_task(RegularizationConfig(), GORConfig()),
        regularization_batch((a, b)),
    )
    autoencoder = DenoisingAutoEncoder(
        encoder=encoder(15),
        decoder=TokenReconstructionDecoder.init(
            vocabulary_size=8, hidden_size=4, key=jax.random.key(16)
        ),
    )
    target_ids = jnp.tile(jnp.asarray((1, 2, 3, 4, 0)), (8, 1))
    cases["denoising_reconstruction"] = TrainingCase(
        "denoising_reconstruction",
        autoencoder,
        build_task(DenoisingConfig(), DenoisingAutoEncoderConfig(pad_token_id=0)),
        denoising_batch(damaged=a, target_input_ids=target_ids),
    )
    cases["mega_batch"] = TrainingCase(
        "mega_batch",
        encoder(17),
        build_task(MegaBatchConfig(), MegaBatchMarginConfig()),
        mega_batch(anchor=a, positive=b),
    )
    expected = set(BUILTIN_TASKS.definitions)
    if set(cases) != expected:
        raise AssertionError(
            f"task coverage mismatch: missing={sorted(expected - set(cases))}, "
            f"extra={sorted(set(cases) - expected)}"
        )
    return cases


def evaluation_cases() -> dict[str, EvaluationCase]:
    identity = IdentityEncoder(jnp.asarray(1.0), _metadata("identity", 2))
    scorer = Scorer(jnp.asarray(((1.0,), (0.0,))), jnp.zeros((1,)))
    pair = pairwise_batch(
        left=Inputs(jnp.asarray(((1.0, 0.0), (0.0, 1.0)))),
        right=Inputs(jnp.asarray(((0.9, 0.1), (0.7, 0.7)))),
        labels=jnp.asarray((0.9938837, 0.7071068)),
    )
    pointwise = PointwiseBatch(
        Inputs(
            jnp.asarray(
                tuple((2.0, 0.0) if index % 2 else (0.0, 2.0) for index in range(8))
            )
        ),
        jnp.asarray(tuple(index % 2 for index in range(8)), dtype=jnp.int32),
        jnp.ones((8,), dtype=jnp.bool_),
    )
    triplet = explicit_triplet_batch(
        anchor=Inputs(jnp.asarray(((1.0, 0.0), (0.0, 1.0)))),
        positive=Inputs(jnp.asarray(((0.9, 0.1), (0.1, 0.9)))),
        negative=Inputs(jnp.asarray(((0.0, 1.0), (1.0, 0.0)))),
    )
    listwise = ListwiseRankingBatch(
        Inputs(jnp.asarray((((3.0, 0.0), (2.0, 0.0), (1.0, 0.0)),))),
        jnp.asarray(((2.0, 1.0, 0.0),)),
        jnp.ones((1, 3), dtype=jnp.bool_),
    )
    reward = PairwiseRewardBatch(
        Inputs(jnp.asarray(((2.0, 0.0), (1.0, 0.0)))),
        Inputs(jnp.asarray(((0.0, 0.0), (-1.0, 0.0)))),
        jnp.zeros((2,)),
        jnp.ones((2,), dtype=jnp.bool_),
    )
    jepa_batch = JEPABatch(
        Inputs(
            jnp.asarray(
                (
                    ((1.0, 0.0), (0.9, 0.1)),
                    ((0.0, 1.0), (0.1, 0.9)),
                    ((0.7, 0.7), (0.6, 0.8)),
                )
            )
        ),
        jnp.ones((3, 2), dtype=jnp.bool_),
    )
    mining = MiningEvaluationBatch(
        Inputs(jnp.asarray(((1.0, 0.0), (0.99, 0.01), (0.0, 1.0)))),
        jnp.asarray((10, 11, 12)),
        jnp.ones((3,), dtype=jnp.bool_),
    )
    loss_case = training_cases()["pairwise"]

    def process(values: Sequence[Any], **_options: Any) -> Inputs:
        return Inputs(
            jnp.asarray(
                [
                    [float(len(str(value))), float(str(value).count("a"))]
                    for value in values
                ],
                dtype=jnp.float32,
            )
        )

    ir_evaluator, ir_batches = beir_evaluation(
        queries=({"_id": "q1", "text": "aa"},),
        corpus=({"_id": "d1", "text": "aa"}, {"_id": "d2", "text": "bbbb"}),
        qrels=({"query-id": "q1", "corpus-id": "d1", "score": 1},),
        processor=Processor(process, {"kind": "wandb-acceptance"}),
        batch_size=2,
    )
    return {
        "loss": EvaluationCase(
            "loss", loss_case.model, LossEvaluator(loss_case.task), (loss_case.batch,)
        ),
        "similarity": EvaluationCase(
            "similarity", identity, SimilarityEvaluator(), (pair,)
        ),
        "classification": EvaluationCase(
            "classification", scorer, ClassificationEvaluator(), (pointwise,)
        ),
        "mse": EvaluationCase("mse", scorer, MSEEvaluator(), (pointwise,)),
        "triplet": EvaluationCase("triplet", identity, TripletEvaluator(), (triplet,)),
        "reranking": EvaluationCase(
            "reranking", scorer, RerankingEvaluator(at_k=(1, 3)), (listwise,)
        ),
        "reward": EvaluationCase("reward", scorer, RewardEvaluator(), (reward,)),
        "jepa": EvaluationCase(
            "jepa",
            identity,
            JEPAEvaluator(build_task(JEPAConfig(), LeJEPAConfig(slices=8))),
            (jepa_batch,),
        ),
        "paraphrase_mining": EvaluationCase(
            "paraphrase_mining",
            identity,
            ParaphraseMiningEvaluator(frozenset({(10, 11)}), max_pairs=3),
            (mining,),
        ),
        "information_retrieval": EvaluationCase(
            "information_retrieval", identity, ir_evaluator, tuple(ir_batches)
        ),
    }


def _reporter(
    *,
    project: str,
    entity: str,
    group: str,
    name: str,
    kind: str,
    root: Path,
    execution: Mapping[str, Any],
) -> tuple[RunLogger, Path]:
    run_directory = root / name
    reporter = WandbReporter(
        WandbConfig(
            project=project,
            entity=entity,
            group=group,
            name=name,
            run_id=f"{group}-{name}"[:128],
            tags=("acceptance", kind),
        ),
        job=cast(
            JobConfig,
            AcceptanceJob(
                name=name,
                kind=kind,
                group=group,
                execution=execution,
            ),
        ),
        run_directory=run_directory,
    )
    logger = RunLogger(
        run_directory,
        manifest={"acceptance": {"group": group, "kind": kind}},
        reporters=(reporter,),
        queue_size=8,
    )
    return logger, run_directory


def run_training_case(
    case: TrainingCase,
    *,
    steps: int,
    project: str,
    entity: str,
    group: str,
    root: Path,
    plan_factory: Callable[[Any, Any], ShardingPlan] | None = None,
) -> dict[str, Any]:
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-3)
    model = jax.tree.map(
        lambda value: jnp.array(value, copy=True) if eqx.is_array(value) else value,
        case.model,
    )
    state = init_train_state(model, optimizer)
    plan = None if plan_factory is None else plan_factory(state, optimizer)
    if plan is not None:
        state = plan.place_state(state)
    step = build_train_step(case.task, optimizer, plan=plan, donate_state=True)
    batch = case.batch if plan is None else plan.place_batch(case.batch)
    name = case.kind if plan is None else f"{case.kind}-{plan.mesh.size}-device"
    logger, run_directory = _reporter(
        project=project,
        entity=entity,
        group=group,
        name=name,
        kind="task",
        root=root,
        execution={
            "sharding": "single" if plan is None else "distributed",
            "mesh_shape": [1] if plan is None else list(plan.mesh.devices.shape),
            "mesh_axis_names": [] if plan is None else list(plan.mesh.axis_names),
        },
    )
    initial = [np.asarray(x) for x in jax.tree.leaves(state.model) if eqx.is_array(x)]
    started = time.perf_counter()
    try:
        logger.event("acceptance_started", iteration=0, task=case.kind, steps=steps)
        losses = []
        for iteration in range(1, steps + 1):
            step_started = time.perf_counter()
            result = step(
                state, batch, jax.random.fold_in(jax.random.key(7), iteration)
            )
            jax.block_until_ready(result)
            state = result.state
            loss = float(result.metrics.loss)
            if not bool(result.metrics.numeric_finite) or bool(
                result.metrics.skipped_update
            ):
                raise RuntimeError(f"{name} produced an invalid update at {iteration}")
            losses.append(loss)
            task_metrics = {
                name if name.startswith("train/") else f"train/{name}": value
                for name, value in result.metrics.task.items()
            }
            logger.metrics(
                MetricRecord(
                    iteration=iteration,
                    values={
                        "train/loss": loss,
                        "train/skipped_update": False,
                        "train/numeric_finite": True,
                        "train/gradient_global_norm": float(
                            result.metrics.gradient_global_norm
                        ),
                        "train/clipped_gradient_global_norm": float(
                            result.metrics.clipped_gradient_global_norm
                        ),
                        "train/update_global_norm": float(
                            result.metrics.update_global_norm
                        ),
                        **task_metrics,
                        "perf/step_seconds": time.perf_counter() - step_started,
                    },
                )
            )
        final = [np.asarray(x) for x in jax.tree.leaves(state.model) if eqx.is_array(x)]
        changed = any(
            not np.array_equal(left, right)
            for left, right in zip(initial, final, strict=True)
        )
        if int(state.step) != steps or not changed:
            raise RuntimeError(f"{name} did not complete {steps} parameter updates")
        duration = time.perf_counter() - started
        logger.finish(
            "completed", completed_iterations=steps, duration_seconds=duration
        )
        return {
            "name": name,
            "kind": "task",
            "status": "completed",
            "steps": steps,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "duration_seconds": duration,
            "run_directory": str(run_directory),
        }
    except BaseException as error:
        logger.finish("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        logger.close()


def run_evaluation_case(
    case: EvaluationCase,
    *,
    project: str,
    entity: str,
    group: str,
    root: Path,
    device_count: int = 1,
) -> dict[str, Any]:
    name = f"eval-{case.kind}"
    if device_count > 1:
        name = f"{name}-{device_count}-device"
    logger, run_directory = _reporter(
        project=project,
        entity=entity,
        group=group,
        name=name,
        kind="evaluator",
        root=root,
        execution={
            "sharding": "single" if device_count == 1 else "data_parallel",
            "mesh_shape": [device_count],
            "mesh_axis_names": [] if device_count == 1 else ["data"],
        },
    )
    place_batch: Callable[[Any], Any] = jax.device_put
    if device_count > 1:
        mesh = jax.make_mesh(
            (device_count,), ("data",), devices=jax.devices()[:device_count]
        )
        sharding = NamedSharding(mesh, P("data"))

        def place_batch(tree: Any) -> Any:
            return jax.tree.map(
                lambda value: (
                    jax.device_put(value, sharding)
                    if isinstance(value, jax.Array)
                    else value
                ),
                tree,
            )

    started = time.perf_counter()
    try:
        logger.event("evaluation_started", iteration=0, evaluator=case.kind)
        result = EvaluationRunner(case.evaluator, namespace="eval").run(
            case.model,
            case.batches,
            iteration=0,
            place_batch=place_batch,
        )
        if not result.metrics or not all(
            np.isfinite(value) for value in result.metrics.values()
        ):
            raise RuntimeError(f"{name} produced invalid metrics")
        duration = time.perf_counter() - started
        logger.metrics(
            MetricRecord(
                iteration=0,
                event="evaluation",
                values={
                    **result.metrics,
                    "perf/evaluation_seconds": duration,
                    "perf/evaluation_batches": len(case.batches),
                },
            )
        )
        logger.finish("completed", completed_iterations=0, duration_seconds=duration)
        return {
            "name": name,
            "kind": "evaluator",
            "status": "completed",
            "steps": 0,
            "metrics": dict(result.metrics),
            "duration_seconds": duration,
            "run_directory": str(run_directory),
        }
    except BaseException as error:
        logger.finish("failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        logger.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("tasks", "evaluators", "distributed", "distributed_evaluators"),
        required=True,
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--project", default="representax")
    parser.add_argument("--entity", default="ckgresla")
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is required but will never be printed")
    args.run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if args.mode == "tasks":
        cases = sorted(training_cases().values(), key=lambda case: case.kind)
        cases = [
            case
            for index, case in enumerate(cases)
            if index % args.shards == args.shard
        ]
        for case in cases:
            results.append(
                run_training_case(
                    case,
                    steps=args.steps,
                    project=args.project,
                    entity=args.entity,
                    group=args.group,
                    root=args.run_root,
                )
            )
    elif args.mode == "evaluators":
        cases = sorted(evaluation_cases().values(), key=lambda case: case.kind)
        cases = [
            case
            for index, case in enumerate(cases)
            if index % args.shards == args.shard
        ]
        for case in cases:
            results.append(
                run_evaluation_case(
                    case,
                    project=args.project,
                    entity=args.entity,
                    group=args.group,
                    root=args.run_root,
                )
            )
    elif args.mode == "distributed":
        distributed_case = training_cases()["pointwise_scoring"]
        topologies = ((2, "ddp"), (4, "fsdp"))
        selected_topologies = [
            topology
            for index, topology in enumerate(topologies)
            if index % args.shards == args.shard
        ]
        for device_count, mode in selected_topologies:
            devices = jax.devices()
            if len(devices) < device_count:
                raise RuntimeError(
                    f"distributed acceptance requires {device_count} devices"
                )
            mesh = jax.make_mesh(
                (device_count,), ("data",), devices=devices[:device_count]
            )

            def plan_factory(
                state: Any, optimizer: Any, *, mode=mode, mesh=mesh
            ) -> ShardingPlan:
                if mode == "ddp":
                    return ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
                return ShardingPlan.fsdp(
                    state,
                    optimizer,
                    mesh,
                    parameter_axis_name="data",
                    data_axis_name="data",
                    minimum_parameter_elements=1,
                )

            result = run_training_case(
                replace(distributed_case, kind=f"{distributed_case.kind}-{mode}"),
                steps=args.steps,
                project=args.project,
                entity=args.entity,
                group=args.group,
                root=args.run_root,
                plan_factory=plan_factory,
            )
            result["sharding"] = mode
            results.append(result)
    else:
        topologies = ((2, "ddp"), (4, "fsdp"))
        selected_topologies = [
            topology
            for index, topology in enumerate(topologies)
            if index % args.shards == args.shard
        ]
        classification = evaluation_cases()["classification"]
        for device_count, mode in selected_topologies:
            results.append(
                run_evaluation_case(
                    replace(classification, kind=f"classification-{mode}"),
                    project=args.project,
                    entity=args.entity,
                    group=args.group,
                    root=args.run_root,
                    device_count=device_count,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"mode": args.mode, "completed": len(results), "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
