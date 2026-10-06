"""Serializable cross-encoder task and loss configurations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, PositiveFloat

from representax.tasks.config import LossConfig, TaskConfig

ScoreActivation = Literal["identity", "sigmoid", "tanh"]
ReductionLog = Literal["natural", "binary"]
LambdaWeighting = Literal[
    "none", "ndcg_loss1", "ndcg_loss2", "lambda_rank", "ndcg_loss2pp"
]


class PointwiseScoringConfig(TaskConfig):
    kind: Literal["pointwise_scoring"] = "pointwise_scoring"


class PairwiseRankingConfig(TaskConfig):
    kind: Literal["pairwise_ranking"] = "pairwise_ranking"


class ListwiseRankingConfig(TaskConfig):
    kind: Literal["listwise_ranking"] = "listwise_ranking"


class CrossInBatchRankingConfig(TaskConfig):
    kind: Literal["cross_in_batch_ranking"] = "cross_in_batch_ranking"


class BinaryCrossEntropyConfig(LossConfig):
    kind: Literal["cross_binary_cross_entropy"] = "cross_binary_cross_entropy"
    activation: ScoreActivation = "identity"
    positive_weight: PositiveFloat | None = None


class CrossEntropyConfig(LossConfig):
    kind: Literal["cross_entropy"] = "cross_entropy"


class ScoreMSEConfig(LossConfig):
    kind: Literal["cross_score_mse"] = "cross_score_mse"
    activation: ScoreActivation = "identity"


class MarginMSEConfig(LossConfig):
    kind: Literal["cross_margin_mse"] = "cross_margin_mse"
    activation: ScoreActivation = "identity"


class RankNetConfig(LossConfig):
    kind: Literal["cross_ranknet"] = "cross_ranknet"
    activation: ScoreActivation = "identity"
    sigma: PositiveFloat = 1.0
    k: int | None = Field(default=None, gt=0)
    reduction_log: ReductionLog = "binary"


class LambdaLossConfig(LossConfig):
    kind: Literal["cross_lambda"] = "cross_lambda"
    activation: ScoreActivation = "identity"
    weighting: LambdaWeighting = "ndcg_loss2pp"
    k: int | None = Field(default=None, gt=0)
    sigma: PositiveFloat = 1.0
    reduction_log: ReductionLog = "binary"
    mu: PositiveFloat = 10.0


class ListNetConfig(LossConfig):
    kind: Literal["cross_listnet"] = "cross_listnet"
    activation: ScoreActivation = "identity"


class ListMLEConfig(LossConfig):
    kind: Literal["cross_list_mle"] = "cross_list_mle"
    activation: ScoreActivation = "identity"
    respect_input_order: bool = True
    position_aware: bool = False


class CrossMNRConfig(LossConfig):
    kind: Literal["cross_mnr"] = "cross_mnr"
    activation: ScoreActivation = "sigmoid"
    scale: PositiveFloat = 10.0


__all__ = [
    "BinaryCrossEntropyConfig",
    "CrossEntropyConfig",
    "CrossInBatchRankingConfig",
    "CrossMNRConfig",
    "LambdaLossConfig",
    "LambdaWeighting",
    "ListwiseRankingConfig",
    "ListMLEConfig",
    "ListNetConfig",
    "MarginMSEConfig",
    "PairwiseRankingConfig",
    "PointwiseScoringConfig",
    "RankNetConfig",
    "ReductionLog",
    "ScoreActivation",
    "ScoreMSEConfig",
]
