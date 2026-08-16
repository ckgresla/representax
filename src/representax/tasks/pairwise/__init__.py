"""Labeled-pair task family."""

from .batch import PairwiseBatch, pairwise_batch
from .config import (
    AngleConfig,
    ContrastiveConfig,
    CoSENTConfig,
    CosineRegressionConfig,
    PairwiseConfig,
)
from .losses import (
    PairDistance,
    PairLossTerms,
    PairRankingTerms,
    contrastive_loss_terms,
    cosine_regression_loss_terms,
    online_contrastive_loss_terms,
    pair_ranking_loss_terms,
    pairwise_angle_similarity,
    pairwise_cosine_similarity,
    pairwise_distance,
)
from .task import AnglETask, ContrastiveTask, CoSENTTask, CosineRegressionTask

__all__ = [
    "AngleConfig",
    "AnglETask",
    "CoSENTConfig",
    "CoSENTTask",
    "ContrastiveConfig",
    "ContrastiveTask",
    "CosineRegressionConfig",
    "CosineRegressionTask",
    "PairDistance",
    "PairLossTerms",
    "PairRankingTerms",
    "PairwiseBatch",
    "PairwiseConfig",
    "contrastive_loss_terms",
    "cosine_regression_loss_terms",
    "online_contrastive_loss_terms",
    "pair_ranking_loss_terms",
    "pairwise_angle_similarity",
    "pairwise_batch",
    "pairwise_cosine_similarity",
    "pairwise_distance",
]
