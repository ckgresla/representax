"""Supervised pair classification."""

from .batch import PairClassificationBatch, pair_classification_batch
from .config import PairClassificationConfig, SoftmaxClassificationConfig
from .losses import (
    SoftmaxClassificationTerms,
    pair_features,
    softmax_classification_loss_terms,
)
from .task import SoftmaxClassificationTask

__all__ = [
    "PairClassificationBatch",
    "PairClassificationConfig",
    "SoftmaxClassificationConfig",
    "SoftmaxClassificationTask",
    "SoftmaxClassificationTerms",
    "pair_classification_batch",
    "pair_features",
    "softmax_classification_loss_terms",
]
