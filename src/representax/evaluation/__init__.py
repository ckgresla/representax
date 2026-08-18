"""Reusable evaluators and exact host-side metric reducers."""

from .loss import LossBatchOutput, LossEvaluator
from .protocol import Evaluator
from .similarity import (
    SIMILARITY_FUNCTIONS,
    EmbeddingSimilarityBatchOutput,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    embedding_similarity_metrics,
)

__all__ = [
    "EmbeddingSimilarityBatchOutput",
    "EmbeddingSimilarityEvaluator",
    "Evaluator",
    "LossBatchOutput",
    "LossEvaluator",
    "SIMILARITY_FUNCTIONS",
    "SimilarityFunction",
    "embedding_similarity_metrics",
]
