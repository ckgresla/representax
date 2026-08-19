"""Reusable evaluators and exact host-side metric reducers."""

from .loss import LossBatchOutput, LossEvaluator
from .protocol import Evaluator
from .retrieval import (
    RETRIEVAL_SCORE_FUNCTIONS,
    InformationRetrievalEvaluator,
    RetrievalBatchOutput,
    RetrievalEvaluationBatch,
    RetrievalInputKind,
    RetrievalScoreFunction,
    information_retrieval_metrics,
    retrieval_evaluation_batch,
)
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
    "InformationRetrievalEvaluator",
    "LossBatchOutput",
    "LossEvaluator",
    "RETRIEVAL_SCORE_FUNCTIONS",
    "RetrievalBatchOutput",
    "RetrievalEvaluationBatch",
    "RetrievalInputKind",
    "RetrievalScoreFunction",
    "SIMILARITY_FUNCTIONS",
    "SimilarityFunction",
    "embedding_similarity_metrics",
    "information_retrieval_metrics",
    "retrieval_evaluation_batch",
]
