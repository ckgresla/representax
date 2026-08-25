"""Reusable evaluators and exact host-side metric reducers."""

from .adapters import nanobeir_evaluation
from .classification import (
    ClassificationBatchOutput,
    ClassificationEvaluator,
    classification_metrics,
)
from .jepa import JEPABatchOutput, JEPAEvaluator
from .loss import LossBatchOutput, LossEvaluator
from .mining import (
    MiningBatchOutput,
    MiningEvaluationBatch,
    ParaphraseMiningEvaluator,
    TranslationBatchOutput,
    TranslationEvaluator,
)
from .protocol import Evaluator
from .ranking import (
    RankingBatchOutput,
    RerankingEvaluator,
    RewardEvaluator,
    ranking_metrics,
)
from .regression import MSEEvaluator, RegressionBatchOutput, regression_metrics
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
from .sequential import SequentialEvaluator
from .similarity import (
    SIMILARITY_FUNCTIONS,
    EmbeddingSimilarityBatchOutput,
    EmbeddingSimilarityEvaluator,
    SimilarityFunction,
    embedding_similarity_metrics,
)
from .triplet import TripletBatchOutput, TripletEvaluator

__all__ = [
    "ClassificationBatchOutput",
    "ClassificationEvaluator",
    "EmbeddingSimilarityBatchOutput",
    "EmbeddingSimilarityEvaluator",
    "Evaluator",
    "InformationRetrievalEvaluator",
    "JEPABatchOutput",
    "JEPAEvaluator",
    "LossBatchOutput",
    "LossEvaluator",
    "MSEEvaluator",
    "MiningBatchOutput",
    "MiningEvaluationBatch",
    "ParaphraseMiningEvaluator",
    "RETRIEVAL_SCORE_FUNCTIONS",
    "RankingBatchOutput",
    "RegressionBatchOutput",
    "RetrievalBatchOutput",
    "RetrievalEvaluationBatch",
    "RetrievalInputKind",
    "RetrievalScoreFunction",
    "RerankingEvaluator",
    "RewardEvaluator",
    "SIMILARITY_FUNCTIONS",
    "SequentialEvaluator",
    "SimilarityFunction",
    "TranslationBatchOutput",
    "TranslationEvaluator",
    "TripletBatchOutput",
    "TripletEvaluator",
    "classification_metrics",
    "embedding_similarity_metrics",
    "information_retrieval_metrics",
    "nanobeir_evaluation",
    "ranking_metrics",
    "regression_metrics",
    "retrieval_evaluation_batch",
]
