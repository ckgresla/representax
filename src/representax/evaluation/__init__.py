"""Reusable evaluators and exact host-side metric reducers."""

from .adapters import beir_evaluation
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
)
from .protocol import Evaluator
from .ranking import (
    RankingBatchOutput,
    RerankingEvaluator,
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
from .reward import RewardBatchOutput, RewardEvaluationKind, RewardEvaluator
from .sequential import SequentialEvaluator
from .similarity import (
    SIMILARITY_FUNCTIONS,
    SimilarityBatchOutput,
    SimilarityEvaluator,
    SimilarityFunction,
    similarity_metrics,
)
from .triplet import TripletBatchOutput, TripletEvaluator

__all__ = [
    "ClassificationBatchOutput",
    "ClassificationEvaluator",
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
    "RewardBatchOutput",
    "RewardEvaluationKind",
    "SIMILARITY_FUNCTIONS",
    "SequentialEvaluator",
    "SimilarityFunction",
    "SimilarityBatchOutput",
    "SimilarityEvaluator",
    "TripletBatchOutput",
    "TripletEvaluator",
    "classification_metrics",
    "beir_evaluation",
    "information_retrieval_metrics",
    "ranking_metrics",
    "regression_metrics",
    "retrieval_evaluation_batch",
    "similarity_metrics",
]
