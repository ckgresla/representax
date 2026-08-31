"""Reusable evaluators and exact host-side metric reducers."""

from .adapters import (
    BANKING77_TEST,
    BANKING77_TRAIN,
    CIFAR100_TEST,
    CIFAR100_TRAIN,
    SPRINT_DUPLICATE_QUESTIONS,
    TWENTY_NEWSGROUPS,
    CanonicalEvaluationSource,
    PackedColumns,
    beir_evaluation,
    clustering_evaluation_batches,
    clustering_samples,
    labeled_evaluation_batches,
    pair_classification_batches,
)
from .classification import (
    ClassificationBatchOutput,
    ClassificationEvaluator,
    classification_metrics,
)
from .clustering import (
    ClusteringBatchOutput,
    ClusteringEvaluator,
    clustering_metrics,
)
from .jepa import JEPABatchOutput, JEPAEvaluator
from .jepa_representation import (
    JEPARepresentationEvaluator,
    knn_accuracy,
    representation_geometry_metrics,
)
from .loss import LossBatchOutput, LossEvaluator
from .mining import (
    MiningBatchOutput,
    MiningEvaluationBatch,
    ParaphraseMiningEvaluator,
)
from .pair_classification import (
    PairClassificationEvaluator,
    PairClassificationOutput,
    pair_classification_metrics,
)
from .probe import ClassificationProbeEvaluator
from .protocol import Evaluator
from .ranking import (
    RankingBatchOutput,
    RerankingEvaluator,
    ranking_metrics,
)
from .regression import MSEEvaluator, RegressionBatchOutput, regression_metrics
from .representation import (
    EvaluationSplit,
    LabeledEmbeddingOutput,
    LabeledEvaluationBatch,
    labeled_evaluation_batch,
    linear_probe_metrics,
)
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
    "BANKING77_TEST",
    "BANKING77_TRAIN",
    "CIFAR100_TEST",
    "CIFAR100_TRAIN",
    "SPRINT_DUPLICATE_QUESTIONS",
    "TWENTY_NEWSGROUPS",
    "CanonicalEvaluationSource",
    "ClassificationBatchOutput",
    "ClassificationEvaluator",
    "ClassificationProbeEvaluator",
    "ClusteringBatchOutput",
    "ClusteringEvaluator",
    "Evaluator",
    "EvaluationSplit",
    "InformationRetrievalEvaluator",
    "JEPABatchOutput",
    "JEPAEvaluator",
    "JEPARepresentationEvaluator",
    "LabeledEmbeddingOutput",
    "LabeledEvaluationBatch",
    "LossBatchOutput",
    "LossEvaluator",
    "MSEEvaluator",
    "MiningBatchOutput",
    "MiningEvaluationBatch",
    "ParaphraseMiningEvaluator",
    "PairClassificationEvaluator",
    "PairClassificationOutput",
    "PackedColumns",
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
    "clustering_metrics",
    "clustering_evaluation_batches",
    "clustering_samples",
    "beir_evaluation",
    "information_retrieval_metrics",
    "knn_accuracy",
    "labeled_evaluation_batch",
    "labeled_evaluation_batches",
    "linear_probe_metrics",
    "pair_classification_metrics",
    "pair_classification_batches",
    "ranking_metrics",
    "regression_metrics",
    "retrieval_evaluation_batch",
    "representation_geometry_metrics",
    "similarity_metrics",
]
