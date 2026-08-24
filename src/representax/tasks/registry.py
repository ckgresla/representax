"""Immutable registries for structured task and loss configurations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from representax.core import Route

from .classification import (
    PairClassificationBatch,
    PairClassificationConfig,
    SoftmaxClassificationConfig,
    SoftmaxClassificationTask,
)
from .config import LossConfig, LossModifierConfig, TaskConfig
from .contrastive_tension import (
    ContrastiveTensionBatch,
    ContrastiveTensionConfig,
    ContrastiveTensionExamples,
    ContrastiveTensionExamplesConfig,
    ContrastiveTensionInBatchConfig,
    ContrastiveTensionInBatchTask,
    ContrastiveTensionPairsConfig,
    ContrastiveTensionTask,
)
from .distillation.batch import (
    DistributionDistillationBatch,
    EmbeddingDistillationBatch,
    MarginDistillationBatch,
)
from .distillation.config import (
    DistributionDistillationConfig,
    DistributionKLLossConfig,
    EmbeddingDistillationConfig,
    EmbeddingDistillationLossConfig,
    MarginDistillationConfig,
    MarginMSELossConfig,
)
from .distillation.task import (
    DistributionDistillationTask,
    EmbeddingDistillationTask,
    MarginDistillationTask,
)
from .guided import GISTBatch, GISTConfig, GISTTask, GuidedRetrievalConfig
from .late_interaction import (
    LateInteractionConfig,
    LateInteractionContrastiveConfig,
    LateInteractionTask,
)
from .mega_batch import (
    MegaBatch,
    MegaBatchConfig,
    MegaBatchMarginConfig,
    MegaBatchMarginTask,
)
from .modifiers import (
    AdaptiveLayerModifierConfig,
    AdaptiveLayerTask,
    Matryoshka2dModifierConfig,
    Matryoshka2dTask,
    MatryoshkaModifierConfig,
    MatryoshkaTask,
)
from .pairwise.batch import PairwiseBatch
from .pairwise.config import (
    AngleConfig,
    ContrastiveConfig,
    CoSENTConfig,
    CosineRegressionConfig,
    PairwiseConfig,
)
from .pairwise.task import AnglETask, ContrastiveTask, CoSENTTask, CosineRegressionTask
from .reconstruction import (
    DenoisingAutoEncoderConfig,
    DenoisingAutoEncoderTask,
    DenoisingBatch,
    DenoisingConfig,
)
from .regularization import (
    GlobalOrthogonalRegularizationTask,
    GORConfig,
    RegularizationBatch,
    RegularizationConfig,
)
from .retrieval.batch import RetrievalBatch
from .retrieval.config import MNRConfig, RetrievalConfig
from .retrieval.mnr import MNRTask
from .triplet.batch import ExplicitTripletBatch, LabeledExamplesBatch
from .triplet.config import (
    BatchHardSoftMarginLossConfig,
    BatchTripletLossConfig,
    ExplicitTripletConfig,
    ExplicitTripletLossConfig,
    LabeledExamplesConfig,
)
from .triplet.task import BatchTripletTask, ExplicitTripletTask


def _index_definitions(definitions: Iterable[Any], *, label: str) -> Mapping[str, Any]:
    indexed = {}
    for definition in definitions:
        if definition.kind in indexed:
            raise ValueError(f"duplicate {label} kind {definition.kind!r}")
        indexed[definition.kind] = definition
    return MappingProxyType(indexed)


def _parse_registered(
    value: Any,
    *,
    base_type: type,
    definitions: Mapping[str, Any],
    label: str,
) -> Any:
    if isinstance(value, base_type):
        kind = value.kind
        try:
            definition = definitions[kind]
        except KeyError as error:
            raise ValueError(f"{label} kind {kind!r} is not registered") from error
        if not isinstance(value, definition.config_type):
            raise TypeError(
                f"{label} {kind!r} requires {definition.config_type.__name__}, "
                f"received {type(value).__name__}"
            )
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} config must be a mapping or {base_type.__name__}")
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise TypeError(f"{label} config must contain a string kind")
    try:
        definition = definitions[kind]
    except KeyError as error:
        raise ValueError(f"{label} kind {kind!r} is not registered") from error
    return definition.config_type.model_validate(value)


@dataclass(frozen=True, slots=True)
class TaskDefinition:
    """One task identity and its model-ready batch contract."""

    kind: str
    config_type: type[TaskConfig]
    batch_type: type

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("task kind must be non-empty")


class TaskRegistry:
    """Closed task definitions with explicit immutable extension."""

    def __init__(self, definitions: Iterable[TaskDefinition]) -> None:
        self._definitions: Mapping[str, TaskDefinition] = _index_definitions(
            definitions,
            label="task",
        )

    @property
    def definitions(self) -> Mapping[str, TaskDefinition]:
        return self._definitions

    def definition(self, kind: str) -> TaskDefinition:
        try:
            return self._definitions[kind]
        except KeyError as error:
            raise KeyError(f"task kind {kind!r} is not registered") from error

    def parse(self, value: Any) -> TaskConfig:
        return _parse_registered(
            value,
            base_type=TaskConfig,
            definitions=self._definitions,
            label="task",
        )

    def extended(self, *definitions: TaskDefinition) -> TaskRegistry:
        return TaskRegistry((*self._definitions.values(), *definitions))


@dataclass(frozen=True, slots=True)
class LossDefinition:
    """One loss identity, runtime builder, and compatibility contract."""

    kind: str
    config_type: type[LossConfig]
    build: Callable[[TaskConfig, LossConfig], Any]
    task_kinds: frozenset[str]
    training_strategies: frozenset[str]
    microbatch_accumulation: bool = False

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("loss kind must be non-empty")
        if not self.task_kinds:
            raise ValueError("a loss must support at least one task")
        if not self.training_strategies:
            raise ValueError("a loss must support at least one training strategy")


class LossRegistry:
    """Closed loss definitions with explicit immutable extension."""

    def __init__(self, definitions: Iterable[LossDefinition]) -> None:
        self._definitions: Mapping[str, LossDefinition] = _index_definitions(
            definitions,
            label="loss",
        )

    @property
    def definitions(self) -> Mapping[str, LossDefinition]:
        return self._definitions

    def definition(self, kind: str) -> LossDefinition:
        try:
            return self._definitions[kind]
        except KeyError as error:
            raise KeyError(f"loss kind {kind!r} is not registered") from error

    def parse(self, value: Any) -> LossConfig:
        return _parse_registered(
            value,
            base_type=LossConfig,
            definitions=self._definitions,
            label="loss",
        )

    def build(self, task: TaskConfig, config: LossConfig) -> Any:
        definition = self.definition(config.kind)
        if not isinstance(config, definition.config_type):
            raise TypeError(
                f"loss {config.kind!r} requires {definition.config_type.__name__}, "
                f"received {type(config).__name__}"
            )
        return definition.build(task, config)

    def extended(self, *definitions: LossDefinition) -> LossRegistry:
        return LossRegistry((*self._definitions.values(), *definitions))


@dataclass(frozen=True, slots=True)
class LossModifierDefinition:
    """One composable representation-loss transformation."""

    kind: str
    config_type: type[LossModifierConfig]
    build: Callable[[Any, LossModifierConfig], Any]
    training_strategies: frozenset[str]
    microbatch_accumulation: bool = False

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("loss modifier kind must be non-empty")
        if not self.training_strategies:
            raise ValueError("a loss modifier must support a training strategy")


class LossModifierRegistry:
    """Closed loss-modifier definitions with explicit immutable extension."""

    def __init__(self, definitions: Iterable[LossModifierDefinition]) -> None:
        self._definitions: Mapping[str, LossModifierDefinition] = _index_definitions(
            definitions,
            label="loss modifier",
        )

    @property
    def definitions(self) -> Mapping[str, LossModifierDefinition]:
        return self._definitions

    def definition(self, kind: str) -> LossModifierDefinition:
        try:
            return self._definitions[kind]
        except KeyError as error:
            raise KeyError(f"loss modifier kind {kind!r} is not registered") from error

    def parse(self, value: Any) -> LossModifierConfig:
        return _parse_registered(
            value,
            base_type=LossModifierConfig,
            definitions=self._definitions,
            label="loss modifier",
        )

    def build(self, task: Any, config: LossModifierConfig) -> Any:
        definition = self.definition(config.kind)
        if not isinstance(config, definition.config_type):
            raise TypeError(
                f"loss modifier {config.kind!r} requires "
                f"{definition.config_type.__name__}, received {type(config).__name__}"
            )
        return definition.build(task, config)

    def extended(
        self,
        *definitions: LossModifierDefinition,
    ) -> LossModifierRegistry:
        return LossModifierRegistry((*self._definitions.values(), *definitions))


def _build_mnr_task(task: TaskConfig, loss: LossConfig) -> MNRTask:
    if not isinstance(task, RetrievalConfig) or not isinstance(loss, MNRConfig):
        raise TypeError("mnr requires MNRConfig")
    return MNRTask(
        scale=loss.scale,
        symmetric=loss.symmetric,
        negative_scope=loss.negative_scope,
    )


def _build_late_interaction_task(
    task: TaskConfig,
    loss: LossConfig,
) -> LateInteractionTask:
    if not isinstance(task, LateInteractionConfig) or not isinstance(
        loss, LateInteractionContrastiveConfig
    ):
        raise TypeError("late interaction requires its task and loss configs")
    return LateInteractionTask(
        temperature=loss.temperature,
        symmetric=loss.symmetric,
        negative_scope=loss.negative_scope,
    )


def _pairwise_routes(task: TaskConfig) -> tuple[Route, Route]:
    if not isinstance(task, PairwiseConfig):
        raise TypeError("pairwise losses require PairwiseConfig")
    return task.left_route, task.right_route


def _build_cosine_regression_task(
    task: TaskConfig,
    loss: LossConfig,
) -> CosineRegressionTask:
    if not isinstance(loss, CosineRegressionConfig):
        raise TypeError("cosine_regression requires CosineRegressionConfig")
    left_route, right_route = _pairwise_routes(task)
    return CosineRegressionTask(left_route=left_route, right_route=right_route)


def _build_contrastive_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ContrastiveTask:
    if not isinstance(loss, ContrastiveConfig):
        raise TypeError("contrastive requires ContrastiveConfig")
    left_route, right_route = _pairwise_routes(task)
    return ContrastiveTask(
        distance=loss.distance,
        margin=loss.margin,
        online=loss.mining == "online",
        left_route=left_route,
        right_route=right_route,
    )


def _build_cosent_task(task: TaskConfig, loss: LossConfig) -> CoSENTTask:
    if not isinstance(loss, CoSENTConfig):
        raise TypeError("cosent requires CoSENTConfig")
    left_route, right_route = _pairwise_routes(task)
    return CoSENTTask(
        scale=loss.scale,
        left_route=left_route,
        right_route=right_route,
    )


def _build_angle_task(task: TaskConfig, loss: LossConfig) -> AnglETask:
    if not isinstance(loss, AngleConfig):
        raise TypeError("angle requires AngleConfig")
    left_route, right_route = _pairwise_routes(task)
    return AnglETask(
        scale=loss.scale,
        left_route=left_route,
        right_route=right_route,
    )


def _build_explicit_triplet_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ExplicitTripletTask:
    if not isinstance(task, ExplicitTripletConfig) or not isinstance(
        loss, ExplicitTripletLossConfig
    ):
        raise TypeError(
            "triplet requires ExplicitTripletConfig and ExplicitTripletLossConfig"
        )
    return ExplicitTripletTask(
        distance=loss.distance,
        margin=loss.margin,
        anchor_route=task.anchor_route,
        positive_route=task.positive_route,
        negative_route=task.negative_route,
    )


def _build_batch_triplet_task(
    task: TaskConfig,
    loss: LossConfig,
) -> BatchTripletTask:
    if not isinstance(task, LabeledExamplesConfig) or not isinstance(
        loss, BatchTripletLossConfig
    ):
        raise TypeError(
            "batch_triplet requires LabeledExamplesConfig and BatchTripletLossConfig"
        )
    return BatchTripletTask(
        mining=loss.mining,
        distance=loss.distance,
        margin=loss.margin,
        route=task.route,
    )


def _build_batch_hard_soft_margin_task(
    task: TaskConfig,
    loss: LossConfig,
) -> BatchTripletTask:
    if not isinstance(task, LabeledExamplesConfig) or not isinstance(
        loss, BatchHardSoftMarginLossConfig
    ):
        raise TypeError(
            "batch_hard_soft_margin requires LabeledExamplesConfig and "
            "BatchHardSoftMarginLossConfig"
        )
    return BatchTripletTask(
        mining="hard_soft_margin",
        distance=loss.distance,
        margin=None,
        route=task.route,
    )


def _build_embedding_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> EmbeddingDistillationTask:
    if not isinstance(task, EmbeddingDistillationConfig) or not isinstance(
        loss, EmbeddingDistillationLossConfig
    ):
        raise TypeError(
            "embedding_distillation requires EmbeddingDistillationConfig and "
            "EmbeddingDistillationLossConfig"
        )
    return EmbeddingDistillationTask(distance=loss.distance, routes=task.routes)


def _build_margin_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> MarginDistillationTask:
    if not isinstance(task, MarginDistillationConfig) or not isinstance(
        loss, MarginMSELossConfig
    ):
        raise TypeError(
            "margin_mse requires MarginDistillationConfig and MarginMSELossConfig"
        )
    return MarginDistillationTask(
        similarity=loss.similarity,
        query_route=task.query_route,
        document_route=task.document_route,
    )


def _build_distribution_distillation_task(
    task: TaskConfig,
    loss: LossConfig,
) -> DistributionDistillationTask:
    if not isinstance(task, DistributionDistillationConfig) or not isinstance(
        loss, DistributionKLLossConfig
    ):
        raise TypeError(
            "distribution_kl requires DistributionDistillationConfig and "
            "DistributionKLLossConfig"
        )
    return DistributionDistillationTask(
        similarity=loss.similarity,
        temperature=loss.temperature,
        query_route=task.query_route,
        candidate_route=task.candidate_route,
    )


def _build_gist_task(task: TaskConfig, loss: LossConfig) -> GISTTask:
    if not isinstance(task, GuidedRetrievalConfig) or not isinstance(loss, GISTConfig):
        raise TypeError("gist requires GuidedRetrievalConfig and GISTConfig")
    return GISTTask(
        temperature=loss.temperature,
        margin_strategy=loss.margin_strategy,
        margin=loss.margin,
        contrast_anchors=loss.contrast_anchors,
        contrast_positives=loss.contrast_positives,
    )


def _build_softmax_classification_task(
    task: TaskConfig,
    loss: LossConfig,
) -> SoftmaxClassificationTask:
    if not isinstance(task, PairClassificationConfig) or not isinstance(
        loss, SoftmaxClassificationConfig
    ):
        raise TypeError("softmax classification requires pair classification configs")
    return SoftmaxClassificationTask(
        concatenate_representations=loss.concatenate_representations,
        concatenate_difference=loss.concatenate_difference,
        concatenate_product=loss.concatenate_product,
        left_route=task.left_route,
        right_route=task.right_route,
    )


def _build_contrastive_tension_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ContrastiveTensionTask:
    if not isinstance(task, ContrastiveTensionPairsConfig) or not isinstance(
        loss, ContrastiveTensionConfig
    ):
        raise TypeError("contrastive tension requires its aligned-pair configs")
    return ContrastiveTensionTask()


def _build_contrastive_tension_in_batch_task(
    task: TaskConfig,
    loss: LossConfig,
) -> ContrastiveTensionInBatchTask:
    if not isinstance(task, ContrastiveTensionExamplesConfig) or not isinstance(
        loss, ContrastiveTensionInBatchConfig
    ):
        raise TypeError("in-batch contrastive tension requires example configs")
    return ContrastiveTensionInBatchTask(similarity=loss.similarity)


def _build_gor_task(
    task: TaskConfig,
    loss: LossConfig,
) -> GlobalOrthogonalRegularizationTask:
    if not isinstance(task, RegularizationConfig) or not isinstance(loss, GORConfig):
        raise TypeError("GOR requires regularization task and loss configs")
    return GlobalOrthogonalRegularizationTask(
        similarity=loss.similarity,
        mean_weight=loss.mean_weight,
        second_moment_weight=loss.second_moment_weight,
        aggregation=loss.aggregation,
        routes=task.routes,
    )


def _build_denoising_task(
    task: TaskConfig,
    loss: LossConfig,
) -> DenoisingAutoEncoderTask:
    if not isinstance(task, DenoisingConfig) or not isinstance(
        loss, DenoisingAutoEncoderConfig
    ):
        raise TypeError("denoising autoencoding requires its task and loss configs")
    return DenoisingAutoEncoderTask(
        pad_token_id=loss.pad_token_id,
        route=task.route,
    )


def _build_mega_batch_task(
    task: TaskConfig,
    loss: LossConfig,
) -> MegaBatchMarginTask:
    if not isinstance(task, MegaBatchConfig) or not isinstance(
        loss, MegaBatchMarginConfig
    ):
        raise TypeError("mega-batch margin requires its task and loss configs")
    return MegaBatchMarginTask(
        positive_margin=loss.positive_margin,
        negative_margin=loss.negative_margin,
        anchor_route=task.anchor_route,
        positive_route=task.positive_route,
    )


def _build_matryoshka_modifier(
    task: Any,
    config: LossModifierConfig,
) -> MatryoshkaTask:
    if not isinstance(config, MatryoshkaModifierConfig):
        raise TypeError("matryoshka requires MatryoshkaModifierConfig")
    return MatryoshkaTask(
        task,
        config.dimensions,
        weights=config.weights,
        dimensions_per_step=config.dimensions_per_step,
    )


def _build_adaptive_layer_modifier(
    task: Any,
    config: LossModifierConfig,
) -> AdaptiveLayerTask:
    if not isinstance(config, AdaptiveLayerModifierConfig):
        raise TypeError("adaptive layer requires AdaptiveLayerModifierConfig")
    return AdaptiveLayerTask(
        task,
        layers_per_step=config.layers_per_step,
        final_layer_weight=config.final_layer_weight,
        prior_layer_weight=config.prior_layer_weight,
        kl_divergence_weight=config.kl_divergence_weight,
        kl_temperature=config.kl_temperature,
    )


def _build_matryoshka_2d_modifier(
    task: Any,
    config: LossModifierConfig,
) -> Matryoshka2dTask:
    if not isinstance(config, Matryoshka2dModifierConfig):
        raise TypeError("matryoshka 2D requires Matryoshka2dModifierConfig")
    return Matryoshka2dTask(
        task,
        config.dimensions,
        weights=config.weights,
        dimensions_per_step=config.dimensions_per_step,
        layers_per_step=config.layers_per_step,
        final_layer_weight=config.final_layer_weight,
        prior_layer_weight=config.prior_layer_weight,
        kl_divergence_weight=config.kl_divergence_weight,
        kl_temperature=config.kl_temperature,
    )


BUILTIN_TASKS = TaskRegistry(
    (
        TaskDefinition(
            kind="retrieval",
            config_type=RetrievalConfig,
            batch_type=RetrievalBatch,
        ),
        TaskDefinition(
            kind="late_interaction",
            config_type=LateInteractionConfig,
            batch_type=RetrievalBatch,
        ),
        TaskDefinition(
            kind="pairwise",
            config_type=PairwiseConfig,
            batch_type=PairwiseBatch,
        ),
        TaskDefinition(
            kind="explicit_triplet",
            config_type=ExplicitTripletConfig,
            batch_type=ExplicitTripletBatch,
        ),
        TaskDefinition(
            kind="labeled_examples",
            config_type=LabeledExamplesConfig,
            batch_type=LabeledExamplesBatch,
        ),
        TaskDefinition(
            kind="embedding_distillation",
            config_type=EmbeddingDistillationConfig,
            batch_type=EmbeddingDistillationBatch,
        ),
        TaskDefinition(
            kind="margin_distillation",
            config_type=MarginDistillationConfig,
            batch_type=MarginDistillationBatch,
        ),
        TaskDefinition(
            kind="distribution_distillation",
            config_type=DistributionDistillationConfig,
            batch_type=DistributionDistillationBatch,
        ),
        TaskDefinition(
            kind="guided_retrieval",
            config_type=GuidedRetrievalConfig,
            batch_type=GISTBatch,
        ),
        TaskDefinition(
            kind="pair_classification",
            config_type=PairClassificationConfig,
            batch_type=PairClassificationBatch,
        ),
        TaskDefinition(
            kind="contrastive_tension_pairs",
            config_type=ContrastiveTensionPairsConfig,
            batch_type=ContrastiveTensionBatch,
        ),
        TaskDefinition(
            kind="contrastive_tension_examples",
            config_type=ContrastiveTensionExamplesConfig,
            batch_type=ContrastiveTensionExamples,
        ),
        TaskDefinition(
            kind="representation_regularization",
            config_type=RegularizationConfig,
            batch_type=RegularizationBatch,
        ),
        TaskDefinition(
            kind="denoising_reconstruction",
            config_type=DenoisingConfig,
            batch_type=DenoisingBatch,
        ),
        TaskDefinition(
            kind="mega_batch",
            config_type=MegaBatchConfig,
            batch_type=MegaBatch,
        ),
    )
)
BUILTIN_LOSSES = LossRegistry(
    (
        LossDefinition(
            kind="mnr",
            config_type=MNRConfig,
            build=_build_mnr_task,
            task_kinds=frozenset({"retrieval"}),
            training_strategies=frozenset({"direct", "grad_cache"}),
        ),
        LossDefinition(
            kind="late_interaction_contrastive",
            config_type=LateInteractionContrastiveConfig,
            build=_build_late_interaction_task,
            task_kinds=frozenset({"late_interaction"}),
            training_strategies=frozenset({"direct", "grad_cache"}),
        ),
        LossDefinition(
            kind="cosine_regression",
            config_type=CosineRegressionConfig,
            build=_build_cosine_regression_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
            microbatch_accumulation=True,
        ),
        LossDefinition(
            kind="contrastive",
            config_type=ContrastiveConfig,
            build=_build_contrastive_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="cosent",
            config_type=CoSENTConfig,
            build=_build_cosent_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="angle",
            config_type=AngleConfig,
            build=_build_angle_task,
            task_kinds=frozenset({"pairwise"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="triplet",
            config_type=ExplicitTripletLossConfig,
            build=_build_explicit_triplet_task,
            task_kinds=frozenset({"explicit_triplet"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="batch_triplet",
            config_type=BatchTripletLossConfig,
            build=_build_batch_triplet_task,
            task_kinds=frozenset({"labeled_examples"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="batch_hard_soft_margin",
            config_type=BatchHardSoftMarginLossConfig,
            build=_build_batch_hard_soft_margin_task,
            task_kinds=frozenset({"labeled_examples"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="embedding_distillation",
            config_type=EmbeddingDistillationLossConfig,
            build=_build_embedding_distillation_task,
            task_kinds=frozenset({"embedding_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="margin_mse",
            config_type=MarginMSELossConfig,
            build=_build_margin_distillation_task,
            task_kinds=frozenset({"margin_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="distribution_kl",
            config_type=DistributionKLLossConfig,
            build=_build_distribution_distillation_task,
            task_kinds=frozenset({"distribution_distillation"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="gist",
            config_type=GISTConfig,
            build=_build_gist_task,
            task_kinds=frozenset({"guided_retrieval"}),
            training_strategies=frozenset({"direct", "grad_cache"}),
        ),
        LossDefinition(
            kind="softmax_classification",
            config_type=SoftmaxClassificationConfig,
            build=_build_softmax_classification_task,
            task_kinds=frozenset({"pair_classification"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="contrastive_tension",
            config_type=ContrastiveTensionConfig,
            build=_build_contrastive_tension_task,
            task_kinds=frozenset({"contrastive_tension_pairs"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="contrastive_tension_in_batch",
            config_type=ContrastiveTensionInBatchConfig,
            build=_build_contrastive_tension_in_batch_task,
            task_kinds=frozenset({"contrastive_tension_examples"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="global_orthogonal_regularization",
            config_type=GORConfig,
            build=_build_gor_task,
            task_kinds=frozenset({"representation_regularization"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="denoising_autoencoder",
            config_type=DenoisingAutoEncoderConfig,
            build=_build_denoising_task,
            task_kinds=frozenset({"denoising_reconstruction"}),
            training_strategies=frozenset({"direct"}),
        ),
        LossDefinition(
            kind="mega_batch_margin",
            config_type=MegaBatchMarginConfig,
            build=_build_mega_batch_task,
            task_kinds=frozenset({"mega_batch"}),
            training_strategies=frozenset({"direct", "mega_batch_mining"}),
        ),
    )
)
BUILTIN_LOSS_MODIFIERS = LossModifierRegistry(
    (
        LossModifierDefinition(
            kind="matryoshka",
            config_type=MatryoshkaModifierConfig,
            build=_build_matryoshka_modifier,
            training_strategies=frozenset({"direct", "grad_cache"}),
        ),
        LossModifierDefinition(
            kind="adaptive_layer",
            config_type=AdaptiveLayerModifierConfig,
            build=_build_adaptive_layer_modifier,
            training_strategies=frozenset({"direct"}),
        ),
        LossModifierDefinition(
            kind="matryoshka_2d",
            config_type=Matryoshka2dModifierConfig,
            build=_build_matryoshka_2d_modifier,
            training_strategies=frozenset({"direct"}),
        ),
    )
)


def build_task(
    task: TaskConfig,
    loss: LossConfig,
    *,
    modifiers: Iterable[LossModifierConfig] = (),
    task_registry: TaskRegistry | None = None,
    loss_registry: LossRegistry | None = None,
    modifier_registry: LossModifierRegistry | None = None,
) -> Any:
    """Build a runtime task from compatible scientific task and loss configs."""

    tasks = BUILTIN_TASKS if task_registry is None else task_registry
    losses = BUILTIN_LOSSES if loss_registry is None else loss_registry
    task = tasks.parse(task)
    loss = losses.parse(loss)
    definition = losses.definition(loss.kind)
    if task.kind not in definition.task_kinds:
        raise ValueError(f"loss {loss.kind!r} does not support task {task.kind!r}")
    runtime_task = losses.build(task, loss)
    modifier_definitions = (
        BUILTIN_LOSS_MODIFIERS if modifier_registry is None else modifier_registry
    )
    for modifier in modifiers:
        runtime_task = modifier_definitions.build(
            runtime_task,
            modifier_definitions.parse(modifier),
        )
    return runtime_task


__all__ = [
    "BUILTIN_LOSS_MODIFIERS",
    "BUILTIN_LOSSES",
    "BUILTIN_TASKS",
    "LossDefinition",
    "LossModifierDefinition",
    "LossModifierRegistry",
    "LossRegistry",
    "TaskDefinition",
    "TaskRegistry",
    "build_task",
]
