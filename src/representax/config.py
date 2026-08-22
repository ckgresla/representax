"""Serializable user-facing configuration for reproducible training."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SerializeAsAny,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing_extensions import TypeAliasType

from ._config import (
    Execution,
    FrozenConfig,
    NonEmptyString,
    ParameterRole,
    Scientific,
    project_parameters,
)
from .core import Route
from .data.distribution import DataDistributionConfig
from .tasks.config import LossConfig, LossModifierConfig, TaskConfig

RematerializationPolicy = Literal["none", "selective", "full"]
PrecisionDType = Literal["float32", "bfloat16"]
MatrixDType = Literal["float32", "bfloat16", "float8_e4m3fn"]
MetricMode = Literal["min", "max"]
ExportSelection = Literal["final", "best"]
JsonScalar = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)


class ComponentConfig(FrozenConfig):
    """Serializable reference to one registry/build target and its parameters."""

    target: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ModelConfig(ComponentConfig):
    """Model definition consumed by a model registry or builder."""


class OptimizationConfig(FrozenConfig):
    """Optimizer transformation and its scientific hyperparameters."""

    optimizer: ComponentConfig
    schedule: ComponentConfig | None = None
    schedule_parameter: NonEmptyString = "learning_rate"
    max_gradient_norm: Scientific[PositiveFloat | None] = 1.0


class DataConfig(FrozenConfig):
    """Reproducible source distribution plus native Grain execution policy."""

    distribution: Scientific[DataDistributionConfig]
    collate: Scientific[ComponentConfig | None] = None
    drop_remainder: Scientific[bool] = True
    num_threads: Execution[NonNegativeInt] = 16
    prefetch_buffer_size: Execution[NonNegativeInt] = 16
    host_memory_budget_bytes: Execution[PositiveInt | None] = None
    data_wait_heartbeat_seconds: Execution[PositiveFloat | None] = None
    data_wait_timeout_seconds: Execution[PositiveFloat | None] = None

    @model_validator(mode="before")
    @classmethod
    def wrap_distribution(cls, value: object) -> object:
        """Accept the concise distribution body as a complete data config."""

        if isinstance(value, DataDistributionConfig):
            return {"distribution": value}
        if (
            isinstance(value, Mapping)
            and "distribution" not in value
            and "sources" in value
        ):
            return {"distribution": value}
        return value

    @model_validator(mode="after")
    def validate_data_wait_thresholds(self) -> Self:
        """Keep liveness heartbeats strictly inside the fatal deadline."""

        if (
            self.data_wait_heartbeat_seconds is not None
            and self.data_wait_timeout_seconds is not None
            and self.data_wait_heartbeat_seconds >= self.data_wait_timeout_seconds
        ):
            raise ValueError(
                "data_wait_heartbeat_seconds must be below data_wait_timeout_seconds"
            )
        return self


class EvaluatorConfig(FrozenConfig):
    """Training-objective evaluator and its stable metric namespace."""

    kind: Literal["loss"] = "loss"
    name: NonEmptyString = "loss"


class EmbeddingSimilarityEvaluatorConfig(EvaluatorConfig):
    """Corpus-level correlations between labels and paired similarities."""

    kind: Literal["embedding_similarity"] = "embedding_similarity"
    name: NonEmptyString = "similarity"
    similarity_functions: tuple[
        Literal["cosine", "dot", "euclidean", "manhattan"], ...
    ] = ("cosine", "dot", "euclidean", "manhattan")
    main_similarity: Literal["cosine", "dot", "euclidean", "manhattan"] | None = None
    left_route: Route = Route.GENERIC
    right_route: Route = Route.GENERIC

    @model_validator(mode="after")
    def validate_similarity(self) -> Self:
        if not self.similarity_functions:
            raise ValueError("at least one similarity function is required")
        if len(set(self.similarity_functions)) != len(self.similarity_functions):
            raise ValueError("similarity functions must be unique")
        if (
            self.main_similarity is not None
            and self.main_similarity not in self.similarity_functions
        ):
            raise ValueError("main_similarity must be one of similarity_functions")
        return self


class EvaluationConfig(FrozenConfig):
    """Offline-compatible validation data, cadence, and model selection."""

    data: DataConfig
    batch_size: Scientific[PositiveInt]
    evaluators: Scientific[tuple[SerializeAsAny[EvaluatorConfig], ...]] = (
        EvaluatorConfig(),
    )
    every_steps: PositiveInt | None = None
    on_start: bool = False
    on_end: bool = True
    max_batches: Scientific[PositiveInt | None] = None
    primary_metric: Scientific[NonEmptyString] = "valid/loss"
    primary_metric_mode: Scientific[MetricMode] = "min"
    save_best: bool = True
    keep_best: PositiveInt = 1

    @field_validator("evaluators", mode="before")
    @classmethod
    def validate_evaluators(cls, value: object) -> object:
        if not isinstance(value, (list, tuple)):
            raise TypeError("evaluators must be a list or tuple")
        parsed = []
        for evaluator in value:
            if isinstance(evaluator, EvaluatorConfig):
                parsed.append(evaluator)
                continue
            if not isinstance(evaluator, Mapping):
                raise TypeError("each evaluator must be a config or mapping")
            kind = evaluator.get("kind", "loss")
            if kind == "loss":
                parsed.append(EvaluatorConfig.model_validate(evaluator))
            elif kind == "embedding_similarity":
                parsed.append(
                    EmbeddingSimilarityEvaluatorConfig.model_validate(evaluator)
                )
            else:
                raise ValueError(f"unknown evaluator kind {kind!r}")
        return tuple(parsed)

    @model_validator(mode="after")
    def validate_evaluation(self) -> Self:
        names = tuple(evaluator.name for evaluator in self.evaluators)
        if not names:
            raise ValueError("evaluation requires at least one evaluator")
        if len(set(names)) != len(names):
            raise ValueError("evaluator names must be unique")
        if not self.primary_metric.startswith("valid/"):
            raise ValueError("primary_metric must use the valid/ namespace")
        if self.save_best and not (
            self.on_start or self.on_end or self.every_steps is not None
        ):
            raise ValueError("best-model selection requires an evaluation cadence")
        return self


class HuggingFaceExportConfig(FrozenConfig):
    """Optional verified export through a native checkpoint adapter."""

    source_checkpoint: NonEmptyString
    adapter: ComponentConfig
    verify_reload: bool = True


class ExportConfig(FrozenConfig):
    """Atomic inference artifact publication after successful training."""

    enabled: bool = True
    selection: ExportSelection = "final"
    directory_name: NonEmptyString = "final-model"
    huggingface: HuggingFaceExportConfig | None = None

    @field_validator("directory_name")
    @classmethod
    def validate_directory_name(cls, value: str) -> str:
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("export directory_name must be one path component")
        return value


class MeshConfig(FrozenConfig):
    """Serializable logical mesh shape and axis names."""

    axis_shapes: tuple[PositiveInt, ...] = (1,)
    axis_names: tuple[NonEmptyString, ...] = ("data",)

    @model_validator(mode="after")
    def validate_axes(self) -> Self:
        if len(self.axis_shapes) != len(self.axis_names):
            raise ValueError("mesh axis_shapes and axis_names must have equal length")
        if not self.axis_names:
            raise ValueError("a mesh must contain at least one logical axis")
        if len(set(self.axis_names)) != len(self.axis_names):
            raise ValueError("mesh axis names must be unique")
        return self

    @property
    def device_count(self) -> int:
        return math.prod(self.axis_shapes)

    def axis_size(self, name: str) -> int:
        try:
            index = self.axis_names.index(name)
        except ValueError as error:
            raise KeyError(f"mesh axis {name!r} is not defined") from error
        return self.axis_shapes[index]


PartitionAxis = NonEmptyString | tuple[NonEmptyString, ...] | None


class PartitionRuleConfig(FrozenConfig):
    """Map model-leaf paths to one serializable ``PartitionSpec``."""

    pattern: NonEmptyString
    axes: tuple[PartitionAxis, ...] = ()

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(
                f"invalid partition-rule regular expression: {error}"
            ) from error
        return value

    @field_validator("axes")
    @classmethod
    def validate_partition_axes(
        cls,
        value: tuple[PartitionAxis, ...],
    ) -> tuple[PartitionAxis, ...]:
        names: list[str] = []
        for axis in value:
            if isinstance(axis, tuple) and not axis:
                raise ValueError("a grouped partition axis cannot be empty")
            if axis is None:
                continue
            names.extend(axis if isinstance(axis, tuple) else (axis,))
        if len(set(names)) != len(names):
            raise ValueError("a partition rule cannot reuse one mesh axis")
        return value


class DDPConfig(FrozenConfig):
    """Named replicated-state data parallelism preset."""

    kind: Literal["ddp"] = "ddp"
    axis: NonEmptyString = "data"


class FSDPConfig(FrozenConfig):
    """Named fully-sharded data parallelism preset."""

    kind: Literal["fsdp"] = "fsdp"
    data_axis: NonEmptyString = "data"
    parameter_axis: NonEmptyString | None = None
    minimum_parameter_elements: PositiveInt = 2**18

    @property
    def resolved_parameter_axis(self) -> str:
        return self.parameter_axis or self.data_axis


class CustomShardingConfig(FrozenConfig):
    """Explicit parameter layouts over a compatible named JAX mesh."""

    kind: Literal["custom"] = "custom"
    data_axis: NonEmptyString | None = None
    parameter_axes: tuple[NonEmptyString, ...]
    parameter_rules: tuple[PartitionRuleConfig, ...]
    default_parameter_axes: tuple[PartitionAxis, ...] = ()

    @model_validator(mode="after")
    def validate_custom_sharding(self) -> Self:
        if not self.parameter_axes:
            raise ValueError("custom sharding requires at least one parameter axis")
        if len(set(self.parameter_axes)) != len(self.parameter_axes):
            raise ValueError("custom parameter axes must be unique")
        if not self.parameter_rules:
            raise ValueError("custom sharding requires at least one parameter rule")
        return self


ShardingConfig = Annotated[
    DDPConfig | FSDPConfig | CustomShardingConfig,
    Field(discriminator="kind"),
]


class BatchConfig(FrozenConfig):
    """Per-data-replica batch size and optimizer accumulation."""

    micro_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt = 1


class GradCacheConfig(FrozenConfig):
    """Memory bounds for exact cached differentiation of representation losses."""

    micro_batch_size: PositiveInt = Field(
        description="Default number of examples encoded by each replayed forward pass."
    )
    query_micro_batch_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Optional query-encoder override for asymmetric or multimodal towers; "
            "defaults to micro_batch_size."
        ),
    )
    document_micro_batch_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Optional document-encoder override for asymmetric or multimodal towers; "
            "defaults to micro_batch_size."
        ),
    )
    loss_row_chunk_size: PositiveInt | None = Field(
        default=None,
        description=(
            "Rows evaluated at once in the representation-level similarity loss; "
            "defaults to micro_batch_size and does not rerun either encoder."
        ),
    )

    @property
    def resolved_query_micro_batch_size(self) -> int:
        return self.query_micro_batch_size or self.micro_batch_size

    @property
    def resolved_document_micro_batch_size(self) -> int:
        return self.document_micro_batch_size or self.micro_batch_size

    @property
    def resolved_loss_row_chunk_size(self) -> int:
        return self.loss_row_chunk_size or self.micro_batch_size


class MegaBatchMiningConfig(FrozenConfig):
    """Memory bounds for frozen-candidate mega-batch hard-negative mining."""

    micro_batch_size: PositiveInt
    loss_row_chunk_size: PositiveInt | None = None

    @property
    def resolved_loss_row_chunk_size(self) -> int:
        return self.loss_row_chunk_size or self.micro_batch_size


class PrecisionConfig(FrozenConfig):
    """Serializable execution dtypes at each numerical boundary."""

    parameter_dtype: Literal["float32"] = Field(
        default="float32",
        description="Persistent model and checkpoint dtype.",
    )
    compute_dtype: PrecisionDType = Field(
        default="float32",
        description="Transient parameter view used by the forward program.",
    )
    activation_dtype: PrecisionDType = Field(
        default="float32",
        description="Floating model-input and hidden-activation dtype.",
    )
    matrix_dtype: MatrixDType | None = Field(
        default=None,
        description=("Scaled linear-algebra operand dtype; defaults to compute_dtype."),
    )
    accumulation_dtype: Literal["float32"] = Field(
        default="float32",
        description="Representation, gradient, and metric reduction dtype.",
    )
    loss_dtype: Literal["float32"] = Field(
        default="float32",
        description="Sensitive objective and scalar-loss dtype.",
    )

    @classmethod
    def bfloat16_mixed(cls) -> Self:
        """Conventional BF16 compute with FP32 master and objective state."""

        return cls(compute_dtype="bfloat16", activation_dtype="bfloat16")

    @classmethod
    def float8_mixed(cls) -> Self:
        """Experimental FP8 matrices with BF16 communication and other compute."""

        return cls(
            compute_dtype="bfloat16",
            activation_dtype="bfloat16",
            matrix_dtype="float8_e4m3fn",
        )

    @property
    def resolved_matrix_dtype(self) -> MatrixDType:
        """Resolve the matrix operand dtype against ordinary compute."""

        return self.matrix_dtype or self.compute_dtype

    @property
    def communication_dtype(self) -> PrecisionDType:
        """FSDP communicates the transient compute view, not FP32 masters."""

        return self.compute_dtype


class QuantizedLoRAConfig(FrozenConfig):
    """Four-bit frozen base weights plus trainable low-rank adapters."""

    bits: Literal[4] = 4
    rank: PositiveInt
    alpha: PositiveFloat
    target_pattern: NonEmptyString = ".*"
    initialization_scale: PositiveFloat | None = None

    @field_validator("target_pattern")
    @classmethod
    def validate_target_pattern(cls, pattern: str) -> str:
        """Reject invalid target-path regular expressions during configuration."""

        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"invalid adapter target_pattern: {error}") from error
        return pattern


class TrainingConfig(FrozenConfig):
    """Scientific and efficiency parameters governing the training process."""

    global_batch_size: Scientific[PositiveInt]
    max_steps: Scientific[PositiveInt]
    seed: Scientific[NonNegativeInt]
    mesh: Execution[MeshConfig] = MeshConfig()
    sharding: Execution[ShardingConfig] = DDPConfig()
    batch: Execution[BatchConfig]
    grad_cache: Execution[GradCacheConfig | None] = None
    mega_batch_mining: Execution[MegaBatchMiningConfig | None] = None
    adapter: Scientific[QuantizedLoRAConfig | None] = None
    precision: Execution[PrecisionConfig] = PrecisionConfig()
    activation_rematerialization: Execution[RematerializationPolicy] = Field(
        default="full",
        description=(
            "Activation checkpointing policy (also called gradient checkpointing); "
            "JAX calls this rematerialization."
        ),
    )
    donate_buffers: Execution[bool] = True

    @model_validator(mode="after")
    def validate_loss_execution(self) -> Self:
        if self.grad_cache is not None and self.mega_batch_mining is not None:
            raise ValueError("configure only one specialized loss execution")
        return self

    @model_validator(mode="after")
    def validate_sharding_axes(self) -> Self:
        mesh_axes = set(self.mesh.axis_names)
        sharding = self.sharding
        if isinstance(sharding, DDPConfig):
            referenced_axes = {sharding.axis}
        elif isinstance(sharding, FSDPConfig):
            referenced_axes = {
                sharding.data_axis,
                sharding.resolved_parameter_axis,
            }
        else:
            referenced_axes = set(sharding.parameter_axes)
            if sharding.data_axis is not None:
                referenced_axes.add(sharding.data_axis)
            rule_axes = {
                name
                for rule in sharding.parameter_rules
                for axis in rule.axes
                if axis is not None
                for name in (axis if isinstance(axis, tuple) else (axis,))
            }
            default_axes = {
                name
                for axis in sharding.default_parameter_axes
                if axis is not None
                for name in (axis if isinstance(axis, tuple) else (axis,))
            }
            unsupported = (rule_axes | default_axes) - set(sharding.parameter_axes)
            if unsupported:
                raise ValueError(
                    "partition rules use axes absent from parameter_axes: "
                    f"{sorted(unsupported)}"
                )
        unknown = referenced_axes - mesh_axes
        if unknown:
            raise ValueError(
                f"sharding axes are absent from the mesh: {sorted(unknown)}"
            )
        return self


class LoggingConfig(FrozenConfig):
    """Asynchronous local and downstream reporting mechanics."""

    console_every: PositiveInt = 1
    reporter_queue_size: PositiveInt = 16


class CheckpointConfig(FrozenConfig):
    """Checkpoint cadence and bounded retention."""

    every: PositiveInt
    keep: PositiveInt = 3
    additional_iterations: tuple[PositiveInt, ...] = ()
    save_final: bool = True
    asynchronous: bool = True

    @model_validator(mode="after")
    def validate_iterations(self) -> Self:
        if len(set(self.additional_iterations)) != len(self.additional_iterations):
            raise ValueError("additional checkpoint iterations must be unique")
        return self

    def should_save(self, iteration: int, *, final: bool) -> bool:
        return (
            iteration % self.every == 0
            or iteration in self.additional_iterations
            or (final and self.save_final)
        )


class JobConfig(FrozenConfig):
    """Fully serializable configuration for one reproducible training job."""

    name: NonEmptyString
    model: Scientific[ModelConfig]
    task: Scientific[SerializeAsAny[TaskConfig]]
    loss: Scientific[SerializeAsAny[LossConfig]]
    loss_modifiers: Scientific[tuple[SerializeAsAny[LossModifierConfig], ...]] = ()
    optimization: Scientific[OptimizationConfig]
    data: DataConfig
    training: TrainingConfig
    checkpointing: CheckpointConfig | None = None
    logging: LoggingConfig = LoggingConfig()
    evaluation: EvaluationConfig | None = None
    export: ExportConfig = ExportConfig()

    @field_validator("task", mode="before")
    @classmethod
    def validate_registered_task(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> TaskConfig:
        from .tasks.registry import BUILTIN_TASKS, TaskRegistry

        registry = None if info.context is None else info.context.get("task_registry")
        if registry is None:
            registry = BUILTIN_TASKS
        if not isinstance(registry, TaskRegistry):
            raise TypeError("task_registry validation context must be a TaskRegistry")
        return registry.parse(value)

    @field_validator("loss", mode="before")
    @classmethod
    def validate_registered_loss(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> LossConfig:
        from .tasks.registry import BUILTIN_LOSSES, LossRegistry

        registry = None if info.context is None else info.context.get("loss_registry")
        if registry is None:
            registry = BUILTIN_LOSSES
        if not isinstance(registry, LossRegistry):
            raise TypeError("loss_registry validation context must be a LossRegistry")
        return registry.parse(value)

    @field_validator("loss_modifiers", mode="before")
    @classmethod
    def validate_registered_loss_modifiers(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> tuple[LossModifierConfig, ...]:
        from .tasks.registry import BUILTIN_LOSS_MODIFIERS, LossModifierRegistry

        registry = (
            None if info.context is None else info.context.get("modifier_registry")
        )
        if registry is None:
            registry = BUILTIN_LOSS_MODIFIERS
        if not isinstance(registry, LossModifierRegistry):
            raise TypeError(
                "modifier_registry validation context must be a LossModifierRegistry"
            )
        if not isinstance(value, (list, tuple)):
            raise TypeError("loss_modifiers must be a list or tuple")
        parsed = tuple(registry.parse(modifier) for modifier in value)
        kinds = tuple(modifier.kind for modifier in parsed)
        if len(set(kinds)) != len(kinds):
            raise ValueError("loss modifier kinds must be unique")
        if len(parsed) > 1 and any(
            kind in {"matryoshka", "adaptive_layer", "matryoshka_2d"} for kind in kinds
        ):
            raise ValueError(
                "dimension/layer modifiers are mutually exclusive; use "
                "matryoshka_2d for their joint objective"
            )
        return parsed

    @model_validator(mode="after")
    def validate_loss_capabilities(self, info: ValidationInfo) -> Self:
        from .tasks.registry import (
            BUILTIN_LOSS_MODIFIERS,
            BUILTIN_LOSSES,
            LossModifierRegistry,
            LossRegistry,
        )

        registered = None if info.context is None else info.context.get("loss_registry")
        losses = BUILTIN_LOSSES if registered is None else registered
        if not isinstance(losses, LossRegistry):
            raise TypeError("loss_registry validation context must be a LossRegistry")
        definition = losses.definition(self.loss.kind)
        if self.task.kind not in definition.task_kinds:
            raise ValueError(
                f"loss {self.loss.kind!r} does not support task {self.task.kind!r}"
            )
        if self.training.grad_cache is not None:
            strategy = "grad_cache"
        elif self.training.mega_batch_mining is not None:
            strategy = "mega_batch_mining"
        else:
            strategy = "direct"
        if strategy not in definition.training_strategies:
            raise ValueError(
                f"loss {self.loss.kind!r} does not support training strategy "
                f"{strategy!r}"
            )
        accumulation_steps = self.training.batch.gradient_accumulation_steps
        if accumulation_steps > 1:
            if strategy != "direct":
                raise ValueError(
                    "gradient accumulation composes only with direct execution; "
                    "GradCache and mega-batch execution already own their logical "
                    "batch decomposition"
                )
            if not definition.microbatch_accumulation:
                raise ValueError(
                    f"loss {self.loss.kind!r} does not decompose exactly across "
                    "gradient-accumulation microbatches"
                )
        modifiers = (
            None if info.context is None else info.context.get("modifier_registry")
        )
        if modifiers is None:
            modifiers = BUILTIN_LOSS_MODIFIERS
        if not isinstance(modifiers, LossModifierRegistry):
            raise TypeError(
                "modifier_registry validation context must be a LossModifierRegistry"
            )
        for modifier in self.loss_modifiers:
            modifier_definition = modifiers.definition(modifier.kind)
            if strategy not in modifier_definition.training_strategies:
                raise ValueError(
                    f"loss modifier {modifier.kind!r} does not support training "
                    f"strategy {strategy!r}"
                )
            if accumulation_steps > 1 and not (
                modifier_definition.microbatch_accumulation
            ):
                raise ValueError(
                    f"loss modifier {modifier.kind!r} does not decompose exactly "
                    "across gradient-accumulation microbatches"
                )
        return self

    @model_validator(mode="after")
    def validate_checkpoint_schedule(self) -> Self:
        if self.checkpointing is not None and any(
            iteration > self.training.max_steps
            for iteration in self.checkpointing.additional_iterations
        ):
            raise ValueError(
                "additional checkpoint iterations cannot exceed training.max_steps"
            )
        if (
            self.evaluation is not None
            and self.evaluation.save_best
            and self.checkpointing is None
        ):
            raise ValueError("best-model selection requires checkpointing")
        if (
            self.export.enabled
            and self.export.selection == "best"
            and (self.evaluation is None or not self.evaluation.save_best)
        ):
            raise ValueError("best-model export requires evaluation.save_best")
        return self

    def parameters(self, role: ParameterRole) -> dict[str, object]:
        """Return the role-selected job projection used by planners and resume."""

        return project_parameters(self, role)


__all__ = [
    "BatchConfig",
    "CheckpointConfig",
    "ComponentConfig",
    "CustomShardingConfig",
    "DataConfig",
    "DDPConfig",
    "EmbeddingSimilarityEvaluatorConfig",
    "EvaluationConfig",
    "EvaluatorConfig",
    "ExportConfig",
    "FSDPConfig",
    "GradCacheConfig",
    "HuggingFaceExportConfig",
    "JobConfig",
    "LoggingConfig",
    "MegaBatchMiningConfig",
    "MeshConfig",
    "ModelConfig",
    "OptimizationConfig",
    "ParameterRole",
    "PartitionAxis",
    "PartitionRuleConfig",
    "RematerializationPolicy",
    "ShardingConfig",
    "TrainingConfig",
]
