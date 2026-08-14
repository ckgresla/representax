"""Serializable user-facing configuration for reproducible training."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator
from typing_extensions import TypeAliasType

from ._config import FrozenConfig
from .data.recipe import MixtureRecipe

NonEmptyString = Annotated[str, Field(min_length=1)]
FinitePositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
RematerializationPolicy = Literal["none", "selective", "full"]
JsonScalar = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)


class ScientificConfig(FrozenConfig):
    """Trajectory-defining scientific values that execution may not change."""

    task: NonEmptyString
    global_batch_size: PositiveInt
    max_steps: PositiveInt
    seed: NonNegativeInt
    negative_scope: Literal["local", "global"] = "global"
    numerical_tolerance: FinitePositiveFloat = 1e-5


class ExecutionConfig(FrozenConfig):
    """A measured, topology-specific realization of a scientific config."""

    device_count: PositiveInt
    per_device_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    data_axis_size: PositiveInt
    model_axis_size: PositiveInt = 1
    query_microbatch_size: PositiveInt | None = None
    document_microbatch_size: PositiveInt | None = None
    rematerialization: RematerializationPolicy = "full"
    packing: bool = False
    prefetch_depth: NonNegativeInt = 2
    donate_buffers: bool = True

    @model_validator(mode="after")
    def validate_mesh(self) -> Self:
        if self.data_axis_size * self.model_axis_size != self.device_count:
            raise ValueError("mesh axis sizes must multiply to device_count")
        return self

    @property
    def effective_batch_size(self) -> int:
        return (
            self.data_axis_size
            * self.per_device_batch_size
            * self.gradient_accumulation_steps
        )

    def validate_scientific(self, scientific: ScientificConfig) -> None:
        if self.effective_batch_size != scientific.global_batch_size:
            raise ValueError(
                "execution configuration changes the scientific global batch size: "
                f"{self.effective_batch_size} != {scientific.global_batch_size}"
            )


class RuntimeConfig(FrozenConfig):
    """Host-loop mechanics that do not define the scientific trajectory."""

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


class TrainingConfig(FrozenConfig):
    """Complete host and accelerator contract for one training run."""

    scientific: ScientificConfig
    execution: ExecutionConfig | None = None
    runtime: RuntimeConfig = RuntimeConfig()
    checkpoint: CheckpointConfig | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        if self.execution is not None:
            self.execution.validate_scientific(self.scientific)
        if self.checkpoint is not None and any(
            iteration > self.scientific.max_steps
            for iteration in self.checkpoint.additional_iterations
        ):
            raise ValueError(
                "additional checkpoint iterations cannot exceed scientific.max_steps"
            )
        return self


class ComponentConfig(FrozenConfig):
    """Serializable reference to one registry/build target and its parameters."""

    target: NonEmptyString
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class RunConfig(FrozenConfig):
    """Fully serializable configuration for one reproducible training run."""

    name: NonEmptyString
    model: ComponentConfig
    optimizer: ComponentConfig
    task: ComponentConfig
    data: MixtureRecipe
    training: TrainingConfig


__all__ = [
    "CheckpointConfig",
    "ComponentConfig",
    "ExecutionConfig",
    "FrozenConfig",
    "RematerializationPolicy",
    "RunConfig",
    "RuntimeConfig",
    "ScientificConfig",
    "TrainingConfig",
]
