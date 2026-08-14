"""Scientific intent and topology-dependent execution choices."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

RematerializationPolicy = Literal["none", "selective", "full"]


@dataclass(frozen=True)
class ScientificSpec:
    """Experiment semantics that an execution planner may not change."""

    task: str
    global_batch_size: int
    max_steps: int
    seed: int
    negative_scope: Literal["local", "global"] = "global"
    numerical_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not self.task:
            raise ValueError("task must be non-empty")
        if self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not math.isfinite(self.numerical_tolerance) or self.numerical_tolerance <= 0:
            raise ValueError("numerical_tolerance must be finite and positive")


@dataclass(frozen=True)
class ExecutionPlan:
    """A measured, topology-specific way to satisfy a ScientificSpec."""

    device_count: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    data_axis_size: int
    model_axis_size: int = 1
    query_microbatch_size: int | None = None
    document_microbatch_size: int | None = None
    rematerialization: RematerializationPolicy = "full"
    packing: bool = False
    prefetch_depth: int = 2
    donate_buffers: bool = True

    def __post_init__(self) -> None:
        positive = {
            "device_count": self.device_count,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "data_axis_size": self.data_axis_size,
            "model_axis_size": self.model_axis_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.data_axis_size * self.model_axis_size != self.device_count:
            raise ValueError("mesh axis sizes must multiply to device_count")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be non-negative")
        if self.rematerialization not in {"none", "selective", "full"}:
            raise ValueError(
                "rematerialization must be 'none', 'selective', or 'full'"
            )
        for name, value in (
            ("query_microbatch_size", self.query_microbatch_size),
            ("document_microbatch_size", self.document_microbatch_size),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")

    @property
    def effective_batch_size(self) -> int:
        return (
            self.data_axis_size
            * self.per_device_batch_size
            * self.gradient_accumulation_steps
        )

    def validate_science(self, science: ScientificSpec) -> None:
        if self.effective_batch_size != science.global_batch_size:
            raise ValueError(
                "execution plan changes the scientific global batch size: "
                f"{self.effective_batch_size} != {science.global_batch_size}"
            )


class ExecutionPlanner(Protocol):
    """Boundary that a future Profilax implementation can satisfy."""

    def plan(self, science: ScientificSpec) -> ExecutionPlan: ...
