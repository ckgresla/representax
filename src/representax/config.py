"""Composable, Hydra-Zen-friendly complete run recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data import MixtureRecipe
from .planning import ExecutionPlan, ScientificSpec


@dataclass(frozen=True)
class RunRecipe:
    """Fully reviewable configuration for one reproducible training run."""

    name: str
    model: Any
    task: Any
    data: MixtureRecipe
    science: ScientificSpec
    execution: ExecutionPlan | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("run recipe name must be non-empty")
        if self.execution is not None:
            self.execution.validate_science(self.science)
