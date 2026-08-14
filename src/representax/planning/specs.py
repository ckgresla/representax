"""Protocol for scientific-config-preserving execution planning."""

from __future__ import annotations

from typing import Protocol

from representax.config import ExecutionConfig, ScientificConfig


class ExecutionPlanner(Protocol):
    """Boundary that a future Profilax implementation can satisfy."""

    def plan(self, scientific: ScientificConfig) -> ExecutionConfig: ...


__all__ = ["ExecutionPlanner"]
