"""Protocol for scientific-parameter-preserving execution planning."""

from __future__ import annotations

from typing import Protocol

from representax.config import JobConfig


class ExecutionPlanner(Protocol):
    """Boundary that a future Profilax implementation can satisfy."""

    def plan(self, job: JobConfig) -> JobConfig: ...


__all__ = ["ExecutionPlanner"]
