"""Task families and their explicit construction registry."""

from . import distillation, pairwise, retrieval, triplet
from .config import LossConfig, TaskConfig
from .registry import (
    BUILTIN_LOSSES,
    BUILTIN_TASKS,
    LossDefinition,
    LossRegistry,
    TaskDefinition,
    TaskRegistry,
    build_task,
)

__all__ = [
    "BUILTIN_LOSSES",
    "BUILTIN_TASKS",
    "LossConfig",
    "LossDefinition",
    "LossRegistry",
    "TaskDefinition",
    "TaskConfig",
    "TaskRegistry",
    "build_task",
    "distillation",
    "pairwise",
    "retrieval",
    "triplet",
]
