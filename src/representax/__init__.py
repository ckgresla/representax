"""Representax: task-general representation learning in JAX."""

from . import data, integrations, models, planning, tasks, train
from .config import RunConfig, ScientificConfig
from .core import (
    BoundEncoder,
    Encoder,
    EncoderMetadata,
    LossOutput,
    Modality,
    Route,
    Task,
    bind,
    encode,
    evaluate_loss,
)

__all__ = [
    "BoundEncoder",
    "Encoder",
    "EncoderMetadata",
    "LossOutput",
    "Modality",
    "Route",
    "RunConfig",
    "ScientificConfig",
    "Task",
    "bind",
    "data",
    "encode",
    "evaluate_loss",
    "integrations",
    "models",
    "planning",
    "tasks",
    "train",
]
