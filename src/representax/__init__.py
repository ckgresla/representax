"""Representax: task-general representation learning in JAX."""

from . import data, integrations, models, planning, tasks, train
from .config import JobConfig
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
from .inference import TextEmbeddingModel, embed

__all__ = [
    "BoundEncoder",
    "Encoder",
    "EncoderMetadata",
    "LossOutput",
    "Modality",
    "Route",
    "JobConfig",
    "Task",
    "TextEmbeddingModel",
    "bind",
    "data",
    "encode",
    "embed",
    "evaluate_loss",
    "integrations",
    "models",
    "planning",
    "tasks",
    "train",
]
