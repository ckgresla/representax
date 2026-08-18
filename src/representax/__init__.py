"""Representax: task-general representation learning in JAX."""

from . import data, evaluation, integrations, models, planning, tasks, train
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
from .export import InferenceBundle, export_inference_bundle, load_inference_bundle
from .inference import TextEmbeddingModel, embed

__all__ = [
    "BoundEncoder",
    "Encoder",
    "EncoderMetadata",
    "LossOutput",
    "Modality",
    "Route",
    "JobConfig",
    "InferenceBundle",
    "Task",
    "TextEmbeddingModel",
    "bind",
    "data",
    "encode",
    "embed",
    "evaluation",
    "evaluate_loss",
    "export_inference_bundle",
    "integrations",
    "load_inference_bundle",
    "models",
    "planning",
    "tasks",
    "train",
]
