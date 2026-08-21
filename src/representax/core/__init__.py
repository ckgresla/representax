"""Public core contracts."""

from .model import (
    BUILTIN_MODALITIES,
    BoundEncoder,
    Encoder,
    EncoderMetadata,
    LayerwiseEncoder,
    Modality,
    ModelBundle,
    Processor,
    Route,
    bind,
    encode,
    encode_layers,
)
from .task import EncodeFunction, LossOutput, RepresentationTask, Task, evaluate_loss

__all__ = [
    "BUILTIN_MODALITIES",
    "BoundEncoder",
    "Encoder",
    "EncoderMetadata",
    "LayerwiseEncoder",
    "ModelBundle",
    "EncodeFunction",
    "LossOutput",
    "Modality",
    "Processor",
    "Route",
    "RepresentationTask",
    "Task",
    "bind",
    "encode",
    "encode_layers",
    "evaluate_loss",
]
