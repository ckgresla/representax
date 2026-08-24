"""Public core contracts."""

from .late_interaction import (
    LateInteractionEncoder,
    LateInteractionRepresentation,
    encode_late_interaction,
)
from .model import (
    BUILTIN_MODALITIES,
    BoundEncoder,
    Encoder,
    EncoderMetadata,
    LayerwiseEncoder,
    Modality,
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
    "LateInteractionEncoder",
    "LateInteractionRepresentation",
    "EncodeFunction",
    "LossOutput",
    "Modality",
    "Route",
    "RepresentationTask",
    "Task",
    "bind",
    "encode",
    "encode_layers",
    "encode_late_interaction",
    "evaluate_loss",
]
