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
from .scoring import Scorer, score_logits
from .task import (
    EncodeFunction,
    LossOutput,
    PostUpdateTask,
    RepresentationTask,
    Task,
    evaluate_loss,
)

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
    "PostUpdateTask",
    "Route",
    "Scorer",
    "RepresentationTask",
    "Task",
    "bind",
    "encode",
    "encode_layers",
    "encode_late_interaction",
    "evaluate_loss",
    "score_logits",
]
