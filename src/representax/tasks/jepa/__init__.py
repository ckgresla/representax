"""LeJEPA self-supervised representation learning."""

from .batch import JEPABatch, JEPACollator
from .config import (
    JEPAConfig,
    LeJEPAConfig,
    VJEPA2_1DenseConfig,
    VJEPA2_1TaskConfig,
)
from .losses import invariance_loss, sigreg_loss
from .task import LeJEPATask
from .vjepa2_1 import (
    VJEPA2_1Batch,
    VJEPA2_1Task,
    dense_prediction_loss,
    mask_distance_weights,
)

__all__ = [
    "JEPABatch",
    "JEPACollator",
    "JEPAConfig",
    "LeJEPAConfig",
    "LeJEPATask",
    "VJEPA2_1Batch",
    "VJEPA2_1TaskConfig",
    "VJEPA2_1DenseConfig",
    "VJEPA2_1Task",
    "dense_prediction_loss",
    "invariance_loss",
    "sigreg_loss",
    "mask_distance_weights",
]
