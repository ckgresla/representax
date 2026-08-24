"""Late-interaction representation learning."""

from .config import LateInteractionConfig, LateInteractionContrastiveConfig
from .scoring import maxsim_scores
from .task import (
    LateInteractionLossTerms,
    LateInteractionTask,
    late_interaction_loss_terms,
)

__all__ = [
    "LateInteractionConfig",
    "LateInteractionContrastiveConfig",
    "LateInteractionLossTerms",
    "LateInteractionTask",
    "late_interaction_loss_terms",
    "maxsim_scores",
]
