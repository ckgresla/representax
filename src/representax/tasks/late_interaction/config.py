"""Serializable late-interaction retrieval configuration."""

from __future__ import annotations

from typing import Literal

from representax._config import FinitePositiveFloat
from representax.tasks.config import LossConfig, TaskConfig


class LateInteractionConfig(TaskConfig):
    """Query/document retrieval through token-level representations."""

    kind: Literal["late_interaction"] = "late_interaction"


class LateInteractionContrastiveConfig(LossConfig):
    """ColBERT-style contrastive MaxSim objective."""

    kind: Literal["late_interaction_contrastive"] = "late_interaction_contrastive"
    temperature: FinitePositiveFloat = 0.02
    symmetric: bool = False
    negative_scope: Literal["local", "global"] = "global"


__all__ = ["LateInteractionConfig", "LateInteractionContrastiveConfig"]
