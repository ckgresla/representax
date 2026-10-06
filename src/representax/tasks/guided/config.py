"""Serializable scientific configuration for guide-filtered retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeFloat

from representax._config import FinitePositiveFloat
from representax.tasks.config import LossConfig, TaskConfig


class GuidedRetrievalConfig(TaskConfig):
    """Aligned retrieval examples carrying offline guide representations."""

    kind: Literal["guided_retrieval"] = "guided_retrieval"


class GISTConfig(LossConfig):
    """Scientific GISTEmbed false-negative filtering policy."""

    kind: Literal["gist"] = "gist"
    temperature: FinitePositiveFloat = 0.01
    margin_strategy: Literal["absolute", "relative"] = "absolute"
    margin: NonNegativeFloat = 0.0
    contrast_anchors: bool = True
    contrast_positives: bool = True


__all__ = ["GISTConfig", "GuidedRetrievalConfig"]
