"""Serializable contrastive-tension configuration."""

from __future__ import annotations

from typing import Literal

from representax.tasks.config import LossConfig, TaskConfig


class ContrastiveTensionPairsConfig(TaskConfig):
    kind: Literal["contrastive_tension_pairs"] = "contrastive_tension_pairs"


class ContrastiveTensionExamplesConfig(TaskConfig):
    kind: Literal["contrastive_tension_examples"] = "contrastive_tension_examples"


class ContrastiveTensionConfig(LossConfig):
    kind: Literal["contrastive_tension"] = "contrastive_tension"


class ContrastiveTensionInBatchConfig(LossConfig):
    kind: Literal["contrastive_tension_in_batch"] = "contrastive_tension_in_batch"
    similarity: Literal["cosine", "dot"] = "cosine"


__all__ = [
    "ContrastiveTensionConfig",
    "ContrastiveTensionExamplesConfig",
    "ContrastiveTensionInBatchConfig",
    "ContrastiveTensionPairsConfig",
]
