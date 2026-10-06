"""Serializable scientific configuration for labeled-pair objectives."""

from __future__ import annotations

from typing import Literal

from representax._config import FinitePositiveFloat
from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig

from .losses import PairDistance


class PairwiseConfig(TaskConfig):
    """Aligned input pairs and the semantic route used for each side."""

    kind: Literal["pairwise"] = "pairwise"
    left_route: Route = Route.GENERIC
    right_route: Route = Route.GENERIC


class CosineRegressionConfig(LossConfig):
    """MSE between pair cosine similarity and a float label."""

    kind: Literal["cosine_regression"] = "cosine_regression"


class ContrastiveConfig(LossConfig):
    """Pairwise contrastive distance with optional online hard mining."""

    kind: Literal["contrastive"] = "contrastive"
    distance: PairDistance = "cosine"
    margin: FinitePositiveFloat = 0.5
    mining: Literal["all", "online"] = "all"


class CoSENTConfig(LossConfig):
    """Cosine score-ordering objective for float-labeled pairs."""

    kind: Literal["cosent"] = "cosent"
    scale: FinitePositiveFloat = 20.0


class AngleConfig(LossConfig):
    """AnglE score-ordering objective for float-labeled pairs."""

    kind: Literal["angle"] = "angle"
    scale: FinitePositiveFloat = 20.0


__all__ = [
    "AngleConfig",
    "CoSENTConfig",
    "ContrastiveConfig",
    "CosineRegressionConfig",
    "PairwiseConfig",
]
