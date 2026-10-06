"""Serializable task and loss configuration for triplet learning."""

from __future__ import annotations

from typing import Literal

from representax._config import FinitePositiveFloat
from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig

from .losses import BatchTripletDistance, ExplicitTripletDistance


class ExplicitTripletConfig(TaskConfig):
    """Aligned triplets and the semantic route used for each member."""

    kind: Literal["explicit_triplet"] = "explicit_triplet"
    anchor_route: Route = Route.GENERIC
    positive_route: Route = Route.GENERIC
    negative_route: Route = Route.GENERIC


class LabeledExamplesConfig(TaskConfig):
    """Class-labeled examples used for within-batch mining."""

    kind: Literal["labeled_examples"] = "labeled_examples"
    route: Route = Route.GENERIC


class ExplicitTripletLossConfig(LossConfig):
    """Margin triplet loss over supplied anchor-positive-negative rows."""

    kind: Literal["triplet"] = "triplet"
    distance: ExplicitTripletDistance = "euclidean"
    margin: FinitePositiveFloat = 5.0


class BatchTripletLossConfig(LossConfig):
    """Margin triplet loss with all, hard, or semi-hard in-batch mining."""

    kind: Literal["batch_triplet"] = "batch_triplet"
    mining: Literal["all", "hard", "semi_hard"] = "hard"
    distance: BatchTripletDistance = "euclidean"
    margin: FinitePositiveFloat = 5.0


class BatchHardSoftMarginLossConfig(LossConfig):
    """Hard in-batch mining with a softplus margin instead of a fixed margin."""

    kind: Literal["batch_hard_soft_margin"] = "batch_hard_soft_margin"
    distance: BatchTripletDistance = "euclidean"


__all__ = [
    "BatchHardSoftMarginLossConfig",
    "BatchTripletLossConfig",
    "ExplicitTripletConfig",
    "ExplicitTripletLossConfig",
    "LabeledExamplesConfig",
]
