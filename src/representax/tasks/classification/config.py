"""Serializable pair-classification configuration."""

from __future__ import annotations

from typing import Literal

from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig


class PairClassificationConfig(TaskConfig):
    """Aligned pair classification and its semantic encoder routes."""

    kind: Literal["pair_classification"] = "pair_classification"
    left_route: Route = Route.GENERIC
    right_route: Route = Route.GENERIC


class SoftmaxClassificationConfig(LossConfig):
    """Sentence Transformers-compatible pair feature policy."""

    kind: Literal["softmax_classification"] = "softmax_classification"
    concatenate_representations: bool = True
    concatenate_difference: bool = True
    concatenate_product: bool = False


__all__ = ["PairClassificationConfig", "SoftmaxClassificationConfig"]
