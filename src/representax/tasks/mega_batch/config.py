"""Serializable mega-batch margin configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeFloat

from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig


class MegaBatchConfig(TaskConfig):
    kind: Literal["mega_batch"] = "mega_batch"
    anchor_route: Route = Route.GENERIC
    positive_route: Route = Route.GENERIC


class MegaBatchMarginConfig(LossConfig):
    kind: Literal["mega_batch_margin"] = "mega_batch_margin"
    positive_margin: NonNegativeFloat = 0.8
    negative_margin: NonNegativeFloat = 0.3


__all__ = ["MegaBatchConfig", "MegaBatchMarginConfig"]
