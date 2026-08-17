"""Serializable denoising reconstruction configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import NonNegativeInt

from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig


class DenoisingConfig(TaskConfig):
    kind: Literal["denoising_reconstruction"] = "denoising_reconstruction"
    route: Route = Route.GENERIC


class DenoisingAutoEncoderConfig(LossConfig):
    kind: Literal["denoising_autoencoder"] = "denoising_autoencoder"
    pad_token_id: NonNegativeInt


__all__ = ["DenoisingAutoEncoderConfig", "DenoisingConfig"]
