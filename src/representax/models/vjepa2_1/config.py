"""Validated architecture configuration for native V-JEPA 2.1."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from representax._config import FrozenConfig


class VJEPA2_1Config(FrozenConfig):
    """Shared image/video encoder and dense predictor architecture."""

    image_size: int = 256
    patch_size: int = 16
    video_frames: int = 16
    tubelet_size: int = 2
    channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    predictor_hidden_size: int = 384
    predictor_depth: int = 12
    predictor_heads: int = 12
    supervision_layers: tuple[int, ...] = (2, 5, 8, 11)
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02

    @property
    def spatial_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def image_token_count(self) -> int:
        return self.spatial_grid**2

    @property
    def video_token_count(self) -> int:
        return self.video_frames // self.tubelet_size * self.spatial_grid**2

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        positive = (
            "image_size",
            "patch_size",
            "video_frames",
            "tubelet_size",
            "channels",
            "hidden_size",
            "depth",
            "heads",
            "predictor_hidden_size",
            "predictor_depth",
            "predictor_heads",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.video_frames % self.tubelet_size:
            raise ValueError("video_frames must be divisible by tubelet_size")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.predictor_hidden_size % self.predictor_heads:
            raise ValueError(
                "predictor_hidden_size must be divisible by predictor_heads"
            )
        if not self.supervision_layers:
            raise ValueError("at least one supervision layer is required")
        if tuple(sorted(set(self.supervision_layers))) != self.supervision_layers:
            raise ValueError("supervision_layers must be unique and increasing")
        if self.supervision_layers[-1] >= self.depth:
            raise ValueError("supervision layers must index the online encoder")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.layer_norm_epsilon <= 0 or self.initializer_range <= 0:
            raise ValueError(
                "normalization epsilon and initializer range must be positive"
            )
        return self


__all__ = ["VJEPA2_1Config"]
