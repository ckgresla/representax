"""Validated native LeJEPA ViT and projector configuration."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator

from representax._config import FrozenConfig


class LeJEPAViTConfig(FrozenConfig):
    """timm-compatible dynamic-resolution Vision Transformer configuration."""

    image_size: int = 224
    local_image_size: int = 98
    patch_size: int = 16
    channels: int = 3
    hidden_size: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    drop_path_rate: float = 0.1
    projector_bottleneck: int = 512
    projector_hidden_size: int = 2048
    projection_dimension: int = 512

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.grid_size**2

    @property
    def intermediate_size(self) -> int:
        return int(self.hidden_size * self.mlp_ratio)

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        positive = (
            "image_size",
            "local_image_size",
            "patch_size",
            "channels",
            "hidden_size",
            "depth",
            "heads",
            "projector_bottleneck",
            "projector_hidden_size",
            "projection_dimension",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.local_image_size > self.image_size:
            raise ValueError("local_image_size cannot exceed image_size")
        if self.local_image_size < self.patch_size:
            raise ValueError("local_image_size must contain at least one patch")
        if self.hidden_size % self.heads:
            raise ValueError("hidden_size must be divisible by heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.layer_norm_epsilon <= 0 or self.initializer_range <= 0:
            raise ValueError(
                "normalization epsilon and initializer range must be positive"
            )
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if self.depth < 2:
            raise ValueError("canonical evaluation requires at least two ViT blocks")
        return self

    @classmethod
    def vit_base_patch16(cls) -> Self:
        """Return the exact timm ViT-B/16 architectural profile."""

        return cls()

    @classmethod
    def vit_large_patch16(cls) -> Self:
        """Return the 304M-class paper ViT-L/16 backbone profile."""

        return cls(
            hidden_size=1024,
            depth=24,
            heads=16,
            projection_dimension=512,
        )

    @classmethod
    def vit_small_patch16(cls) -> Self:
        """Return the bounded timm ViT-S/16 lifecycle-canary profile."""

        return cls(
            hidden_size=384,
            depth=12,
            heads=6,
            projection_dimension=512,
        )


__all__ = ["LeJEPAViTConfig"]
