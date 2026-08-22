"""Validated native configuration for CLIP and BGE-VL CLIP checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig

BGE_VL_BASE_MODEL_ID = "BAAI/BGE-VL-base"
BGE_VL_BASE_REVISION = "cc4c733ed997dbee4ac70ccffb911e70c9c24b93"
CLIP_VIT_B32_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
CLIP_VIT_B32_REVISION = "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
CLIPActivation = Literal["quick_gelu", "gelu", "gelu_new"]


def _activation(value: object) -> CLIPActivation:
    name = str(value)
    if name not in ("quick_gelu", "gelu", "gelu_new"):
        raise ValueError(f"unsupported CLIP activation {name!r}")
    return name


class CLIPTextConfig(FrozenConfig):
    """CLIP's causal text-transformer configuration."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    hidden_activation: CLIPActivation
    layer_norm_epsilon: float
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    initializer_range: float = 0.02

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("text hidden_size must divide attention heads")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if not 0 <= getattr(self, name) < self.vocab_size:
                raise ValueError(f"{name} must index the text vocabulary")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        for name in ("layer_norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> CLIPTextConfig:
        text = value.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("CLIP config must contain text_config")
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            hidden_activation=_activation(text.get("hidden_act", "quick_gelu")),
            layer_norm_epsilon=float(text.get("layer_norm_eps", 1e-5)),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            bos_token_id=int(text.get("bos_token_id", 0)),
            eos_token_id=int(text.get("eos_token_id", 2)),
            pad_token_id=int(text.get("pad_token_id", 1)),
            initializer_range=float(text.get("initializer_range", 0.02)),
        )


class CLIPVisionConfig(FrozenConfig):
    """Fixed-resolution CLIP vision-transformer configuration."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    image_size: int
    patch_size: int
    num_channels: int = 3
    hidden_activation: CLIPActivation
    layer_norm_epsilon: float
    attention_dropout: float
    initializer_range: float = 0.02

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_count(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "image_size",
            "patch_size",
            "num_channels",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("vision hidden_size must divide attention heads")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must divide patch_size")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")
        for name in ("layer_norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> CLIPVisionConfig:
        vision = value.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("CLIP config must contain vision_config")
        return cls(
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_hidden_layers=int(vision["num_hidden_layers"]),
            num_attention_heads=int(vision["num_attention_heads"]),
            image_size=int(vision["image_size"]),
            patch_size=int(vision["patch_size"]),
            num_channels=int(vision.get("num_channels", 3)),
            hidden_activation=_activation(vision.get("hidden_act", "quick_gelu")),
            layer_norm_epsilon=float(vision.get("layer_norm_eps", 1e-5)),
            attention_dropout=float(vision.get("attention_dropout", 0.0)),
            initializer_range=float(vision.get("initializer_range", 0.02)),
        )


class CLIPConfig(FrozenConfig):
    """Complete dual-encoder architecture and shared projection dimension."""

    text: CLIPTextConfig
    vision: CLIPVisionConfig
    projection_dimension: int
    logit_scale_initial_value: float = 2.6592

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.projection_dimension <= 0:
            raise ValueError("projection_dimension must be positive")
        if not math.isfinite(self.logit_scale_initial_value):
            raise ValueError("logit_scale_initial_value must be finite")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> CLIPConfig:
        if str(value.get("model_type", "")) != "clip":
            raise ValueError("expected model_type='clip'")
        return cls(
            text=CLIPTextConfig.from_hf_config(value),
            vision=CLIPVisionConfig.from_hf_config(value),
            projection_dimension=int(value["projection_dim"]),
            logit_scale_initial_value=float(
                value.get("logit_scale_init_value", 2.6592)
            ),
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Serialize the native architecture as a Transformers CLIP config."""

        return {
            "architectures": ["CLIPModel"],
            "model_type": "clip",
            "projection_dim": self.projection_dimension,
            "logit_scale_init_value": self.logit_scale_initial_value,
            "initializer_factor": 1.0,
            "text_config": {
                "model_type": "clip_text_model",
                "vocab_size": self.text.vocab_size,
                "hidden_size": self.text.hidden_size,
                "intermediate_size": self.text.intermediate_size,
                "num_hidden_layers": self.text.num_hidden_layers,
                "num_attention_heads": self.text.num_attention_heads,
                "max_position_embeddings": self.text.max_position_embeddings,
                "hidden_act": self.text.hidden_activation,
                "layer_norm_eps": self.text.layer_norm_epsilon,
                "attention_dropout": self.text.attention_dropout,
                "bos_token_id": self.text.bos_token_id,
                "eos_token_id": self.text.eos_token_id,
                "pad_token_id": self.text.pad_token_id,
                "initializer_range": self.text.initializer_range,
                "projection_dim": self.projection_dimension,
            },
            "vision_config": {
                "model_type": "clip_vision_model",
                "hidden_size": self.vision.hidden_size,
                "intermediate_size": self.vision.intermediate_size,
                "num_hidden_layers": self.vision.num_hidden_layers,
                "num_attention_heads": self.vision.num_attention_heads,
                "image_size": self.vision.image_size,
                "patch_size": self.vision.patch_size,
                "num_channels": self.vision.num_channels,
                "hidden_act": self.vision.hidden_activation,
                "layer_norm_eps": self.vision.layer_norm_epsilon,
                "attention_dropout": self.vision.attention_dropout,
                "initializer_range": self.vision.initializer_range,
                "projection_dim": self.projection_dimension,
            },
        }


__all__ = [
    "BGE_VL_BASE_MODEL_ID",
    "BGE_VL_BASE_REVISION",
    "CLIP_VIT_B32_MODEL_ID",
    "CLIP_VIT_B32_REVISION",
    "CLIPConfig",
    "CLIPTextConfig",
    "CLIPVisionConfig",
]
