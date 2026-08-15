"""Native configuration for the ModernVBERT text tower."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig

AttentionType = Literal["full_attention", "sliding_attention"]

MODERNVBERT_MODEL_ID = "ModernVBERT/modernvbert-embed"
MODERNVBERT_REVISION = "da507113c3fdbc2e49d39c4b0148025c6bd008f9"


class ModernVBERTTextConfig(FrozenConfig):
    """Architecture values required by the native text implementation."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    layer_types: tuple[AttentionType, ...]
    local_attention: int
    full_attention_rope_theta: float
    sliding_attention_rope_theta: float
    norm_epsilon: float
    max_position_embeddings: int
    initializer_range: float = 0.02

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per layer")
        supported = {"full_attention", "sliding_attention"}
        if any(layer_type not in supported for layer_type in self.layer_types):
            raise ValueError("unsupported ModernVBERT attention type")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dimension % 2:
            raise ValueError("rotary attention requires an even head dimension")
        if self.local_attention <= 0:
            raise ValueError("local_attention must be positive")
        for name, value in (
            ("full_attention_rope_theta", self.full_attention_rope_theta),
            ("sliding_attention_rope_theta", self.sliding_attention_rope_theta),
            ("norm_epsilon", self.norm_epsilon),
            ("initializer_range", self.initializer_range),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def rope_theta(self, layer_type: AttentionType) -> float:
        if layer_type == "full_attention":
            return self.full_attention_rope_theta
        if layer_type == "sliding_attention":
            return self.sliding_attention_rope_theta
        raise ValueError(f"unsupported attention type: {layer_type!r}")

    @classmethod
    def from_hf_config(
        cls,
        value: Mapping[str, Any],
    ) -> ModernVBERTTextConfig:
        """Map a Transformers ModernVBERT config into native values."""

        text = value.get("text_config", value)
        if not isinstance(text, Mapping):
            raise TypeError("text_config must be a mapping")
        rope = text.get("rope_parameters", {})
        if not isinstance(rope, Mapping):
            raise TypeError("rope_parameters must be a mapping")

        def theta(kind: AttentionType, legacy_name: str) -> float:
            section = rope.get(kind, {})
            if isinstance(section, Mapping) and "rope_theta" in section:
                return float(section["rope_theta"])
            if legacy_name in text:
                return float(text[legacy_name])
            raise KeyError(f"missing RoPE theta for {kind}")

        def attention_type(item: Any) -> AttentionType:
            resolved = str(item)
            if resolved not in ("full_attention", "sliding_attention"):
                raise ValueError(
                    f"unsupported ModernVBERT attention type: {resolved!r}"
                )
            return resolved

        layer_types = tuple(attention_type(item) for item in text["layer_types"])
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            layer_types=layer_types,
            local_attention=int(text["local_attention"]),
            full_attention_rope_theta=theta("full_attention", "global_rope_theta"),
            sliding_attention_rope_theta=theta("sliding_attention", "local_rope_theta"),
            norm_epsilon=float(text.get("norm_eps", text.get("layer_norm_eps", 1e-5))),
            max_position_embeddings=int(text["max_position_embeddings"]),
            initializer_range=float(text.get("initializer_range", 0.02)),
        )


class ModernVBERTVisionConfig(FrozenConfig):
    """Architecture values for ModernVBERT's SigLIP vision tower."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_channels: int
    image_size: int
    patch_size: int
    norm_epsilon: float
    hidden_activation: str = "gelu_pytorch_tanh"

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_channels": self.num_channels,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError(
                "vision hidden_size must be divisible by num_attention_heads"
            )
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0:
            raise ValueError("vision norm_epsilon must be finite and positive")
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError("only the checkpoint's tanh GELU is supported")
        return self

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_grid(self) -> int:
        return self.image_size // self.patch_size

    @property
    def patch_count(self) -> int:
        return self.patch_grid**2

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> ModernVBERTVisionConfig:
        vision = value.get("vision_config", value)
        if not isinstance(vision, Mapping):
            raise TypeError("vision_config must be a mapping")
        return cls(
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_hidden_layers=int(vision["num_hidden_layers"]),
            num_attention_heads=int(vision["num_attention_heads"]),
            num_channels=int(vision["num_channels"]),
            image_size=int(vision["image_size"]),
            patch_size=int(vision["patch_size"]),
            norm_epsilon=float(vision["layer_norm_eps"]),
            hidden_activation=str(vision.get("hidden_act", "gelu_pytorch_tanh")),
        )


class ModernVBERTConfig(FrozenConfig):
    """Complete native ModernVBERT architecture configuration."""

    text: ModernVBERTTextConfig
    vision: ModernVBERTVisionConfig
    image_token_id: int
    pixel_shuffle_factor: int

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.image_token_id < 0 or self.image_token_id >= self.text.vocab_size:
            raise ValueError("image_token_id must be inside the text vocabulary")
        if self.pixel_shuffle_factor <= 0:
            raise ValueError("pixel_shuffle_factor must be positive")
        if self.vision.patch_grid % self.pixel_shuffle_factor:
            raise ValueError("vision patch grid must divide pixel_shuffle_factor")
        connector_input = self.vision.hidden_size * self.pixel_shuffle_factor**2
        if connector_input <= 0 or self.text.hidden_size <= 0:
            raise ValueError("connector dimensions must be positive")
        return self

    @property
    def image_sequence_length(self) -> int:
        return self.vision.patch_count // self.pixel_shuffle_factor**2

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> ModernVBERTConfig:
        return cls(
            text=ModernVBERTTextConfig.from_hf_config(value),
            vision=ModernVBERTVisionConfig.from_hf_config(value),
            image_token_id=int(value["image_token_id"]),
            pixel_shuffle_factor=int(value["pixel_shuffle_factor"]),
        )
