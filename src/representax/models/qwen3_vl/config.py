"""Validated native configuration for Qwen3-VL representation models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from representax._config import FrozenConfig

QWEN3_VL_EMBEDDING_2B_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
QWEN3_VL_EMBEDDING_2B_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
QWEN3_VL_RERANKER_2B_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
QWEN3_VL_RERANKER_2B_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
EAGER_EMBED_V1_MODEL_ID = "eagerworks/eager-embed-v1"
EAGER_EMBED_V1_REVISION = "51dfdee0d1d1067afe00d816dca2cd72a02f6bec"


class Qwen3VLTextConfig(FrozenConfig):
    """Causal GQA language tower used by every Qwen3-VL checkpoint size."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    max_position_embeddings: int
    rope_theta: float
    mrope_section: tuple[int, int, int]
    norm_epsilon: float
    pad_token_id: int
    initializer_range: float = 0.02

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dimension",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dimension % 2:
            raise ValueError("head_dimension must be even")
        if sum(self.mrope_section) * 2 != self.head_dimension:
            raise ValueError("mrope_section must partition half the head dimension")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must index the vocabulary")
        for name in ("rope_theta", "norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen3VLTextConfig:
        text = value.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("Qwen3-VL config must contain text_config")
        rope = text.get("rope_parameters", text.get("rope_scaling", {}))
        if not isinstance(rope, Mapping):
            raise ValueError("text_config rope configuration must be an object")
        section = rope.get("mrope_section", (24, 20, 20))
        if not isinstance(section, (list, tuple)) or len(section) != 3:
            raise ValueError("mrope_section must contain temporal, height, and width")
        pad_token_id = text.get("pad_token_id", value.get("pad_token_id", 151643))
        if pad_token_id is None:
            pad_token_id = 151643
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(
                text.get("num_key_value_heads", text["num_attention_heads"])
            ),
            head_dimension=int(
                text.get(
                    "head_dim",
                    int(text["hidden_size"]) // int(text["num_attention_heads"]),
                )
            ),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 500_000.0))),
            mrope_section=tuple(int(item) for item in section),
            norm_epsilon=float(text.get("rms_norm_eps", 1e-6)),
            pad_token_id=int(pad_token_id),
            initializer_range=float(text.get("initializer_range", 0.02)),
        )


class Qwen3VLVisionConfig(FrozenConfig):
    """Patch-transformer vision tower shared by image and video inputs."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    in_channels: int
    patch_size: int
    spatial_merge_size: int
    temporal_patch_size: int
    output_size: int
    num_position_embeddings: int
    deepstack_visual_indexes: tuple[int, ...]
    initializer_range: float = 0.02
    norm_epsilon: float = 1e-6

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dimension(self) -> int:
        return (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )

    @property
    def spatial_merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
            "temporal_patch_size",
            "output_size",
            "num_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("vision hidden_size must divide attention heads")
        side = math.isqrt(self.num_position_embeddings)
        if side * side != self.num_position_embeddings:
            raise ValueError("num_position_embeddings must be a square grid")
        if tuple(sorted(set(self.deepstack_visual_indexes))) != (
            self.deepstack_visual_indexes
        ):
            raise ValueError("deepstack_visual_indexes must be unique and sorted")
        if any(not 0 <= index < self.depth for index in self.deepstack_visual_indexes):
            raise ValueError("deepstack indexes must name vision layers")
        for name in ("initializer_range", "norm_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen3VLVisionConfig:
        vision = value.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("Qwen3-VL config must contain vision_config")
        return cls(
            depth=int(vision["depth"]),
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_attention_heads=int(vision["num_heads"]),
            in_channels=int(vision.get("in_channels", 3)),
            patch_size=int(vision["patch_size"]),
            spatial_merge_size=int(vision["spatial_merge_size"]),
            temporal_patch_size=int(vision["temporal_patch_size"]),
            output_size=int(vision["out_hidden_size"]),
            num_position_embeddings=int(vision["num_position_embeddings"]),
            deepstack_visual_indexes=tuple(
                int(item) for item in vision.get("deepstack_visual_indexes", ())
            ),
            initializer_range=float(vision.get("initializer_range", 0.02)),
        )


class Qwen3VLConfig(FrozenConfig):
    """Complete multimodal model configuration."""

    text: Qwen3VLTextConfig
    vision: Qwen3VLVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("vision output_size must equal text hidden_size")
        for name in (
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            token = getattr(self, name)
            if not 0 <= token < self.text.vocab_size:
                raise ValueError(f"{name} must index the vocabulary")
        if (
            len(
                {
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    self.vision_end_token_id,
                }
            )
            != 4
        ):
            raise ValueError("multimodal token identifiers must be distinct")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen3VLConfig:
        if str(value.get("model_type", "")) != "qwen3_vl":
            raise ValueError("expected model_type='qwen3_vl'")
        return cls(
            text=Qwen3VLTextConfig.from_hf_config(value),
            vision=Qwen3VLVisionConfig.from_hf_config(value),
            image_token_id=int(value.get("image_token_id", 151655)),
            video_token_id=int(value.get("video_token_id", 151656)),
            vision_start_token_id=int(value.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(value.get("vision_end_token_id", 151653)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Serialize the native architecture for a Transformers reload."""

        return {
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "dtype": "float32",
            "tie_word_embeddings": True,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "pad_token_id": self.text.pad_token_id,
            "text_config": {
                "model_type": "qwen3_vl_text",
                "vocab_size": self.text.vocab_size,
                "hidden_size": self.text.hidden_size,
                "intermediate_size": self.text.intermediate_size,
                "num_hidden_layers": self.text.num_hidden_layers,
                "num_attention_heads": self.text.num_attention_heads,
                "num_key_value_heads": self.text.num_key_value_heads,
                "head_dim": self.text.head_dimension,
                "max_position_embeddings": self.text.max_position_embeddings,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": self.text.rope_theta,
                    "mrope_section": list(self.text.mrope_section),
                },
                "rms_norm_eps": self.text.norm_epsilon,
                "pad_token_id": self.text.pad_token_id,
                "initializer_range": self.text.initializer_range,
                "hidden_act": "silu",
                "attention_bias": False,
                "attention_dropout": 0.0,
                "tie_word_embeddings": True,
                "use_cache": True,
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": self.vision.depth,
                "hidden_size": self.vision.hidden_size,
                "intermediate_size": self.vision.intermediate_size,
                "num_heads": self.vision.num_attention_heads,
                "in_channels": self.vision.in_channels,
                "patch_size": self.vision.patch_size,
                "spatial_merge_size": self.vision.spatial_merge_size,
                "temporal_patch_size": self.vision.temporal_patch_size,
                "out_hidden_size": self.vision.output_size,
                "num_position_embeddings": self.vision.num_position_embeddings,
                "deepstack_visual_indexes": list(self.vision.deepstack_visual_indexes),
                "initializer_range": self.vision.initializer_range,
                "hidden_act": "gelu_pytorch_tanh",
            },
        }


__all__ = [
    "EAGER_EMBED_V1_MODEL_ID",
    "EAGER_EMBED_V1_REVISION",
    "QWEN3_VL_EMBEDDING_2B_MODEL_ID",
    "QWEN3_VL_EMBEDDING_2B_REVISION",
    "QWEN3_VL_RERANKER_2B_MODEL_ID",
    "QWEN3_VL_RERANKER_2B_REVISION",
    "Qwen3VLConfig",
    "Qwen3VLTextConfig",
    "Qwen3VLVisionConfig",
]
