"""Validated configuration shared by Qwen2-VL and Qwen2.5-VL."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.qwen2_5_omni.config import Qwen2_5OmniTextConfig

BGE_VL_SCREENSHOT_MODEL_ID = "BAAI/BGE-VL-Screenshot"
BGE_VL_SCREENSHOT_REVISION = "2b0f1cd3e4acf66be759d840954e0c9f1c9a42cf"
NOMIC_MULTIMODAL_3B_MODEL_ID = "nomic-ai/nomic-embed-multimodal-3b"
NOMIC_MULTIMODAL_3B_REVISION = "29259db79bc6ee5fcc9e6abc8a8e16d8491e5116"
NOMIC_MULTIMODAL_7B_MODEL_ID = "nomic-ai/nomic-embed-multimodal-7b"
NOMIC_MULTIMODAL_7B_REVISION = "234bc2738e2d5ae77beca8f94e1577a7a48fc609"
JINA_RERANKER_M0_MODEL_ID = "jinaai/jina-reranker-m0"
JINA_RERANKER_M0_REVISION = "94bfe0aeb2d4dd7978362699cddd5893d4e0adc8"

QWEN2_5_VL_3B_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN2_5_VL_3B_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
QWEN2_5_VL_7B_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN2_5_VL_7B_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
QWEN2_VL_2B_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
QWEN2_VL_2B_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"

Qwen2VLGeneration = Literal["qwen2_vl", "qwen2_5_vl"]


class Qwen2VLVisionConfig(FrozenConfig):
    """Generation-aware image/video patch-transformer configuration."""

    generation: Qwen2VLGeneration
    depth: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    output_size: int
    window_size: int
    full_attention_layers: tuple[int, ...]
    norm: Literal["layer", "rms"]
    mlp: Literal["gelu", "quick_gelu", "swiglu"]
    tokens_per_second: float = 2.0
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

    @property
    def merger_window_size(self) -> int:
        return self.window_size // self.spatial_merge_size // self.patch_size

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "depth",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "in_channels",
            "patch_size",
            "temporal_patch_size",
            "spatial_merge_size",
            "output_size",
            "window_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("vision hidden_size must divide attention heads")
        if self.window_size % (self.spatial_merge_size * self.patch_size):
            raise ValueError("window_size must divide merged patch geometry")
        if tuple(sorted(set(self.full_attention_layers))) != self.full_attention_layers:
            raise ValueError("full_attention_layers must be unique and sorted")
        if any(not 0 <= index < self.depth for index in self.full_attention_layers):
            raise ValueError("full_attention_layers must name vision layers")
        for name in ("initializer_range", "norm_epsilon", "tokens_per_second"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(
        cls,
        value: Mapping[str, Any],
        *,
        generation: Qwen2VLGeneration,
        text_hidden_size: int,
    ) -> Qwen2VLVisionConfig:
        raw = value.get("vision_config")
        if not isinstance(raw, Mapping):
            raise ValueError("Qwen2-VL config must contain vision_config")
        if generation == "qwen2_vl":
            hidden = int(raw.get("embed_dim", 1280))
            depth = int(raw.get("depth", 32))
            mlp = str(raw.get("hidden_act", "quick_gelu"))
            if mlp not in {"gelu", "quick_gelu", "swiglu"}:
                raise ValueError(f"unsupported Qwen2-VL vision activation {mlp!r}")
            return cls(
                generation=generation,
                depth=depth,
                hidden_size=hidden,
                intermediate_size=int(raw.get("intermediate_size", hidden * 4)),
                num_attention_heads=int(raw.get("num_heads", 16)),
                in_channels=int(raw.get("in_channels", raw.get("in_chans", 3))),
                patch_size=int(
                    raw.get("patch_size", raw.get("spatial_patch_size", 14))
                ),
                temporal_patch_size=int(raw.get("temporal_patch_size", 2)),
                spatial_merge_size=int(raw.get("spatial_merge_size", 2)),
                output_size=int(raw.get("hidden_size", text_hidden_size)),
                window_size=int(raw.get("window_size", 112)),
                full_attention_layers=tuple(range(depth)),
                norm="layer",
                mlp=mlp,
                tokens_per_second=float(raw.get("tokens_per_second", 25.0)),
                initializer_range=float(raw.get("initializer_range", 0.02)),
            )
        depth = int(raw.get("depth", 32))
        return cls(
            generation=generation,
            depth=depth,
            hidden_size=int(raw.get("hidden_size", 1280)),
            intermediate_size=int(raw.get("intermediate_size", 3420)),
            num_attention_heads=int(raw.get("num_heads", 16)),
            in_channels=int(raw.get("in_channels", raw.get("in_chans", 3))),
            patch_size=int(raw.get("patch_size", raw.get("spatial_patch_size", 14))),
            temporal_patch_size=int(raw.get("temporal_patch_size", 2)),
            spatial_merge_size=int(raw.get("spatial_merge_size", 2)),
            output_size=int(raw.get("out_hidden_size", text_hidden_size)),
            window_size=int(raw.get("window_size", 112)),
            full_attention_layers=tuple(
                int(item) for item in raw.get("fullatt_block_indexes", (7, 15, 23, 31))
            ),
            norm="rms",
            mlp="swiglu",
            tokens_per_second=float(raw.get("tokens_per_second", 2.0)),
            initializer_range=float(raw.get("initializer_range", 0.02)),
        )


class Qwen2VLConfig(FrozenConfig):
    """Complete native Qwen2/Qwen2.5 vision-language backbone configuration."""

    generation: Qwen2VLGeneration
    text: Qwen2_5OmniTextConfig
    vision: Qwen2VLVisionConfig
    pad_token_id: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("vision output_size must equal text hidden_size")
        for name in (
            "pad_token_id",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            token = getattr(self, name)
            if not 0 <= token < self.text.vocab_size:
                raise ValueError(f"{name} must index the vocabulary")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen2VLConfig:
        model_type = str(value.get("model_type", ""))
        if model_type not in {"qwen2_vl", "qwen2_5_vl"}:
            raise ValueError("expected model_type='qwen2_vl' or 'qwen2_5_vl'")
        generation: Qwen2VLGeneration = (
            "qwen2_vl" if model_type == "qwen2_vl" else "qwen2_5_vl"
        )
        raw_text = value.get("text_config", value)
        if not isinstance(raw_text, Mapping):
            raise ValueError("text_config must be an object")
        text_value = dict(raw_text)
        rope = text_value.get("rope_parameters", text_value.get("rope_scaling", {}))
        if not isinstance(rope, Mapping):
            rope = {}
        text_value["rope_parameters"] = {
            **rope,
            "rope_theta": float(
                rope.get("rope_theta", text_value.get("rope_theta", 1_000_000.0))
            ),
        }
        layer_count = int(text_value["num_hidden_layers"])
        text_value.setdefault("layer_types", ["full_attention"] * layer_count)
        text_value.setdefault(
            "head_dim",
            int(text_value["hidden_size"]) // int(text_value["num_attention_heads"]),
        )
        text = Qwen2_5OmniTextConfig.from_hf_config({"text_config": text_value})
        vision = Qwen2VLVisionConfig.from_hf_config(
            value,
            generation=generation,
            text_hidden_size=text.hidden_size,
        )
        pad = value.get("pad_token_id", raw_text.get("pad_token_id", 151643))
        if pad is None:
            pad = 151643
        return cls(
            generation=generation,
            text=text,
            vision=vision,
            pad_token_id=int(pad),
            image_token_id=int(value.get("image_token_id", 151655)),
            video_token_id=int(value.get("video_token_id", 151656)),
            vision_start_token_id=int(value.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(value.get("vision_end_token_id", 151653)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Serialize the architecture fields required for HF reload."""

        vision: dict[str, Any] = {
            "model_type": self.generation,
            "depth": self.vision.depth,
            "hidden_size": self.vision.hidden_size,
            "intermediate_size": self.vision.intermediate_size,
            "num_heads": self.vision.num_attention_heads,
            "in_channels": self.vision.in_channels,
            "patch_size": self.vision.patch_size,
            "temporal_patch_size": self.vision.temporal_patch_size,
            "spatial_merge_size": self.vision.spatial_merge_size,
            "window_size": self.vision.window_size,
            "tokens_per_second": int(self.vision.tokens_per_second),
        }
        if self.generation == "qwen2_vl":
            vision["embed_dim"] = self.vision.hidden_size
            vision["hidden_size"] = self.vision.output_size
            vision["mlp_ratio"] = (
                self.vision.intermediate_size // self.vision.hidden_size
            )
        else:
            vision["out_hidden_size"] = self.vision.output_size
            vision["fullatt_block_indexes"] = list(self.vision.full_attention_layers)
        text = self.text.model_dump(mode="json")
        text.update(
            {
                "rms_norm_eps": self.text.norm_epsilon,
                "head_dim": self.text.head_dimension,
                "num_key_value_heads": self.text.num_key_value_heads,
                "num_attention_heads": self.text.num_attention_heads,
                "num_hidden_layers": self.text.num_hidden_layers,
                "max_position_embeddings": self.text.max_position_embeddings,
                "rope_theta": self.text.rope_theta,
                "rope_scaling": {
                    "rope_type": "default",
                    "mrope_section": list(self.text.mrope_section),
                },
            }
        )
        return {
            **text,
            "model_type": self.generation,
            "vision_config": vision,
            "pad_token_id": self.pad_token_id,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
        }


__all__ = [
    "BGE_VL_SCREENSHOT_MODEL_ID",
    "BGE_VL_SCREENSHOT_REVISION",
    "JINA_RERANKER_M0_MODEL_ID",
    "JINA_RERANKER_M0_REVISION",
    "NOMIC_MULTIMODAL_3B_MODEL_ID",
    "NOMIC_MULTIMODAL_3B_REVISION",
    "NOMIC_MULTIMODAL_7B_MODEL_ID",
    "NOMIC_MULTIMODAL_7B_REVISION",
    "QWEN2_5_VL_3B_MODEL_ID",
    "QWEN2_5_VL_3B_REVISION",
    "QWEN2_5_VL_7B_MODEL_ID",
    "QWEN2_5_VL_7B_REVISION",
    "QWEN2_VL_2B_MODEL_ID",
    "QWEN2_VL_2B_REVISION",
    "Qwen2VLConfig",
    "Qwen2VLGeneration",
    "Qwen2VLVisionConfig",
]
