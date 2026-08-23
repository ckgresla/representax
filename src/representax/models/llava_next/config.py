"""Validated native configuration for LLaVA-NeXT retrieval checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.clip import CLIPVisionConfig
from representax.models.decoder import RotaryDecoderConfig

BGE_VL_MLLM_S1_MODEL_ID = "BAAI/BGE-VL-MLLM-S1"
BGE_VL_MLLM_S1_REVISION = "455ac20c111813fbb263dd0f22d47d173a971582"
BGE_VL_MLLM_S2_MODEL_ID = "BAAI/BGE-VL-MLLM-S2"
BGE_VL_MLLM_S2_REVISION = "20137e245f277e7eca277bbb436ce7d632a16406"
BGE_VL_V15_ZS_MODEL_ID = "BAAI/BGE-VL-v1.5-zs"
BGE_VL_V15_ZS_REVISION = "a7ca46102a1a8be517e85cc1f03d1df39498e56c"
BGE_VL_V15_MMEB_MODEL_ID = "BAAI/BGE-VL-v1.5-mmeb"
BGE_VL_V15_MMEB_REVISION = "59f60b95765b32014df235059c4d8c60e8204be5"
E5_V_MODEL_ID = "royokong/e5-v"
E5_V_REVISION = "684c4c91ebabce3806d4fd8ac52c9c543043f962"

VisionFeatureStrategy = Literal["default", "full"]


class LlavaNextConfig(FrozenConfig):
    """CLIP vision, multimodal projection, and Llama/Mistral text tower."""

    text: RotaryDecoderConfig
    vision: CLIPVisionConfig
    image_grid_pinpoints: tuple[tuple[int, int], ...]
    image_token_id: int
    pad_token_id: int
    vision_feature_layer: int
    vision_feature_select_strategy: VisionFeatureStrategy
    projector_hidden_activation: Literal["gelu"] = "gelu"
    projector_bias: bool = True
    use_image_newline: bool = True
    initializer_range: float = 0.02

    @property
    def selected_vision_layer_count(self) -> int:
        if self.vision_feature_layer < 0:
            return self.vision.num_hidden_layers + 1 + self.vision_feature_layer
        return self.vision_feature_layer

    @property
    def selected_vision_tokens(self) -> int:
        additional = 0 if self.vision_feature_select_strategy == "default" else 1
        return self.vision.patch_count + additional

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if not self.image_grid_pinpoints:
            raise ValueError("image_grid_pinpoints must be non-empty")
        if any(
            height <= 0 or width <= 0 for height, width in self.image_grid_pinpoints
        ):
            raise ValueError("image grid resolutions must be positive")
        if not 0 <= self.image_token_id < self.text.vocab_size:
            raise ValueError("image_token_id must index the vocabulary")
        if not 0 <= self.pad_token_id < self.text.vocab_size:
            raise ValueError("pad_token_id must index the vocabulary")
        if not 0 <= self.selected_vision_layer_count <= self.vision.num_hidden_layers:
            raise ValueError("vision_feature_layer is outside the vision stack")
        if not math.isfinite(self.initializer_range) or self.initializer_range <= 0:
            raise ValueError("initializer_range must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> LlavaNextConfig:
        if str(value.get("model_type", "")) != "llava_next":
            raise ValueError("expected model_type='llava_next'")
        text = value.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("LLaVA-NeXT config must contain text_config")
        points = value.get("image_grid_pinpoints")
        if not isinstance(points, (list, tuple)):
            raise ValueError("image_grid_pinpoints must be an array")
        strategy = str(value.get("vision_feature_select_strategy", "default"))
        if strategy not in {"default", "full"}:
            raise ValueError(f"unsupported vision feature strategy {strategy!r}")
        activation = str(value.get("projector_hidden_act", "gelu"))
        if activation != "gelu":
            raise ValueError(f"unsupported projector activation {activation!r}")
        pad_token_id = value.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = text.get("pad_token_id")
        if pad_token_id is None:
            pad_token_id = 0
        return cls(
            text=RotaryDecoderConfig.from_hf_config(text),
            vision=CLIPVisionConfig.from_hf_config(value),
            image_grid_pinpoints=tuple(
                (int(point[0]), int(point[1])) for point in points
            ),
            image_token_id=int(value["image_token_index"]),
            pad_token_id=int(pad_token_id),
            vision_feature_layer=int(value.get("vision_feature_layer", -2)),
            vision_feature_select_strategy=strategy,
            projector_hidden_activation=activation,
            projector_bias=bool(value.get("multimodal_projector_bias", True)),
            use_image_newline=bool(value.get("use_image_newline_parameter", True)),
            initializer_range=float(value.get("initializer_range", 0.02)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        return {
            "architectures": ["LlavaNextModel"],
            "model_type": "llava_next",
            "image_grid_pinpoints": [
                list(point) for point in self.image_grid_pinpoints
            ],
            "image_token_index": self.image_token_id,
            "pad_token_id": self.pad_token_id,
            "vision_feature_layer": self.vision_feature_layer,
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "projector_hidden_act": self.projector_hidden_activation,
            "multimodal_projector_bias": self.projector_bias,
            "use_image_newline_parameter": self.use_image_newline,
            "initializer_range": self.initializer_range,
            "text_config": self.text.to_hf_config(),
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
            },
        }


__all__ = [
    "BGE_VL_MLLM_S1_MODEL_ID",
    "BGE_VL_MLLM_S1_REVISION",
    "BGE_VL_MLLM_S2_MODEL_ID",
    "BGE_VL_MLLM_S2_REVISION",
    "BGE_VL_V15_MMEB_MODEL_ID",
    "BGE_VL_V15_MMEB_REVISION",
    "BGE_VL_V15_ZS_MODEL_ID",
    "BGE_VL_V15_ZS_REVISION",
    "E5_V_MODEL_ID",
    "E5_V_REVISION",
    "LlavaNextConfig",
]
