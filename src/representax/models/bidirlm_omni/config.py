"""Validated native configuration for BidirLM Omni representation models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.qwen3_vl.config import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

BIDIRLM_OMNI_2_5B_MODEL_ID = "BidirLM/BidirLM-Omni-2.5B-Embedding"
BIDIRLM_OMNI_2_5B_REVISION = "447a6e31be61b84443144afda21374339ce408e6"


class BidirLMOmniAudioConfig(FrozenConfig):
    """Three-stage convolutional audio frontend and bidirectional transformer."""

    num_mel_bins: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_size: int
    downsample_hidden_size: int
    output_size: int
    max_source_positions: int
    window_size: int
    inference_window_size: int
    convolution_chunk_size: int
    initializer_range: float = 0.02
    norm_epsilon: float = 1e-5

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def frequency_bins_after_convolution(self) -> int:
        value = self.num_mel_bins
        for _ in range(3):
            value = (value - 1) // 2 + 1
        return value

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "num_mel_bins",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_size",
            "downsample_hidden_size",
            "output_size",
            "max_source_positions",
            "window_size",
            "inference_window_size",
            "convolution_chunk_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("audio hidden_size must divide attention heads")
        if self.inference_window_size % (2 * self.window_size):
            raise ValueError(
                "inference_window_size must divide the post-convolution chunk policy"
            )
        for name in ("initializer_range", "norm_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> BidirLMOmniAudioConfig:
        audio = value.get("audio_config")
        if not isinstance(audio, Mapping):
            raise ValueError("BidirLM Omni config must contain audio_config")
        return cls(
            num_mel_bins=int(audio["num_mel_bins"]),
            num_hidden_layers=int(audio["encoder_layers"]),
            num_attention_heads=int(audio["encoder_attention_heads"]),
            intermediate_size=int(audio["encoder_ffn_dim"]),
            hidden_size=int(audio["d_model"]),
            downsample_hidden_size=int(audio["downsample_hidden_size"]),
            output_size=int(audio["output_dim"]),
            max_source_positions=int(audio["max_source_positions"]),
            window_size=int(audio["n_window"]),
            inference_window_size=int(audio["n_window_infer"]),
            convolution_chunk_size=int(audio["conv_chunksize"]),
            initializer_range=float(audio.get("initializer_range", 0.02)),
        )


class BidirLMOmniConfig(FrozenConfig):
    """Complete text, image, video, and audio encoder configuration."""

    text: Qwen3VLTextConfig
    vision: Qwen3VLVisionConfig
    audio: BidirLMOmniAudioConfig
    audio_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("vision output_size must equal text hidden_size")
        if self.audio.output_size != self.text.hidden_size:
            raise ValueError("audio output_size must equal text hidden_size")
        tokens = (
            self.audio_token_id,
            self.audio_start_token_id,
            self.audio_end_token_id,
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        )
        if any(not 0 <= token < self.text.vocab_size for token in tokens):
            raise ValueError("multimodal token identifiers must index the vocabulary")
        if len(set(tokens)) != len(tokens):
            raise ValueError("multimodal token identifiers must be distinct")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> BidirLMOmniConfig:
        if str(value.get("model_type", "")) != "bidirlm_omni":
            raise ValueError("expected model_type='bidirlm_omni'")
        text = value.get("text_config")
        vision = value.get("vision_config")
        if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
            raise ValueError("BidirLM Omni requires text_config and vision_config")
        rope = text.get("rope_scaling", value.get("rope_scaling", {}))
        if not isinstance(rope, Mapping):
            rope = {}
        raw_indexes = tuple(
            int(item) for item in vision.get("deepstack_visual_indexes", ())
        )
        # The 2.5B checkpoint declares index 24 for a zero-based 24-block tower.
        # Its reference forward never executes that tap, so the native graph omits it.
        active_indexes = tuple(
            index for index in raw_indexes if index < int(vision["depth"])
        )
        qwen_shape = {
            "text_config": {
                **dict(text),
                "max_position_embeddings": int(
                    text.get("max_position_embeddings", 128_000)
                ),
                "pad_token_id": int(
                    text.get("pad_token_id", value.get("pad_token_id", 0))
                ),
                "rope_parameters": {
                    "rope_theta": float(
                        rope.get("rope_theta", text.get("rope_theta", 5_000_000.0))
                    ),
                    "mrope_section": list(rope.get("mrope_section", (24, 20, 20))),
                },
            },
            "vision_config": {
                **dict(vision),
                "deepstack_visual_indexes": list(active_indexes),
            },
        }
        return cls(
            text=Qwen3VLTextConfig.from_hf_config(qwen_shape),
            vision=Qwen3VLVisionConfig.from_hf_config(qwen_shape),
            audio=BidirLMOmniAudioConfig.from_hf_config(value),
            audio_token_id=int(value.get("audio_token_id", 151676)),
            audio_start_token_id=int(value.get("audio_start_token_id", 151669)),
            audio_end_token_id=int(value.get("audio_end_token_id", 151670)),
            image_token_id=int(value.get("image_token_id", 151655)),
            video_token_id=int(value.get("video_token_id", 151656)),
            vision_start_token_id=int(value.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(value.get("vision_end_token_id", 151653)),
        )


__all__ = [
    "BIDIRLM_OMNI_2_5B_MODEL_ID",
    "BIDIRLM_OMNI_2_5B_REVISION",
    "BidirLMOmniAudioConfig",
    "BidirLMOmniConfig",
]
