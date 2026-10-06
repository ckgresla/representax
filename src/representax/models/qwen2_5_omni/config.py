"""Validated native configuration for Qwen2.5-Omni representation models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from representax._config import FrozenConfig

LCO_OMNI_3B_2605_MODEL_ID = "LCO-Embedding/LCO-Embedding-Omni-3B-2605"
LCO_OMNI_3B_2605_REVISION = "5f6b5329da5141367da30e06a9826d1322d6c9b2"
NVIDIA_OMNI_EMBED_3B_MODEL_ID = "nvidia/omni-embed-nemotron-3b"
NVIDIA_OMNI_EMBED_3B_REVISION = "865db1bb57e369a85357cf114cbd6b3c5322d19d"


class Qwen2_5OmniTextConfig(FrozenConfig):
    """Causal GQA language tower receiving text, image, audio, and video tokens."""

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
    layer_types: tuple[str, ...]
    sliding_window: int | None = None
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
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must name every text layer")
        if any(
            layer_type not in {"full_attention", "sliding_attention"}
            for layer_type in self.layer_types
        ):
            raise ValueError("unsupported Qwen2.5-Omni text attention layer type")
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError("sliding layers require a sliding_window")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive or None")
        for name in ("rope_theta", "norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen2_5OmniTextConfig:
        text = value.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("Qwen2.5-Omni config must contain text_config")
        rope = text.get("rope_parameters", text.get("rope_scaling", {}))
        if not isinstance(rope, Mapping):
            raise ValueError("text_config rope configuration must be an object")
        section = rope.get("mrope_section")
        if not isinstance(section, (list, tuple)) or len(section) != 3:
            raise ValueError("mrope_section must contain temporal, height, and width")
        layer_count = int(text["num_hidden_layers"])
        layer_types = text.get("layer_types", ("full_attention",) * layer_count)
        if not isinstance(layer_types, (list, tuple)):
            raise ValueError("layer_types must be an array")
        heads = int(text["num_attention_heads"])
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=layer_count,
            num_attention_heads=heads,
            num_key_value_heads=int(text.get("num_key_value_heads", heads)),
            head_dimension=int(text.get("head_dim", int(text["hidden_size"]) // heads)),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rope_theta=float(
                rope.get("rope_theta", text.get("rope_theta", 1_000_000.0))
            ),
            mrope_section=tuple(int(item) for item in section),
            norm_epsilon=float(text.get("rms_norm_eps", 1e-6)),
            layer_types=tuple(str(item) for item in layer_types),
            sliding_window=(
                None
                if text.get("sliding_window") is None
                else int(text["sliding_window"])
            ),
            initializer_range=float(text.get("initializer_range", 0.02)),
        )


class Qwen2_5OmniVisionConfig(FrozenConfig):
    """Windowed patch-transformer tower shared by image and video inputs."""

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
        for name in ("initializer_range", "norm_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen2_5OmniVisionConfig:
        vision = value.get("vision_config")
        if not isinstance(vision, Mapping):
            raise ValueError("Qwen2.5-Omni config must contain vision_config")
        return cls(
            depth=int(vision["depth"]),
            hidden_size=int(vision["hidden_size"]),
            intermediate_size=int(vision["intermediate_size"]),
            num_attention_heads=int(vision["num_heads"]),
            in_channels=int(vision.get("in_channels", vision.get("in_chans", 3))),
            patch_size=int(vision["patch_size"]),
            temporal_patch_size=int(vision["temporal_patch_size"]),
            spatial_merge_size=int(vision["spatial_merge_size"]),
            output_size=int(vision["out_hidden_size"]),
            window_size=int(vision["window_size"]),
            full_attention_layers=tuple(
                int(item) for item in vision.get("fullatt_block_indexes", ())
            ),
            initializer_range=float(vision.get("initializer_range", 0.02)),
        )


class Qwen2_5OmniAudioConfig(FrozenConfig):
    """Window-packed non-causal audio transformer configuration."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_mel_bins: int
    max_source_positions: int
    window_size: int
    output_size: int
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_mel_bins",
            "max_source_positions",
            "window_size",
            "output_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("audio hidden_size must divide attention heads")
        for name in ("initializer_range", "layer_norm_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen2_5OmniAudioConfig:
        audio = value.get("audio_config")
        if not isinstance(audio, Mapping):
            raise ValueError("Qwen2.5-Omni config must contain audio_config")
        return cls(
            hidden_size=int(audio["d_model"]),
            intermediate_size=int(audio["encoder_ffn_dim"]),
            num_hidden_layers=int(audio["encoder_layers"]),
            num_attention_heads=int(audio["encoder_attention_heads"]),
            num_mel_bins=int(audio["num_mel_bins"]),
            max_source_positions=int(audio["max_source_positions"]),
            window_size=int(audio["n_window"]),
            output_size=int(audio["output_dim"]),
            initializer_range=float(audio.get("initializer_range", 0.02)),
        )


class Qwen2_5OmniConfig(FrozenConfig):
    """Complete native thinker configuration; the generative talker is excluded."""

    text: Qwen2_5OmniTextConfig
    vision: Qwen2_5OmniVisionConfig
    audio: Qwen2_5OmniAudioConfig
    pad_token_id: int
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    position_ids_per_second: int
    seconds_per_chunk: int

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("vision output_size must equal text hidden_size")
        if self.audio.output_size != self.text.hidden_size:
            raise ValueError("audio output_size must equal text hidden_size")
        for name in (
            "pad_token_id",
            "image_token_id",
            "video_token_id",
            "audio_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
            "audio_start_token_id",
            "audio_end_token_id",
        ):
            token = getattr(self, name)
            if not 0 <= token < self.text.vocab_size:
                raise ValueError(f"{name} must index the vocabulary")
        if self.position_ids_per_second <= 0 or self.seconds_per_chunk <= 0:
            raise ValueError("temporal position scales must be positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> Qwen2_5OmniConfig:
        model_type = str(value.get("model_type", ""))
        if model_type not in {"qwen2_5_omni_thinker", "nvomniembed"}:
            raise ValueError(
                "expected model_type='qwen2_5_omni_thinker' or 'nvomniembed'"
            )
        return cls(
            text=Qwen2_5OmniTextConfig.from_hf_config(value),
            vision=Qwen2_5OmniVisionConfig.from_hf_config(value),
            audio=Qwen2_5OmniAudioConfig.from_hf_config(value),
            pad_token_id=int(value.get("pad_token_id", 151643)),
            image_token_id=int(value.get("image_token_index", 151655)),
            video_token_id=int(value.get("video_token_index", 151656)),
            audio_token_id=int(value.get("audio_token_index", 151646)),
            vision_start_token_id=int(value.get("vision_start_token_id", 151652)),
            vision_end_token_id=int(value.get("vision_end_token_id", 151653)),
            audio_start_token_id=int(value.get("audio_start_token_id", 151647)),
            audio_end_token_id=int(value.get("audio_end_token_id", 151648)),
            position_ids_per_second=int(value.get("position_id_per_seconds", 25)),
            seconds_per_chunk=int(value.get("seconds_per_chunk", 2)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Serialize the native architecture as a Transformers thinker config."""

        return {
            "architectures": ["Qwen2_5OmniThinkerForConditionalGeneration"],
            "model_type": "qwen2_5_omni_thinker",
            "dtype": "float32",
            "tie_word_embeddings": False,
            "pad_token_id": self.pad_token_id,
            "image_token_index": self.image_token_id,
            "video_token_index": self.video_token_id,
            "audio_token_index": self.audio_token_id,
            "vision_start_token_id": self.vision_start_token_id,
            "vision_end_token_id": self.vision_end_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "position_id_per_seconds": self.position_ids_per_second,
            "seconds_per_chunk": self.seconds_per_chunk,
            "text_config": {
                "model_type": "qwen2_5_omni_text",
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
                "layer_types": list(self.text.layer_types),
                "sliding_window": self.text.sliding_window,
                "initializer_range": self.text.initializer_range,
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "pad_token_id": self.pad_token_id,
                "tie_word_embeddings": True,
                "use_cache": True,
                "use_sliding_window": "sliding_attention" in self.text.layer_types,
            },
            "vision_config": {
                "model_type": "qwen2_5_omni_vision_encoder",
                "depth": self.vision.depth,
                "hidden_size": self.vision.hidden_size,
                "intermediate_size": self.vision.intermediate_size,
                "num_heads": self.vision.num_attention_heads,
                "in_channels": self.vision.in_channels,
                "patch_size": self.vision.patch_size,
                "temporal_patch_size": self.vision.temporal_patch_size,
                "spatial_merge_size": self.vision.spatial_merge_size,
                "out_hidden_size": self.vision.output_size,
                "window_size": self.vision.window_size,
                "fullatt_block_indexes": list(self.vision.full_attention_layers),
                "initializer_range": self.vision.initializer_range,
                "hidden_act": "silu",
            },
            "audio_config": {
                "model_type": "qwen2_5_omni_audio_encoder",
                "d_model": self.audio.hidden_size,
                "encoder_ffn_dim": self.audio.intermediate_size,
                "encoder_layers": self.audio.num_hidden_layers,
                "encoder_attention_heads": self.audio.num_attention_heads,
                "num_mel_bins": self.audio.num_mel_bins,
                "max_source_positions": self.audio.max_source_positions,
                "n_window": self.audio.window_size,
                "output_dim": self.audio.output_size,
                "initializer_range": self.audio.initializer_range,
                "activation_function": "gelu",
                "activation_dropout": 0.0,
                "attention_dropout": 0.0,
                "dropout": 0.0,
                "scale_embedding": False,
            },
        }


__all__ = [
    "LCO_OMNI_3B_2605_MODEL_ID",
    "LCO_OMNI_3B_2605_REVISION",
    "Qwen2_5OmniAudioConfig",
    "Qwen2_5OmniConfig",
    "Qwen2_5OmniTextConfig",
    "Qwen2_5OmniVisionConfig",
]
