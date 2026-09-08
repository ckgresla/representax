"""Native configuration for Jina Embeddings v5 text and omni models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.qwen2_5_omni.config import Qwen2_5OmniAudioConfig
from representax.models.qwen3_vl.config import Qwen3VLTextConfig, Qwen3VLVisionConfig

JINA_V5_NANO_MODEL_ID = "jinaai/jina-embeddings-v5-omni-nano-retrieval"
JINA_V5_NANO_REVISION = "b7287f6b6b562e25bc4a28b939d1f936484b4137"
JINA_V5_SMALL_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
JINA_V5_SMALL_REVISION = "e3ae4b6e4af4ec0799cd931aefaff03235b5f9d4"


class JinaV5TextConfig(FrozenConfig):
    """Executed text architecture embedded in Jina v5 Omni checkpoints."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    max_position_embeddings: int
    rope_theta: float
    norm_epsilon: float
    pad_token_id: int
    output_dimension: int
    causal_attention: bool = True

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
            "output_dimension",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dimension % 2:
            raise ValueError("head_dimension must be even for rotary embeddings")
        if self.num_attention_heads * self.head_dimension < self.hidden_size:
            raise ValueError("attention projection cannot be narrower than hidden_size")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0:
            raise ValueError("norm_epsilon must be finite and positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must index the vocabulary")
        if self.output_dimension > self.hidden_size:
            raise ValueError("output_dimension cannot exceed hidden_size")
        return self

    @classmethod
    def from_hf_config(
        cls,
        value: Mapping[str, Any],
        *,
        output_dimension: int | None = None,
    ) -> JinaV5TextConfig:
        text = value.get("text_config")
        if not isinstance(text, Mapping):
            raise ValueError("Jina v5 config must contain text_config")
        rope = text.get("rope_parameters", {})
        if not isinstance(rope, Mapping):
            raise ValueError("text_config.rope_parameters must be an object")
        hidden = int(text["hidden_size"])
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=hidden,
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(
                text.get("num_key_value_heads", text["num_attention_heads"])
            ),
            head_dimension=int(
                text.get(
                    "head_dim",
                    hidden // int(text["num_attention_heads"]),
                )
            ),
            max_position_embeddings=int(text["max_position_embeddings"]),
            rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
            norm_epsilon=float(text.get("rms_norm_eps", 1e-6)),
            pad_token_id=(
                151643
                if text.get("pad_token_id") is None
                else int(text["pad_token_id"])
            ),
            output_dimension=(
                int(value.get("output_dimension", hidden))
                if output_dimension is None
                else output_dimension
            ),
            causal_attention=bool(text.get("is_causal", True)),
        )


class JinaV5OmniConfig(FrozenConfig):
    """The two released Jina v5 Omni architectures."""

    variant: Literal["nano", "small"]
    text: Qwen3VLTextConfig
    vision: Qwen3VLVisionConfig
    audio: Qwen2_5OmniAudioConfig
    audio_source_output_size: int
    text_attention: Literal["causal", "bidirectional"]
    image_token_id: int
    video_token_id: int
    audio_token_id: int
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    audio_start_token_id: int
    audio_end_token_id: int

    @model_validator(mode="after")
    def validate_omni_architecture(self) -> Self:
        if self.vision.output_size != self.text.hidden_size:
            raise ValueError("vision output size must equal text hidden size")
        if self.audio.output_size != self.text.hidden_size:
            raise ValueError("audio output size must equal text hidden size")
        if self.audio_source_output_size <= 0:
            raise ValueError("audio source output size must be positive")
        expected_attention = "bidirectional" if self.variant == "nano" else "causal"
        if self.text_attention != expected_attention:
            raise ValueError(
                f"Jina v5 {self.variant} text attention must be {expected_attention}"
            )
        tokens = (
            self.image_token_id,
            self.video_token_id,
            self.audio_token_id,
            self.audio_start_token_id,
            self.audio_end_token_id,
        )
        if any(not 0 <= token < self.text.vocab_size for token in tokens):
            raise ValueError("Jina v5 modality tokens must index the vocabulary")
        for token in (self.vision_start_token_id, self.vision_end_token_id):
            if token is not None and not 0 <= token < self.text.vocab_size:
                raise ValueError("Jina v5 vision tokens must index the vocabulary")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> JinaV5OmniConfig:
        model_type = str(value.get("model_type", ""))
        if model_type == "llava_eurobert_audio":
            variant: Literal["nano", "small"] = "nano"
        elif model_type == "qwen3_vl_audio":
            variant = "small"
        else:
            raise ValueError(
                "expected model_type='llava_eurobert_audio' or 'qwen3_vl_audio'"
            )
        text_value = value.get("text_config")
        if not isinstance(text_value, Mapping):
            raise ValueError("Jina v5 config must contain text_config")
        rope = text_value.get("rope_parameters", text_value.get("rope_scaling", {}))
        if not isinstance(rope, Mapping):
            raise ValueError("text_config rope configuration must be an object")
        if variant == "small":
            text = Qwen3VLTextConfig.from_hf_config(value)
        else:
            head_dimension = int(
                text_value.get(
                    "head_dim",
                    int(text_value["hidden_size"])
                    // int(text_value["num_attention_heads"]),
                )
            )
            half = head_dimension // 2
            first = half // 3
            second = (half - first) // 2
            pad_token_id = text_value.get("pad_token_id")
            text = Qwen3VLTextConfig(
                vocab_size=int(text_value["vocab_size"]),
                hidden_size=int(text_value["hidden_size"]),
                intermediate_size=int(text_value["intermediate_size"]),
                num_hidden_layers=int(text_value["num_hidden_layers"]),
                num_attention_heads=int(text_value["num_attention_heads"]),
                num_key_value_heads=int(
                    text_value.get(
                        "num_key_value_heads", text_value["num_attention_heads"]
                    )
                ),
                head_dimension=head_dimension,
                max_position_embeddings=int(text_value["max_position_embeddings"]),
                rope_theta=float(
                    rope.get("rope_theta", text_value.get("rope_theta", 10_000.0))
                ),
                mrope_section=(first, second, half - first - second),
                norm_epsilon=float(text_value.get("rms_norm_eps", 1e-5)),
                pad_token_id=(0 if pad_token_id is None else int(pad_token_id)),
                qk_norm=False,
                initializer_range=float(text_value.get("initializer_range", 0.02)),
            )
        audio = Qwen2_5OmniAudioConfig.from_hf_config(value).model_copy(
            update={"output_size": text.hidden_size}
        )
        image_token = int(
            value.get("image_token_id", value.get("image_token_index", 0))
        )
        vision = Qwen3VLVisionConfig.from_hf_config(value)
        if variant == "nano":
            # Nano discards the Qwen vision merger and uses its top-level
            # 768-wide merger instead; vision_config retains the source 1024.
            vision = vision.model_copy(update={"output_size": text.hidden_size})
        return cls(
            variant=variant,
            text=text,
            vision=vision,
            audio=audio,
            audio_source_output_size=int(value["audio_config"]["output_dim"]),
            text_attention="bidirectional" if variant == "nano" else "causal",
            image_token_id=image_token,
            # Nano intentionally uses one visual placeholder for images and videos.
            video_token_id=int(value.get("video_token_id", image_token)),
            audio_token_id=int(value["audio_token_id"]),
            vision_start_token_id=(
                None
                if value.get("vision_start_token_id") is None
                else int(value["vision_start_token_id"])
            ),
            vision_end_token_id=(
                None
                if value.get("vision_end_token_id") is None
                else int(value["vision_end_token_id"])
            ),
            audio_start_token_id=int(value["audio_start_token_id"]),
            audio_end_token_id=int(value["audio_end_token_id"]),
        )


__all__ = [
    "JINA_V5_NANO_MODEL_ID",
    "JINA_V5_NANO_REVISION",
    "JINA_V5_SMALL_MODEL_ID",
    "JINA_V5_SMALL_REVISION",
    "JinaV5OmniConfig",
    "JinaV5TextConfig",
]
