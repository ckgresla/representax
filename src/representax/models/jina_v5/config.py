"""Native configuration for Jina Embeddings v5 text towers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from representax._config import FrozenConfig

JINA_V5_SMALL_MODEL_ID = "jinaai/jina-embeddings-v5-omni-small-retrieval"
JINA_V5_SMALL_REVISION = "12949877f0092093f366c6450340011320152a05"


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
        )


__all__ = [
    "JINA_V5_SMALL_MODEL_ID",
    "JINA_V5_SMALL_REVISION",
    "JinaV5TextConfig",
]
