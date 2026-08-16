"""Native configuration for the pinned Transformers MPNet encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.components import Activation

MPNET_MODEL_ID = "microsoft/mpnet-base"


class MPNetConfig(FrozenConfig):
    """Architecture values for the standard bidirectional MPNet family."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    relative_attention_num_buckets: int = 32
    hidden_activation: Activation = "gelu"
    hidden_dropout_probability: float = 0.1
    attention_dropout_probability: float = 0.1
    norm_epsilon: float = 1e-12
    initializer_range: float = 0.02
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.relative_attention_num_buckets != 32:
            raise ValueError(
                "Transformers 5.3 MPNet requires relative_attention_num_buckets=32"
            )
        if self.pad_token_id != 1:
            raise ValueError("Transformers MPNet requires pad_token_id=1")
        for name, token_id in (
            ("pad_token_id", self.pad_token_id),
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
        ):
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name} must index the vocabulary")
        for name, value in (
            ("norm_epsilon", self.norm_epsilon),
            ("initializer_range", self.initializer_range),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("hidden_dropout_probability", self.hidden_dropout_probability),
            ("attention_dropout_probability", self.attention_dropout_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        return self

    @property
    def head_dimension(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> MPNetConfig:
        """Map a Transformers 5.3 MPNet config into native values."""

        activation = str(value.get("hidden_act", "gelu"))
        if activation not in ("gelu", "gelu_new", "relu", "silu"):
            raise ValueError(f"unsupported MPNet hidden activation: {activation!r}")
        return cls(
            vocab_size=int(value["vocab_size"]),
            hidden_size=int(value["hidden_size"]),
            intermediate_size=int(value["intermediate_size"]),
            num_hidden_layers=int(value["num_hidden_layers"]),
            num_attention_heads=int(value["num_attention_heads"]),
            max_position_embeddings=int(value["max_position_embeddings"]),
            relative_attention_num_buckets=int(
                value.get("relative_attention_num_buckets", 32)
            ),
            hidden_activation=activation,
            hidden_dropout_probability=float(value.get("hidden_dropout_prob", 0.1)),
            attention_dropout_probability=float(
                value.get("attention_probs_dropout_prob", 0.1)
            ),
            norm_epsilon=float(value.get("layer_norm_eps", 1e-12)),
            initializer_range=float(value.get("initializer_range", 0.02)),
            pad_token_id=int(value.get("pad_token_id", 1)),
            bos_token_id=int(value.get("bos_token_id", 0)),
            eos_token_id=int(value.get("eos_token_id", 2)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        """Serialize the supported native configuration for Transformers 5.3."""

        return {
            "architectures": ["MPNetModel"],
            "model_type": "mpnet",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "hidden_act": self.hidden_activation,
            "hidden_dropout_prob": self.hidden_dropout_probability,
            "attention_probs_dropout_prob": self.attention_dropout_probability,
            "layer_norm_eps": self.norm_epsilon,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "transformers_version": "5.3.0",
        }


__all__ = ["MPNET_MODEL_ID", "MPNetConfig"]
