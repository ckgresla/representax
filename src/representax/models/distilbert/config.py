"""Native configuration for Hugging Face DistilBERT encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from representax.models.bert import BertConfig

CLIP_MULTILINGUAL_MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
CLIP_MULTILINGUAL_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"


class DistilBertConfig(BertConfig):
    """DistilBERT values expressed through the shared BERT layer contract."""

    type_vocab_size: int = 1

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> DistilBertConfig:
        if str(value.get("model_type", "")) != "distilbert":
            raise ValueError("expected model_type='distilbert'")
        activation = str(value.get("activation", "gelu"))
        if activation not in ("gelu", "gelu_new", "relu", "silu"):
            raise ValueError(f"unsupported DistilBERT activation: {activation!r}")
        return cls(
            vocab_size=int(value["vocab_size"]),
            hidden_size=int(value["dim"]),
            intermediate_size=int(value["hidden_dim"]),
            num_hidden_layers=int(value["n_layers"]),
            num_attention_heads=int(value["n_heads"]),
            max_position_embeddings=int(value["max_position_embeddings"]),
            type_vocab_size=1,
            hidden_activation=activation,
            hidden_dropout_probability=float(value.get("dropout", 0.1)),
            attention_dropout_probability=float(value.get("attention_dropout", 0.1)),
            norm_epsilon=float(value.get("layer_norm_eps", 1e-12)),
            initializer_range=float(value.get("initializer_range", 0.02)),
            pad_token_id=int(value.get("pad_token_id", 0)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        return {
            "architectures": ["DistilBertModel"],
            "model_type": "distilbert",
            "vocab_size": self.vocab_size,
            "dim": self.hidden_size,
            "hidden_dim": self.intermediate_size,
            "n_layers": self.num_hidden_layers,
            "n_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "activation": self.hidden_activation,
            "dropout": self.hidden_dropout_probability,
            "attention_dropout": self.attention_dropout_probability,
            "layer_norm_eps": self.norm_epsilon,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "sinusoidal_pos_embds": False,
            "tie_weights_": True,
        }


__all__ = [
    "CLIP_MULTILINGUAL_MODEL_ID",
    "CLIP_MULTILINGUAL_REVISION",
    "DistilBertConfig",
]
