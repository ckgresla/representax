"""Validated configuration for native Qwen2 and Qwen3 text rerankers."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.integrations.huggingface import load_hf_config
from representax.models.qwen2_5_omni import Qwen2_5OmniTextConfig
from representax.models.qwen3_vl import Qwen3VLTextConfig

QWEN3_RERANKER_0_6B_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_RERANKER_0_6B_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
QWEN3_RERANKER_4B_MODEL_ID = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKER_4B_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
QWEN3_RERANKER_8B_MODEL_ID = "Qwen/Qwen3-Reranker-8B"
QWEN3_RERANKER_8B_REVISION = "77d193c791ed757ca307ee72715aa132723da912"
CONTEXTUAL_QWEN3_RERANKER_MODEL_ID = (
    "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
)
CONTEXTUAL_QWEN3_RERANKER_REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"
MXBAI_QWEN2_RERANKER_BASE_MODEL_ID = "mixedbread-ai/mxbai-rerank-base-v2"
MXBAI_QWEN2_RERANKER_BASE_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"
MXBAI_QWEN2_RERANKER_LARGE_MODEL_ID = "mixedbread-ai/mxbai-rerank-large-v2"
MXBAI_QWEN2_RERANKER_LARGE_REVISION = "ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6"

QwenGeneration = Literal["qwen2", "qwen3"]
ScoreActivation = Literal["identity", "sigmoid"]


class QwenRerankerConfig(FrozenConfig):
    """One causal Qwen decoder plus its final-position relevance projection."""

    generation: QwenGeneration
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
    true_token_id: int
    false_token_id: int | None
    tie_word_embeddings: bool
    score_activation: ScoreActivation = "identity"
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
            raise ValueError("key/value heads must divide attention heads")
        if self.head_dimension % 2:
            raise ValueError("head_dimension must be even")
        for name in ("pad_token_id", "true_token_id"):
            if not 0 <= getattr(self, name) < self.vocab_size:
                raise ValueError(f"{name} must index the vocabulary")
        if self.false_token_id is not None:
            if not 0 <= self.false_token_id < self.vocab_size:
                raise ValueError("false_token_id must index the vocabulary")
            if self.false_token_id == self.true_token_id:
                raise ValueError("true and false score tokens must differ")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive or None")
        for name in ("rope_theta", "norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(
        cls,
        value: Mapping[str, Any],
        *,
        true_token_id: int,
        false_token_id: int | None,
        score_activation: ScoreActivation = "identity",
    ) -> QwenRerankerConfig:
        model_type = str(value.get("model_type", ""))
        if model_type not in {"qwen2", "qwen3"}:
            raise ValueError("Qwen rerankers require model_type='qwen2' or 'qwen3'")
        heads = int(value["num_attention_heads"])
        pad_token_id = value.get("pad_token_id", value.get("eos_token_id", 151643))
        if pad_token_id is None:
            pad_token_id = value.get("eos_token_id", 151643)
        return cls(
            generation=model_type,
            vocab_size=int(value["vocab_size"]),
            hidden_size=int(value["hidden_size"]),
            intermediate_size=int(value["intermediate_size"]),
            num_hidden_layers=int(value["num_hidden_layers"]),
            num_attention_heads=heads,
            num_key_value_heads=int(value.get("num_key_value_heads", heads)),
            head_dimension=int(
                value.get("head_dim", int(value["hidden_size"]) // heads)
            ),
            max_position_embeddings=int(value["max_position_embeddings"]),
            rope_theta=float(value.get("rope_theta", 1_000_000.0)),
            norm_epsilon=float(value.get("rms_norm_eps", 1e-6)),
            pad_token_id=int(pad_token_id),
            true_token_id=true_token_id,
            false_token_id=false_token_id,
            tie_word_embeddings=bool(value.get("tie_word_embeddings", False)),
            score_activation=score_activation,
            sliding_window=(
                None
                if not bool(value.get("use_sliding_window", False))
                or value.get("sliding_window") is None
                else int(value["sliding_window"])
            ),
            initializer_range=float(value.get("initializer_range", 0.02)),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path) -> QwenRerankerConfig:
        root = Path(checkpoint)
        score = load_hf_config(root / "1_LogitScore")
        sentence_config_path = root / "config_sentence_transformers.json"
        if not sentence_config_path.is_file():
            raise FileNotFoundError(
                f"Sentence Transformers config not found: {sentence_config_path}"
            )
        sentence_config = json.loads(sentence_config_path.read_text())
        if not isinstance(sentence_config, dict):
            raise TypeError("Sentence Transformers config must contain a JSON object")
        activation_name = sentence_config.get("activation_fn")
        if activation_name is None:
            # CrossEncoder's single-label prediction default is sigmoid.
            score_activation: ScoreActivation = "sigmoid"
        elif str(activation_name).endswith(".Identity"):
            score_activation = "identity"
        elif str(activation_name).endswith(".Sigmoid"):
            score_activation = "sigmoid"
        else:
            raise ValueError(f"unsupported CrossEncoder activation {activation_name!r}")
        return cls.from_hf_config(
            load_hf_config(root),
            true_token_id=int(score["true_token_id"]),
            false_token_id=(
                None
                if score.get("false_token_id") is None
                else int(score["false_token_id"])
            ),
            score_activation=score_activation,
        )

    def qwen2_tower_config(self) -> Qwen2_5OmniTextConfig:
        """Construct the existing compute-native Qwen2 tower configuration."""

        if self.generation != "qwen2":
            raise TypeError("qwen2_tower_config requires a Qwen2 checkpoint")
        return Qwen2_5OmniTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dimension=self.head_dimension,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            mrope_section=(self.head_dimension // 2, 0, 0),
            norm_epsilon=self.norm_epsilon,
            layer_types=("full_attention",) * self.num_hidden_layers,
            sliding_window=self.sliding_window,
            initializer_range=self.initializer_range,
        )

    def qwen3_tower_config(self) -> Qwen3VLTextConfig:
        """Construct the existing native Qwen3 decoder configuration."""

        if self.generation != "qwen3":
            raise TypeError("qwen3_tower_config requires a Qwen3 checkpoint")
        return Qwen3VLTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dimension=self.head_dimension,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            mrope_section=(self.head_dimension // 2, 0, 0),
            norm_epsilon=self.norm_epsilon,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
        )


__all__ = [
    "CONTEXTUAL_QWEN3_RERANKER_MODEL_ID",
    "CONTEXTUAL_QWEN3_RERANKER_REVISION",
    "MXBAI_QWEN2_RERANKER_BASE_MODEL_ID",
    "MXBAI_QWEN2_RERANKER_BASE_REVISION",
    "MXBAI_QWEN2_RERANKER_LARGE_MODEL_ID",
    "MXBAI_QWEN2_RERANKER_LARGE_REVISION",
    "QWEN3_RERANKER_0_6B_MODEL_ID",
    "QWEN3_RERANKER_0_6B_REVISION",
    "QWEN3_RERANKER_4B_MODEL_ID",
    "QWEN3_RERANKER_4B_REVISION",
    "QWEN3_RERANKER_8B_MODEL_ID",
    "QWEN3_RERANKER_8B_REVISION",
    "QwenGeneration",
    "QwenRerankerConfig",
    "ScoreActivation",
]
