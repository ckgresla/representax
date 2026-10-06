"""Validated configuration for NVIDIA Llama Nemotron VL checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self, cast

from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.decoder import RotaryDecoderConfig
from representax.models.modernvbert.config import ModernVBERTVisionConfig

LLAMA_NEMOTRON_EMBED_VL_MODEL_ID = "nvidia/llama-nemotron-embed-vl-1b-v2"
LLAMA_NEMOTRON_EMBED_VL_REVISION = "582e3bf72aee355e3c59ed89de53543c5b0657ee"
LLAMA_NEMOTRON_RERANK_VL_MODEL_ID = "nvidia/llama-nemotron-rerank-vl-1b-v2"
LLAMA_NEMOTRON_RERANK_VL_REVISION = "9e95da054312436dfc703319dd2b793a3bee2465"

NemotronVLMode = Literal["embedding", "reranking"]


class LlamaNemotronVLConfig(FrozenConfig):
    """SigLIP vision, pixel-unshuffle projector, and bidirectional Llama."""

    mode: NemotronVLMode
    text: RotaryDecoderConfig
    vision: ModernVBERTVisionConfig
    image_context_token_id: int
    downsample_ratio: float
    pooling: Literal["avg"] = "avg"
    temperature: float = 1.0
    output_dimension: int = 1

    @property
    def pixel_shuffle_factor(self) -> int:
        return round(1 / self.downsample_ratio)

    @property
    def image_sequence_length(self) -> int:
        return self.vision.patch_count // self.pixel_shuffle_factor**2

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        if self.text.attention_mode != "bidirectional":
            raise ValueError("Llama Nemotron VL requires bidirectional attention")
        if not 0 <= self.image_context_token_id < self.text.vocab_size:
            raise ValueError("image context token must index the text vocabulary")
        if not math.isfinite(self.downsample_ratio) or not (
            0 < self.downsample_ratio <= 1
        ):
            raise ValueError("downsample_ratio must be in (0, 1]")
        factor = self.pixel_shuffle_factor
        if not math.isclose(self.downsample_ratio, 1 / factor):
            raise ValueError("downsample_ratio must be the reciprocal of an integer")
        if self.vision.patch_grid % factor:
            raise ValueError("vision patch grid must divide the pixel-shuffle factor")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.mode == "embedding" and self.output_dimension != self.text.hidden_size:
            raise ValueError("embedding output dimension must equal text hidden size")
        if self.mode == "reranking" and self.output_dimension <= 0:
            raise ValueError("reranking output dimension must be positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> LlamaNemotronVLConfig:
        model_type = str(value.get("model_type", ""))
        modes = {
            "llama_nemotron_vl": "embedding",
            "llama_nemotron_vl_rerank": "reranking",
        }
        try:
            mode = cast(NemotronVLMode, modes[model_type])
        except KeyError as error:
            raise ValueError(
                f"unsupported Llama Nemotron VL type {model_type!r}"
            ) from error
        text = value.get("llm_config")
        if not isinstance(text, Mapping):
            raise ValueError("Llama Nemotron VL config must contain llm_config")
        parsed_text = RotaryDecoderConfig.from_hf_config(text)
        labels = value.get("id2label")
        output_dimension = (
            len(labels)
            if mode == "reranking" and isinstance(labels, Mapping)
            else parsed_text.hidden_size
        )
        pooling = value.get("pooling", text.get("pooling", "avg"))
        if pooling != "avg":
            raise ValueError(f"unsupported Llama Nemotron pooling {pooling!r}")
        return cls(
            mode=mode,
            text=parsed_text,
            vision=ModernVBERTVisionConfig.from_hf_config(value),
            image_context_token_id=int(value["img_context_token_id"]),
            downsample_ratio=float(value.get("downsample_ratio", 0.5)),
            pooling="avg",
            temperature=float(text.get("temperature", 1.0)),
            output_dimension=output_dimension,
        )

    def to_hf_config(self) -> dict[str, Any]:
        text = self.text.to_hf_config()
        reranking = self.mode == "reranking"
        text.update(
            {
                "architectures": [
                    "LlamaBidirectionalForSequenceClassification"
                    if reranking
                    else "LlamaBidirectionalModel"
                ],
                "model_type": "llama_bidirec",
                "pooling": self.pooling,
                "temperature": self.temperature,
            }
        )
        vision = {
            "model_type": "siglip_vision_model",
            "hidden_size": self.vision.hidden_size,
            "intermediate_size": self.vision.intermediate_size,
            "num_hidden_layers": self.vision.num_hidden_layers,
            "num_attention_heads": self.vision.num_attention_heads,
            "num_channels": self.vision.num_channels,
            "image_size": self.vision.image_size,
            "patch_size": self.vision.patch_size,
            "layer_norm_eps": self.vision.norm_epsilon,
            "hidden_act": self.vision.hidden_activation,
            # The source carries SigLIP's optional classification pooler, but
            # Llama Nemotron consumes only patch hidden states. The exported
            # graph should therefore require only tensors it actually executes.
            "vision_use_head": False,
        }
        auto_map = {
            "AutoConfig": (
                "configuration_llama_nemotron_vl."
                + (
                    "LlamaNemotronVLForSequenceClassificationConfig"
                    if reranking
                    else "LlamaNemotronVLConfig"
                )
            ),
            "AutoModel": (
                "modeling_llama_nemotron_vl."
                + (
                    "LlamaNemotronVLForSequenceClassification"
                    if reranking
                    else "LlamaNemotronVLModel"
                )
            ),
        }
        if reranking:
            auto_map["AutoModelForSequenceClassification"] = auto_map["AutoModel"]
        return {
            "architectures": [
                "LlamaNemotronVLForSequenceClassification"
                if reranking
                else "LlamaNemotronVLModel"
            ],
            "model_type": (
                "llama_nemotron_vl_rerank" if reranking else "llama_nemotron_vl"
            ),
            "auto_map": auto_map,
            "llm_config": text,
            "vision_config": vision,
            "img_context_token_id": self.image_context_token_id,
            "downsample_ratio": self.downsample_ratio,
            "pooling": self.pooling,
            "force_image_size": self.vision.image_size,
            "select_layer": -1,
            "ps_version": "v2",
            "dynamic_image_size": True,
            "use_thumbnail": True,
            "template": "bidirectional-llama-retriever",
            "max_input_tiles": 4 if reranking else 2,
            "q_max_length": 512,
            "p_max_length": 10240 if reranking else 4096,
            "query_prefix": "query:",
            "passage_prefix": "passage:",
            "bidirectional_attention": True,
            "temperature": self.temperature,
            "prompt_template": "v1" if reranking else None,
            "vocab_size": self.text.vocab_size,
            "id2label": {
                str(index): f"LABEL_{index}"
                for index in range(self.output_dimension if reranking else 2)
            },
            "label2id": {
                f"LABEL_{index}": index
                for index in range(self.output_dimension if reranking else 2)
            },
            "torch_dtype": "bfloat16",
        }


__all__ = [
    "LLAMA_NEMOTRON_EMBED_VL_MODEL_ID",
    "LLAMA_NEMOTRON_EMBED_VL_REVISION",
    "LLAMA_NEMOTRON_RERANK_VL_MODEL_ID",
    "LLAMA_NEMOTRON_RERANK_VL_REVISION",
    "LlamaNemotronVLConfig",
]
