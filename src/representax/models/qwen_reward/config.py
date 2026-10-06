"""Configuration for the canonical Qwen3 scalar reward model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from representax._config import FrozenConfig
from representax.models.qwen_reranker import QwenRerankerConfig

QWEN3_REWARD_0_6B_MODEL_ID = "Qwen/Qwen3-0.6B"
QWEN3_REWARD_0_6B_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class QwenRewardConfig(FrozenConfig):
    """A Qwen3 text tower and independent scalar sequence head."""

    backbone: QwenRerankerConfig

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> QwenRewardConfig:
        backbone = QwenRerankerConfig.from_hf_config(
            value,
            true_token_id=0,
            false_token_id=None,
        )
        if backbone.generation != "qwen3":
            raise ValueError("canonical Qwen reward models require model_type='qwen3'")
        # The reward model does not retain or train the causal language-model head.
        backbone = backbone.model_copy(update={"tie_word_embeddings": True})
        return cls(backbone=backbone)

    @property
    def hidden_size(self) -> int:
        return self.backbone.hidden_size


__all__ = [
    "QWEN3_REWARD_0_6B_MODEL_ID",
    "QWEN3_REWARD_0_6B_REVISION",
    "QwenRewardConfig",
]
