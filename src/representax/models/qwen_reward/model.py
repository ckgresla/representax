"""Native Qwen3 sequence-classification reward model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import Linear
from representax.models.qwen3_vl.model import last_valid_token_indices
from representax.models.qwen_reranker import QwenReranker, QwenRerankerBatch

from .config import QwenRewardConfig


class QwenRewardModel(eqx.Module):
    """A fully trainable Qwen3 backbone with one independent scalar head."""

    backbone: QwenReranker
    score_head: Linear
    metadata: EncoderMetadata
    config: QwenRewardConfig = eqx.field(static=True)

    @classmethod
    def from_backbone(
        cls,
        backbone: QwenReranker,
        *,
        key: PRNGKeyArray,
        model_id: str,
        revision: str,
    ) -> QwenRewardModel:
        if backbone.lm_head is not None:
            raise ValueError("reward backbones must omit the causal LM head")
        config = QwenRewardConfig(backbone=backbone.config)
        return cls(
            backbone=backbone,
            score_head=Linear.init(
                config.hidden_size,
                1,
                key=key,
                scale=config.backbone.initializer_range,
                dtype=backbone.text.token_embedding.dtype,
                bias=False,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=1,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
            config=config,
        )

    def logits(
        self,
        inputs: QwenRerankerBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        hidden = self.backbone.hidden_states(inputs, key=key)
        rows = jnp.arange(hidden.shape[0])
        positions = last_valid_token_indices(inputs.attention_mask)
        pooled = hidden[rows, positions].astype(self.score_head.weight.dtype)
        return self.score_head(pooled)[..., 0].astype(jnp.float32)

    def score(
        self,
        inputs: QwenRerankerBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        return self.logits(inputs, key=key)

    def encode(
        self,
        inputs: QwenRerankerBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch score"]:
        del route
        return self.score(inputs, key=key)[:, None]


__all__ = ["QwenRewardModel"]
