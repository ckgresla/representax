"""Native Qwen2/Qwen3 final-token reranking model."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import AttentionImplementation
from representax.models.qwen2_5_omni import Qwen2_5OmniTextTower
from representax.models.qwen3_vl import Qwen3VLTextTower
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import QwenRerankerConfig


class QwenRerankerBatch(eqx.Module):
    """One finite bucket of checkpoint-formatted query/document pairs."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")


class QwenReranker(eqx.Module):
    """Causal decoder computing only the one or two logits used for ranking."""

    text: Qwen2_5OmniTextTower | Qwen3VLTextTower
    lm_head: Float[Array, "vocabulary hidden"] | None
    metadata: EncoderMetadata
    config: QwenRerankerConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load native weights and the checkpoint-authored pair processor."""

        from .loading import load_qwen_reranker

        return load_qwen_reranker(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: QwenRerankerConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/qwen-reranker",
        revision: str = "random-init",
    ) -> QwenReranker:
        text_key, head_key = jax.random.split(key)
        if config.generation == "qwen2":
            text = Qwen2_5OmniTextTower.init(
                config.qwen2_tower_config(), key=text_key, dtype=parameter_dtype
            )
        else:
            text = Qwen3VLTextTower.init(
                config.qwen3_tower_config(), key=text_key, dtype=parameter_dtype
            )
        lm_head = None
        if not config.tie_word_embeddings:
            lm_head = config.initializer_range * jax.random.normal(
                head_key,
                (config.vocab_size, config.hidden_size),
                dtype=parameter_dtype,
            )
        return cls(
            text=text,
            lm_head=lm_head,
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=1,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def hidden_states(
        self,
        inputs: QwenRerankerBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, QwenRerankerBatch):
            raise TypeError("Qwen reranker inputs must be QwenRerankerBatch")
        batch, sequence = inputs.input_ids.shape
        positions = jnp.broadcast_to(
            jnp.arange(sequence, dtype=jnp.int32)[None, None],
            (3, batch, sequence),
        )
        compute_dtype = active_compute_dtype(self.compute_dtype)
        if isinstance(self.text, Qwen2_5OmniTextTower):
            return self.text(
                inputs.input_ids,
                inputs.attention_mask,
                positions,
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
        if isinstance(self.text, Qwen3VLTextTower):
            return self.text(
                inputs.input_ids,
                inputs.attention_mask,
                positions,
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
        raise TypeError("unsupported Qwen text tower")

    def logits(
        self,
        inputs: QwenRerankerBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        hidden = self.hidden_states(inputs, key=key)[:, -1].astype(jnp.float32)
        head = self.text.token_embedding if self.lm_head is None else self.lm_head
        true = head[self.config.true_token_id].astype(jnp.float32)
        score = hidden @ true
        if self.config.false_token_id is not None:
            false = head[self.config.false_token_id].astype(jnp.float32)
            score = score - hidden @ false
        return score

    def score(
        self,
        inputs: QwenRerankerBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        """Return checkpoint-configured inference scores."""

        logits = self.logits(inputs, key=key)
        if self.config.score_activation == "sigmoid":
            return jax.nn.sigmoid(logits)
        return logits

    def encode(
        self,
        inputs: QwenRerankerBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch score"]:
        del route
        return self.score(inputs, key=key)[:, None]


__all__ = ["QwenReranker", "QwenRerankerBatch"]
