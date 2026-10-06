"""ModernBERT MLM head over the existing text tower, with tied token weights."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from representax.models.components import AttentionImplementation, LayerNorm, Linear
from representax.planning import RematerializationPolicy

from .config import ModernVBERTTextConfig
from .model import ModernVBERTTextBatch, ModernVBERTTextEncoder


class ModernBERTMaskedLM(eqx.Module):
    """Dense -> exact GELU -> LayerNorm -> tied vocabulary projection.

    The token embedding occurs once in the parameter tree. Both its input and
    output uses contribute to the same gradient and optimizer state.
    """

    encoder: ModernVBERTTextEncoder
    dense: Linear
    norm: LayerNorm
    decoder_bias: Array | None

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig | dict[str, Any],
        *,
        key: PRNGKeyArray,
        parameter_dtype: Any = jnp.float32,
        compute_dtype: Any = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        head_bias: bool = False,
        decoder_bias: bool = True,
    ) -> ModernBERTMaskedLM:
        config = ModernVBERTTextConfig.model_validate(config)
        encoder_key, head_key = jax.random.split(key)
        dtype = jnp.dtype(parameter_dtype)
        return cls(
            encoder=ModernVBERTTextEncoder.init(
                config,
                key=encoder_key,
                parameter_dtype=dtype,
                compute_dtype=jnp.dtype(compute_dtype),
                attention_implementation=attention_implementation,
                rematerialization=rematerialization,
            ),
            dense=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=head_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=head_bias,
            ).input_major(),
            norm=LayerNorm.init(
                config.hidden_size, epsilon=config.norm_epsilon, dtype=dtype
            ),
            decoder_bias=jnp.zeros((config.vocab_size,), dtype=dtype)
            if decoder_bias
            else None,
        )

    def predict_masked(
        self,
        inputs: ModernVBERTTextBatch,
        positions: Int[Array, "batch target"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch target vocabulary"]:
        hidden = self.encoder.hidden_states(inputs, key=key)
        if positions.ndim != 2 or positions.shape[0] != hidden.shape[0]:
            raise ValueError("masked positions must have shape [batch, target]")
        # Keep the large vocabulary projection off unsupervised token positions.
        selected = jnp.take_along_axis(hidden, positions[..., None], axis=1)
        projected = self.norm(jax.nn.gelu(self.dense(selected), approximate=False))
        return Linear(
            weight=self.encoder.tower.token_embedding, bias=self.decoder_bias
        )(projected)
