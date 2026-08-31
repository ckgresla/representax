"""Complete native Qwen2.5-Omni representation model."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    embedding_lookup,
    l2_normalize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .audio import Qwen2_5OmniAudioTower
from .config import Qwen2_5OmniConfig
from .text import Qwen2_5OmniTextTower
from .vision import Qwen2_5OmniVisionTower


class Qwen2_5OmniBatch(eqx.Module):
    """Finite-shape native inputs after model-associated host processing."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    position_ids: Int[Array, "batch position sequence"] | None = None

    pixel_values: Float[Array, "patch pixel"] | None = None
    patch_valid: Bool[Array, " patch"] | None = None
    vision_full_segment_ids: Int[Array, " patch"] | None = None
    vision_window_segment_ids: Int[Array, " patch"] | None = None
    vision_position_ids: Int[Array, "patch coordinate"] | None = None
    reverse_merged_indices: Int[Array, " merged"] | None = None
    visual_token_indices: Int[Array, " visual"] | None = None
    visual_token_valid: Bool[Array, " visual"] | None = None

    input_features: Float[Array, "batch chunk mel feature"] | None = None
    audio_feature_valid: Bool[Array, "batch chunk feature"] | None = None
    audio_after_cnn_valid: Bool[Array, "batch chunk cnn_sequence"] | None = None
    audio_pool_indices: Int[Array, "batch audio_token pair"] | None = None
    audio_token_indices: Int[Array, "batch audio_token"] | None = None
    audio_token_valid: Bool[Array, "batch audio_token"] | None = None

    @property
    def batch_size(self) -> int:
        """Return logical samples independently of packed media dimensions."""

        return self.input_ids.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        if self.position_ids is not None and self.position_ids.shape != (
            self.batch_size,
            3,
            self.input_ids.shape[1],
        ):
            raise ValueError("position_ids must have shape [batch, 3, sequence]")

        vision_values = (
            self.patch_valid,
            self.vision_full_segment_ids,
            self.vision_window_segment_ids,
            self.vision_position_ids,
            self.reverse_merged_indices,
            self.visual_token_indices,
            self.visual_token_valid,
        )
        if self.pixel_values is None:
            if any(value is not None for value in vision_values):
                raise ValueError("vision layout arrays require pixel_values")
        elif any(value is None for value in vision_values):
            raise ValueError("pixel_values require the complete vision layout")

        audio_values = (
            self.audio_feature_valid,
            self.audio_after_cnn_valid,
            self.audio_pool_indices,
            self.audio_token_indices,
            self.audio_token_valid,
        )
        if self.input_features is None:
            if any(value is not None for value in audio_values):
                raise ValueError("audio layout arrays require input_features")
        elif any(value is None for value in audio_values):
            raise ValueError("input_features require the complete audio layout")
        elif any(
            value is not None and value.shape[0] != self.batch_size
            for value in audio_values
        ):
            raise ValueError("audio layout arrays must be row-major")


def replace_tokens(
    hidden: Float[Array, "batch sequence hidden"],
    values: Float[Array, "token hidden"],
    indices: Int[Array, " token"],
    valid: Bool[Array, " token"],
) -> Float[Array, "batch sequence hidden"]:
    """Replace valid flattened token positions without padded-index collisions."""

    flattened = hidden.reshape((-1, hidden.shape[-1]))
    current = flattened[indices]
    delta = jnp.where(valid[:, None], values - current, 0)
    return flattened.at[indices].add(delta).reshape(hidden.shape)


def last_valid_token_indices(
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
) -> Int[Array, " batch"]:
    """Return the final valid token position for left- or right-padded rows."""

    positions = jnp.arange(attention_mask.shape[-1], dtype=jnp.int32)
    return jnp.max(jnp.where(attention_mask.astype(bool), positions, -1), axis=-1)


class Qwen2_5OmniEncoder(eqx.Module):
    """Four-modality thinker with normalized last-token representation pooling."""

    text: Qwen2_5OmniTextTower
    vision: Qwen2_5OmniVisionTower
    audio: Qwen2_5OmniAudioTower
    metadata: EncoderMetadata
    config: Qwen2_5OmniConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)
    text_attention: Literal["causal", "bidirectional"] = eqx.field(
        static=True, default="causal"
    )
    pooling: Literal["last", "mean"] = eqx.field(static=True, default="last")
    source_model_type: Literal["qwen2_5_omni_thinker", "nvomniembed"] = eqx.field(
        static=True, default="qwen2_5_omni_thinker"
    )

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load native weights and their model-associated processor once."""

        from .loading import load_qwen2_5_omni

        return load_qwen2_5_omni(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/qwen2.5-omni",
        revision: str = "random-init",
        text_attention: Literal["causal", "bidirectional"] = "causal",
        pooling: Literal["last", "mean"] = "last",
        source_model_type: Literal[
            "qwen2_5_omni_thinker", "nvomniembed"
        ] = "qwen2_5_omni_thinker",
    ) -> Qwen2_5OmniEncoder:
        text_key, vision_key, audio_key = jax.random.split(key, 3)
        return cls(
            text=Qwen2_5OmniTextTower.init(
                config.text,
                key=text_key,
                dtype=parameter_dtype,
            ),
            vision=Qwen2_5OmniVisionTower.init(
                config.vision,
                key=vision_key,
                dtype=parameter_dtype,
            ),
            audio=Qwen2_5OmniAudioTower.init(
                config.audio,
                key=audio_key,
                dtype=parameter_dtype,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset(
                    {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
                ),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
            text_attention=text_attention,
            pooling=pooling,
            source_model_type=source_model_type,
        )

    def hidden_states(
        self,
        inputs: Qwen2_5OmniBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, Qwen2_5OmniBatch):
            raise TypeError("Qwen2.5-Omni inputs must be Qwen2_5OmniBatch")
        compute_dtype = active_compute_dtype(self.compute_dtype)
        hidden = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
        )

        if inputs.pixel_values is not None:
            assert inputs.patch_valid is not None
            assert inputs.vision_full_segment_ids is not None
            assert inputs.vision_window_segment_ids is not None
            assert inputs.vision_position_ids is not None
            assert inputs.reverse_merged_indices is not None
            assert inputs.visual_token_indices is not None
            assert inputs.visual_token_valid is not None
            visual = self.vision(
                inputs.pixel_values,
                inputs.patch_valid,
                inputs.vision_full_segment_ids,
                inputs.vision_window_segment_ids,
                inputs.vision_position_ids,
                inputs.reverse_merged_indices,
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
            hidden = replace_tokens(
                hidden,
                visual.astype(hidden.dtype),
                inputs.visual_token_indices,
                inputs.visual_token_valid,
            )

        if inputs.input_features is not None:
            assert inputs.audio_feature_valid is not None
            assert inputs.audio_after_cnn_valid is not None
            assert inputs.audio_pool_indices is not None
            assert inputs.audio_token_indices is not None
            assert inputs.audio_token_valid is not None
            batch_size, chunk_count, mel_bins, feature_count = (
                inputs.input_features.shape
            )
            token_count = inputs.audio_token_valid.shape[1]
            cnn_size = self.config.audio.window_size
            pool_offsets = (
                jnp.arange(batch_size, dtype=jnp.int32) * chunk_count * cnn_size
            )
            audio = self.audio(
                inputs.input_features.reshape(
                    batch_size * chunk_count,
                    mel_bins,
                    feature_count,
                ),
                inputs.audio_feature_valid.reshape(batch_size * chunk_count, -1),
                inputs.audio_after_cnn_valid.reshape(batch_size * chunk_count, -1),
                (inputs.audio_pool_indices + pool_offsets[:, None, None]).reshape(
                    batch_size * token_count, 2
                ),
                inputs.audio_token_valid.reshape(batch_size * token_count),
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
            sequence_offsets = (
                jnp.arange(batch_size, dtype=jnp.int32) * inputs.input_ids.shape[1]
            )
            hidden = replace_tokens(
                hidden,
                audio.astype(hidden.dtype),
                (inputs.audio_token_indices + sequence_offsets[:, None]).reshape(-1),
                inputs.audio_token_valid.reshape(-1),
            )

        position_ids = inputs.position_ids
        if position_ids is None:
            batch, sequence = inputs.input_ids.shape
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence, dtype=jnp.int32)[None, None, :],
                (batch, 3, sequence),
            )
        return self.text(
            inputs.input_ids,
            inputs.attention_mask,
            jnp.swapaxes(position_ids, 0, 1),
            inputs_embeds=hidden,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
            causal_attention=self.text_attention == "causal",
        )

    def encode(
        self,
        inputs: Qwen2_5OmniBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        if self.pooling == "last":
            last = last_valid_token_indices(inputs.attention_mask)
            pooled = hidden[jnp.arange(hidden.shape[0]), last]
        elif self.pooling == "mean":
            mask = inputs.attention_mask.astype(jnp.float32)
            pooled = jnp.sum(hidden.astype(jnp.float32) * mask[..., None], axis=1)
            pooled = pooled / jnp.maximum(jnp.sum(mask, axis=1, keepdims=True), 1)
        else:
            raise ValueError(f"unsupported pooling mode: {self.pooling!r}")
        return l2_normalize(pooled)


__all__ = [
    "Qwen2_5OmniBatch",
    "Qwen2_5OmniEncoder",
    "last_valid_token_indices",
    "replace_tokens",
]
