"""Complete native Jina Embeddings v5 Omni encoder."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Route
from representax.models.components import (
    AttentionImplementation,
    embedding_lookup,
    l2_normalize,
)
from representax.models.qwen2_5_omni.audio import Qwen2_5OmniAudioTower
from representax.models.qwen3_vl.text import Qwen3VLTextTower
from representax.models.qwen3_vl.vision import Qwen3VLVisionTower
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import JinaV5OmniConfig


class JinaV5OmniBatch(eqx.Module):
    """Finite text, image, video, and audio inputs for Jina v5 Omni."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    position_ids: Int[Array, "position batch sequence"] | None = None

    pixel_values: Float[Array, "patch pixel"] | None = None
    patch_valid: Bool[Array, " patch"] | None = None
    vision_segment_ids: Int[Array, " patch"] | None = None
    vision_position_ids: Int[Array, "patch coordinate"] | None = None
    position_interpolation_indices: Int[Array, "corner patch"] | None = None
    position_interpolation_weights: Float[Array, "corner patch"] | None = None
    visual_token_indices: Int[Array, " visual"] | None = None
    visual_token_valid: Bool[Array, " visual"] | None = None
    # Processor-built lookup into flattened merge groups; -1 denotes padding.
    visual_group_indices: Int[Array, "batch visual_capacity"] | None = None

    input_features: Float[Array, "batch chunk mel feature"] | None = None
    audio_feature_valid: Bool[Array, "batch chunk feature"] | None = None
    audio_after_cnn_valid: Bool[Array, "batch chunk cnn_sequence"] | None = None
    audio_pool_indices: Int[Array, "batch audio_token pair"] | None = None
    audio_token_indices: Int[Array, "batch audio_token"] | None = None
    audio_token_valid: Bool[Array, "batch audio_token"] | None = None

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def batch_to_scan(self, *, local_chunk_size: int) -> JinaV5OmniBatch:
        """Return scan-major chunks with local media indices and edge-padded rows."""
        from .chunking import batch_to_scan

        return batch_to_scan(self, local_chunk_size=local_chunk_size)

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        if self.position_ids is not None and self.position_ids.shape != (
            3,
            *self.input_ids.shape,
        ):
            raise ValueError("position_ids must have shape [3, batch, sequence]")
        if self.visual_group_indices is not None and (
            self.pixel_values is None
            or self.visual_group_indices.ndim != 2
            or self.visual_group_indices.shape[0] != self.batch_size
        ):
            raise ValueError(
                "visual_group_indices requires vision and batch-major rows"
            )

        vision = (
            self.patch_valid,
            self.vision_segment_ids,
            self.vision_position_ids,
            self.position_interpolation_indices,
            self.position_interpolation_weights,
            self.visual_token_indices,
            self.visual_token_valid,
        )
        if self.pixel_values is None:
            if any(value is not None for value in vision):
                raise ValueError("vision layout arrays require pixel_values")
        elif any(value is None for value in vision):
            raise ValueError("pixel_values require the complete vision layout")

        audio = (
            self.audio_feature_valid,
            self.audio_after_cnn_valid,
            self.audio_pool_indices,
            self.audio_token_indices,
            self.audio_token_valid,
        )
        if self.input_features is None:
            if any(value is not None for value in audio):
                raise ValueError("audio layout arrays require input_features")
        elif any(value is None for value in audio):
            raise ValueError("input_features require the complete audio layout")
        elif self.input_features.shape[0] != self.batch_size or any(
            value is not None and value.shape[0] != self.batch_size for value in audio
        ):
            raise ValueError("audio layout arrays must be row-major")


def _replace_tokens(
    hidden: Float[Array, "batch sequence hidden"],
    values: Float[Array, "token hidden"],
    indices: Int[Array, " token"],
    valid: Bool[Array, " token"],
) -> Float[Array, "batch sequence hidden"]:
    flattened = hidden.reshape((-1, hidden.shape[-1]))
    current = flattened[indices]
    delta = jnp.where(valid[:, None], values - current, 0)
    return flattened.at[indices].add(delta).reshape(hidden.shape)


class JinaV5OmniEncoder(eqx.Module):
    """Nano or Small Jina v5 towers in their shared embedding space."""

    text: Qwen3VLTextTower
    vision: Qwen3VLVisionTower
    audio: Qwen2_5OmniAudioTower
    metadata: EncoderMetadata
    config: JinaV5OmniConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_jina_v5_omni

        return load_jina_v5_omni(model_name_or_path, **options)

    def hidden_states(
        self,
        inputs: JinaV5OmniBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, JinaV5OmniBatch):
            raise TypeError("Jina v5 Omni inputs must be JinaV5OmniBatch")
        compute_dtype = active_compute_dtype(self.compute_dtype)
        hidden = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
        )

        if inputs.pixel_values is not None:
            assert inputs.patch_valid is not None
            assert inputs.vision_segment_ids is not None
            assert inputs.vision_position_ids is not None
            assert inputs.position_interpolation_indices is not None
            assert inputs.position_interpolation_weights is not None
            assert inputs.visual_token_indices is not None
            assert inputs.visual_token_valid is not None
            visual, deepstack = self.vision(
                inputs.pixel_values,
                inputs.patch_valid,
                inputs.vision_segment_ids,
                inputs.vision_position_ids,
                inputs.position_interpolation_indices,
                inputs.position_interpolation_weights,
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
            if deepstack.shape[0]:
                raise ValueError("Jina v5 does not use Qwen3-VL DeepStack features")
            hidden = _replace_tokens(
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
            pool_offsets = (
                jnp.arange(batch_size, dtype=jnp.int32)
                * chunk_count
                * self.config.audio.window_size
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
            hidden = _replace_tokens(
                hidden,
                audio.astype(hidden.dtype),
                (inputs.audio_token_indices + sequence_offsets[:, None]).reshape(-1),
                inputs.audio_token_valid.reshape(-1),
            )

        position_ids = inputs.position_ids
        if position_ids is None:
            batch, sequence = inputs.input_ids.shape
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence, dtype=jnp.int32)[None, None],
                (3, batch, sequence),
            )
        return self.text(
            inputs.input_ids,
            inputs.attention_mask,
            position_ids,
            inputs_embeds=hidden,
            bidirectional=self.config.text_attention == "bidirectional",
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: JinaV5OmniBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        positions = jnp.arange(inputs.attention_mask.shape[-1], dtype=jnp.int32)
        last = jnp.max(
            jnp.where(inputs.attention_mask.astype(bool), positions, -1), axis=-1
        )
        pooled = hidden[jnp.arange(hidden.shape[0]), last]
        return l2_normalize(pooled)


__all__ = ["JinaV5OmniBatch", "JinaV5OmniEncoder"]
