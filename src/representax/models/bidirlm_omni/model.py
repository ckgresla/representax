"""Complete native BidirLM Omni representation model."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    embedding_lookup,
    mean_pool,
)
from representax.models.qwen3_vl.model import replace_visual_tokens
from representax.models.qwen3_vl.text import Qwen3VLTextTower
from representax.models.qwen3_vl.vision import Qwen3VLVisionTower
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .audio import BidirLMOmniAudioTower
from .config import BidirLMOmniConfig


class BidirLMOmniBatch(eqx.Module):
    """Finite native text, vision, video, and audio inputs."""

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
    input_features: Float[Array, "chunk mel frame"] | None = None
    audio_chunk_lengths: Int[Array, " chunk"] | None = None
    audio_output_valid: Bool[Array, " audio_sequence"] | None = None
    audio_segment_ids: Int[Array, " audio_sequence"] | None = None
    audio_feature_indices: Int[Array, " audio_token"] | None = None
    audio_token_indices: Int[Array, " audio_token"] | None = None
    audio_token_valid: Bool[Array, " audio_token"] | None = None

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

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
            self.audio_chunk_lengths,
            self.audio_output_valid,
            self.audio_segment_ids,
            self.audio_feature_indices,
            self.audio_token_indices,
            self.audio_token_valid,
        )
        if self.input_features is None:
            if any(value is not None for value in audio):
                raise ValueError("audio layout arrays require input_features")
        elif any(value is None for value in audio):
            raise ValueError("input_features require the complete audio layout")


class BidirLMOmniEncoder(eqx.Module):
    """Bidirectional shared-space encoder for text, image, video, and audio."""

    text: Qwen3VLTextTower
    vision: Qwen3VLVisionTower
    audio: BidirLMOmniAudioTower
    metadata: EncoderMetadata
    config: BidirLMOmniConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load native weights and their model-associated processor once."""

        from .loading import load_bidirlm_omni

        return load_bidirlm_omni(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: BidirLMOmniConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/bidirlm-omni",
        revision: str = "random-init",
    ) -> BidirLMOmniEncoder:
        text_key, vision_key, audio_key = jax.random.split(key, 3)
        return cls(
            text=Qwen3VLTextTower.init(
                config.text, key=text_key, dtype=parameter_dtype
            ),
            vision=Qwen3VLVisionTower.init(
                config.vision, key=vision_key, dtype=parameter_dtype
            ),
            audio=BidirLMOmniAudioTower.init(
                config.audio, key=audio_key, dtype=parameter_dtype
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset(
                    {Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}
                ),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def hidden_states(
        self,
        inputs: BidirLMOmniBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, BidirLMOmniBatch):
            raise TypeError("BidirLM Omni inputs must be BidirLMOmniBatch")
        compute_dtype = active_compute_dtype(self.compute_dtype)
        hidden = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
        )
        if inputs.input_features is not None:
            assert inputs.audio_output_valid is not None
            assert inputs.audio_chunk_lengths is not None
            assert inputs.audio_segment_ids is not None
            assert inputs.audio_feature_indices is not None
            assert inputs.audio_token_indices is not None
            assert inputs.audio_token_valid is not None
            audio = self.audio(
                inputs.input_features,
                inputs.audio_chunk_lengths,
                inputs.audio_output_valid,
                inputs.audio_segment_ids,
                compute_dtype=compute_dtype,
                attention_implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
            )
            audio = audio[inputs.audio_feature_indices]
            hidden = replace_visual_tokens(
                hidden,
                audio.astype(hidden.dtype),
                inputs.audio_token_indices,
                inputs.audio_token_valid,
            )
        deepstack = None
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
            hidden = replace_visual_tokens(
                hidden,
                visual.astype(hidden.dtype),
                inputs.visual_token_indices,
                inputs.visual_token_valid,
            )
        position_ids = inputs.position_ids
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                (jnp.cumsum(inputs.attention_mask, axis=-1) - 1)[None],
                (3, *inputs.input_ids.shape),
            )
        return self.text(
            inputs.input_ids,
            inputs.attention_mask,
            position_ids,
            inputs_embeds=hidden,
            deepstack_visual=deepstack,
            visual_token_indices=inputs.visual_token_indices,
            visual_token_valid=inputs.visual_token_valid,
            bidirectional=True,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: BidirLMOmniBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        return mean_pool(self.hidden_states(inputs, key=key), inputs.attention_mask)


__all__ = ["BidirLMOmniBatch", "BidirLMOmniEncoder"]
