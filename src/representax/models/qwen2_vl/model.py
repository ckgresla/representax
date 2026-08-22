"""Native Qwen2-VL and Qwen2.5-VL embedding and reranking models."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.components import (
    AttentionImplementation,
    Linear,
    embedding_lookup,
    l2_normalize,
)
from representax.models.qwen2_5_omni.text import Qwen2_5OmniTextTower
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import Qwen2VLConfig
from .vision import Qwen2VLVisionTower


class Qwen2VLBatch(eqx.Module):
    """Fixed-shape model-ready text/image/video arrays."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    position_ids: Int[Array, "position batch sequence"]
    pixel_values: Float[Array, "patch pixel"] | None = None
    patch_valid: Bool[Array, " patch"] | None = None
    vision_full_segment_ids: Int[Array, " patch"] | None = None
    vision_window_segment_ids: Int[Array, " patch"] | None = None
    vision_position_ids: Int[Array, "patch coordinate"] | None = None
    reverse_merged_indices: Int[Array, " merged"] | None = None
    visual_token_indices: Int[Array, " visual"] | None = None
    visual_token_valid: Bool[Array, " visual"] | None = None

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        if self.position_ids.shape != (3, *self.input_ids.shape):
            raise ValueError("position_ids must have shape [3, batch, sequence]")
        vision = (
            self.patch_valid,
            self.vision_full_segment_ids,
            self.vision_window_segment_ids,
            self.vision_position_ids,
            self.reverse_merged_indices,
            self.visual_token_indices,
            self.visual_token_valid,
        )
        if self.pixel_values is None and any(value is not None for value in vision):
            raise ValueError("vision layout arrays require pixel_values")
        if self.pixel_values is not None and any(value is None for value in vision):
            raise ValueError("pixel_values require the complete vision layout")


def _inject_visual(
    hidden: Float[Array, "batch sequence hidden"],
    visual: Float[Array, "visual hidden"],
    indices: Int[Array, " visual"],
    valid: Bool[Array, " visual"],
) -> Float[Array, "batch sequence hidden"]:
    flattened = hidden.reshape((-1, hidden.shape[-1]))
    current = flattened[indices]
    delta = jnp.where(valid[:, None], visual - current, 0)
    return flattened.at[indices].add(delta).reshape(hidden.shape)


class Qwen2VLEncoder(eqx.Module):
    """One native backbone for Qwen2/Qwen2.5 dense representation models."""

    text: Qwen2_5OmniTextTower
    vision: Qwen2VLVisionTower
    metadata: EncoderMetadata
    config: Qwen2VLConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)
    text_attention: Literal["causal", "bidirectional"] = eqx.field(
        static=True, default="causal"
    )
    pooling: Literal["first", "last", "mean"] = eqx.field(static=True, default="last")

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_qwen2_vl_embedding

        return load_qwen2_vl_embedding(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: Qwen2VLConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/qwen2-vl",
        revision: str = "random-init",
        text_attention: Literal["causal", "bidirectional"] = "causal",
        pooling: Literal["first", "last", "mean"] = "last",
    ) -> Qwen2VLEncoder:
        text_key, vision_key = jax.random.split(key)
        return cls(
            text=Qwen2_5OmniTextTower.init(
                config.text, key=text_key, dtype=parameter_dtype
            ),
            vision=Qwen2VLVisionTower.init(
                config.vision, key=vision_key, dtype=parameter_dtype
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
            text_attention=text_attention,
            pooling=pooling,
        )

    def hidden_states(
        self,
        inputs: Qwen2VLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        compute_dtype = active_compute_dtype(self.compute_dtype)
        embedded = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
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
            embedded = _inject_visual(
                embedded,
                visual,
                inputs.visual_token_indices,
                inputs.visual_token_valid,
            )
        return self.text(
            inputs.input_ids,
            inputs.attention_mask,
            inputs.position_ids,
            inputs_embeds=embedded,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
            causal_attention=self.text_attention == "causal",
        )

    def encode(
        self,
        inputs: Qwen2VLBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        mask = inputs.attention_mask.astype(bool)
        if self.pooling == "first":
            index = jnp.argmax(mask, axis=1)
            pooled = hidden[jnp.arange(hidden.shape[0]), index]
        elif self.pooling == "last":
            positions = jnp.arange(mask.shape[1])
            index = jnp.max(jnp.where(mask, positions, -1), axis=1)
            pooled = hidden[jnp.arange(hidden.shape[0]), index]
        else:
            weights = mask.astype(jnp.float32)
            pooled = jnp.sum(hidden.astype(jnp.float32) * weights[..., None], axis=1)
            pooled = pooled / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1)
        return l2_normalize(pooled)


class Qwen2VLReranker(eqx.Module):
    """Qwen2-VL pair scorer used by Jina reranker m0."""

    model: Qwen2VLEncoder
    hidden: Linear
    output: Linear
    score_bias: float = eqx.field(static=True, default=2.65)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_qwen2_vl_reranker

        return load_qwen2_vl_reranker(model_name_or_path, **options)

    def score(
        self,
        inputs: Qwen2VLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        states = self.model.hidden_states(inputs, key=key)
        logits = self.output(jax.nn.relu(self.hidden(states[:, -1])))[:, 0]
        return jax.nn.sigmoid(logits - self.score_bias)


__all__ = ["Qwen2VLBatch", "Qwen2VLEncoder", "Qwen2VLReranker"]
