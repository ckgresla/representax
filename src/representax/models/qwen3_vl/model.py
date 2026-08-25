"""Complete native Qwen3-VL representation and reranking models."""

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
    l2_normalize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import Qwen3VLConfig
from .text import Qwen3VLTextTower
from .vision import Qwen3VLVisionTower


class Qwen3VLBatch(eqx.Module):
    """Fixed-shape native inputs after model-associated host processing."""

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

    @property
    def batch_size(self) -> int:
        """Return logical samples independently of packed patch dimensions."""

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
        visual_values = (
            self.patch_valid,
            self.vision_segment_ids,
            self.vision_position_ids,
            self.position_interpolation_indices,
            self.position_interpolation_weights,
            self.visual_token_indices,
            self.visual_token_valid,
        )
        if self.pixel_values is None:
            if any(value is not None for value in visual_values):
                raise ValueError("vision layout arrays require pixel_values")
        elif any(value is None for value in visual_values):
            raise ValueError("pixel_values require the complete vision layout")


def replace_visual_tokens(
    hidden: Float[Array, "batch sequence hidden"],
    visual: Float[Array, "visual hidden"],
    indices: Int[Array, " visual"],
    valid: Bool[Array, " visual"],
) -> Float[Array, "batch sequence hidden"]:
    """Replace valid flattened token positions without padded-index collisions."""

    flattened = hidden.reshape((-1, hidden.shape[-1]))
    current = flattened[indices]
    delta = jnp.where(valid[:, None], visual - current, 0)
    return flattened.at[indices].add(delta).reshape(hidden.shape)


def last_valid_token_indices(
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
) -> Int[Array, " batch"]:
    """Return the final valid token position for left- or right-padded rows."""

    positions = jnp.arange(attention_mask.shape[-1], dtype=jnp.int32)
    return jnp.max(jnp.where(attention_mask.astype(bool), positions, -1), axis=-1)


class Qwen3VLEncoder(eqx.Module):
    """Qwen3-VL backbone with last-token normalized representation pooling."""

    text: Qwen3VLTextTower
    vision: Qwen3VLVisionTower
    metadata: EncoderMetadata
    config: Qwen3VLConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load native weights and their model-associated processor once."""

        from .loading import load_qwen3_vl_embedding

        return load_qwen3_vl_embedding(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: Qwen3VLConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/qwen3-vl",
        revision: str = "random-init",
    ) -> Qwen3VLEncoder:
        text_key, vision_key = jax.random.split(key)
        return cls(
            text=Qwen3VLTextTower.init(
                config.text,
                key=text_key,
                dtype=parameter_dtype,
            ),
            vision=Qwen3VLVisionTower.init(
                config.vision,
                key=vision_key,
                dtype=parameter_dtype,
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
        )

    def hidden_states(
        self,
        inputs: Qwen3VLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, Qwen3VLBatch):
            raise TypeError("Qwen3-VL inputs must be Qwen3VLBatch")
        compute_dtype = active_compute_dtype(self.compute_dtype)
        hidden = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
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
            deepstack_visual=deepstack,
            visual_token_indices=inputs.visual_token_indices,
            visual_token_valid=inputs.visual_token_valid,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: Qwen3VLBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        last = last_valid_token_indices(inputs.attention_mask)
        pooled = hidden[jnp.arange(hidden.shape[0]), last]
        return l2_normalize(pooled)


class Qwen3VLReranker(eqx.Module):
    """Binary relevant/not-relevant score using the checkpoint's tied rows."""

    model: Qwen3VLEncoder
    true_token_id: int = eqx.field(static=True, default=9693)
    false_token_id: int = eqx.field(static=True, default=2152)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        """Load native reranking weights and their paired-input processor once."""

        from .loading import load_qwen3_vl_reranker

        return load_qwen3_vl_reranker(model_name_or_path, **options)

    def logits(
        self,
        inputs: Qwen3VLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        hidden = self.model.hidden_states(inputs, key=key)
        last = last_valid_token_indices(inputs.attention_mask)
        pooled = hidden[jnp.arange(hidden.shape[0]), last].astype(jnp.float32)
        direction = (
            self.model.text.token_embedding[self.true_token_id]
            - self.model.text.token_embedding[self.false_token_id]
        ).astype(jnp.float32)
        return pooled @ direction

    def score(
        self,
        inputs: Qwen3VLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " batch"]:
        """Return checkpoint-configured inference probabilities."""

        return jax.nn.sigmoid(self.logits(inputs, key=key))


__all__ = [
    "Qwen3VLBatch",
    "Qwen3VLEncoder",
    "Qwen3VLReranker",
    "last_valid_token_indices",
    "replace_visual_tokens",
]
