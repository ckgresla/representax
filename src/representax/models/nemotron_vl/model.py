"""Native Equinox Llama Nemotron VL embedding and reranking models."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Route
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    embedding_lookup,
    mean_pool,
)
from representax.models.decoder import RotaryDecoderTower
from representax.models.modernvbert.vision import SigLIPVisionTower, pixel_shuffle
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import LlamaNemotronVLConfig


class LlamaNemotronVLBatch(eqx.Module):
    """Fixed-shape text with an optional flat bucket of image tiles."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    pixel_values: Float[Array, "tile channel height width"] | None = None
    visual_token_indices: Int[Array, " visual"] | None = None
    visual_token_valid: Bool[Array, " visual"] | None = None

    @property
    def batch_size(self) -> int:
        """Return examples independently of the flat image-tile bucket."""

        return self.input_ids.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        layout = (self.visual_token_indices, self.visual_token_valid)
        if self.pixel_values is None and any(value is not None for value in layout):
            raise ValueError("visual placement requires pixel_values")
        if self.pixel_values is not None and any(value is None for value in layout):
            raise ValueError("pixel_values require complete visual placement")
        if self.pixel_values is not None:
            if self.pixel_values.ndim != 4:
                raise ValueError(
                    "pixel_values must have shape [tile, channel, height, width]"
                )
            assert self.visual_token_indices is not None
            assert self.visual_token_valid is not None
            if self.visual_token_indices.shape != self.visual_token_valid.shape:
                raise ValueError("visual placement arrays must align")


class NemotronVLProjector(eqx.Module):
    norm: LayerNorm
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: LlamaNemotronVLConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> NemotronVLProjector:
        input_key, output_key = jax.random.split(key)
        expanded = config.vision.hidden_size * config.pixel_shuffle_factor**2
        return cls(
            norm=LayerNorm.init(expanded, epsilon=1e-5, dtype=dtype, bias=True),
            input=Linear.init(
                expanded,
                config.text.hidden_size,
                key=input_key,
                scale=config.text.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                config.text.hidden_size,
                config.text.hidden_size,
                key=output_key,
                scale=config.text.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(self, hidden: Float[Array, "*batch vision"]) -> jax.Array:
        hidden = self.norm(hidden)
        hidden = jax.nn.gelu(self.input(hidden), approximate=False)
        return self.output(hidden)


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


class LlamaNemotronVLBackbone(eqx.Module):
    """Shared native vision-language encoder graph."""

    text: RotaryDecoderTower
    vision: SigLIPVisionTower
    projector: NemotronVLProjector
    config: LlamaNemotronVLConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: LlamaNemotronVLConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
    ) -> LlamaNemotronVLBackbone:
        text_key, vision_key, projector_key = jax.random.split(key, 3)
        return cls(
            text=RotaryDecoderTower.init(
                config.text, key=text_key, dtype=parameter_dtype
            ),
            vision=SigLIPVisionTower.init(
                config.vision, key=vision_key, dtype=parameter_dtype
            ),
            projector=NemotronVLProjector.init(
                config, key=projector_key, dtype=parameter_dtype
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def image_features(
        self,
        pixel_values: Float[Array, "tile channel height width"],
    ) -> Float[Array, "tile image_token hidden"]:
        compute_dtype = active_compute_dtype(self.compute_dtype)
        hidden = self.vision(
            pixel_values,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
        )
        hidden = pixel_shuffle(hidden, factor=self.config.pixel_shuffle_factor)
        return self.projector(hidden)

    def hidden_states(
        self,
        inputs: LlamaNemotronVLBatch,
    ) -> Float[Array, "batch sequence hidden"]:
        if not isinstance(inputs, LlamaNemotronVLBatch):
            raise TypeError("Llama Nemotron VL inputs must be LlamaNemotronVLBatch")
        compute_dtype = active_compute_dtype(self.compute_dtype)
        embedded = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
        )
        if inputs.pixel_values is not None:
            assert inputs.visual_token_indices is not None
            assert inputs.visual_token_valid is not None
            visual = self.image_features(inputs.pixel_values).reshape(
                (-1, self.config.text.hidden_size)
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
            inputs_embeds=embedded,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )


class LlamaNemotronVLEncoder(eqx.Module):
    """Mean-pooled multimodal bi-encoder."""

    model: LlamaNemotronVLBackbone
    metadata: EncoderMetadata

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_nemotron_vl

        model, processor = load_nemotron_vl(model_name_or_path, **options)
        if not isinstance(model, cls):
            raise TypeError("checkpoint is not a Llama Nemotron VL embedder")
        return model, processor

    def encode(
        self,
        inputs: LlamaNemotronVLBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route, key
        hidden = self.model.hidden_states(inputs)
        return mean_pool(hidden.astype(jnp.float32), inputs.attention_mask)


class LlamaNemotronVLReranker(eqx.Module):
    """Mean-pooled multimodal cross-encoder with an FP32 score head."""

    model: LlamaNemotronVLBackbone
    score: Linear
    metadata: EncoderMetadata

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_nemotron_vl

        model, processor = load_nemotron_vl(model_name_or_path, **options)
        if not isinstance(model, cls):
            raise TypeError("checkpoint is not a Llama Nemotron VL reranker")
        return model, processor

    def logits(
        self,
        inputs: LlamaNemotronVLBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch output"]:
        del key
        hidden = self.model.hidden_states(inputs)
        pooled = mean_pool(hidden.astype(jnp.float32), inputs.attention_mask)
        return self.score(pooled) / self.model.config.temperature

    def encode(
        self,
        inputs: LlamaNemotronVLBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch score"]:
        del route
        return self.logits(inputs, key=key)


__all__ = [
    "LlamaNemotronVLBackbone",
    "LlamaNemotronVLBatch",
    "LlamaNemotronVLEncoder",
    "LlamaNemotronVLReranker",
    "NemotronVLProjector",
]
