"""Native Equinox LLaVA-NeXT representation model."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Modality, Route
from representax.models.clip import CLIPVisionTower
from representax.models.components import (
    AttentionImplementation,
    Linear,
    embedding_lookup,
    l2_normalize,
)
from representax.models.decoder import RotaryDecoderTower
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import LlavaNextConfig


class LlavaNextBatch(eqx.Module):
    """Fixed-shape text and host-planned any-resolution image arrays."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    pixel_values: Float[Array, "image tile channel height width"] | None = None
    tile_valid: Bool[Array, "image tile"] | None = None
    pack_indices: Int[Array, " visual"] | None = None
    pack_valid: Bool[Array, " visual"] | None = None
    visual_token_indices: Int[Array, " visual"] | None = None

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        vision = (
            self.tile_valid,
            self.pack_indices,
            self.pack_valid,
            self.visual_token_indices,
        )
        if self.pixel_values is None and any(value is not None for value in vision):
            raise ValueError("vision layout arrays require pixel_values")
        if self.pixel_values is not None and any(value is None for value in vision):
            raise ValueError("pixel_values require the complete vision layout")
        if self.pixel_values is not None:
            if self.pixel_values.ndim != 5:
                raise ValueError(
                    "pixel_values must have shape [image, tile, channel, height, width]"
                )
            assert self.tile_valid is not None
            if self.tile_valid.shape != self.pixel_values.shape[:2]:
                raise ValueError("tile_valid must align with image and tile axes")
            assert self.pack_indices is not None
            assert self.pack_valid is not None
            assert self.visual_token_indices is not None
            if not (
                self.pack_indices.shape
                == self.pack_valid.shape
                == self.visual_token_indices.shape
            ):
                raise ValueError("vision packing arrays must align")


class LlavaNextProjector(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: LlavaNextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> LlavaNextProjector:
        input_key, output_key = jax.random.split(key)
        return cls(
            input=Linear.init(
                config.vision.hidden_size,
                config.text.hidden_size,
                key=input_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=config.projector_bias,
            ),
            output=Linear.init(
                config.text.hidden_size,
                config.text.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=config.projector_bias,
            ),
        )

    def __call__(self, hidden: Float[Array, "*batch vision"]) -> jax.Array:
        return self.output(jax.nn.gelu(self.input(hidden), approximate=False))


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


class LlavaNextEncoder(eqx.Module):
    """One CLIP-plus-rotary-decoder family for BGE and E5-V retrieval."""

    text: RotaryDecoderTower
    vision: CLIPVisionTower
    projector: LlavaNextProjector
    image_newline: Float[Array, " hidden"]
    metadata: EncoderMetadata
    config: LlavaNextConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    @classmethod
    def load_from_hf(cls, model_name_or_path, **options):
        from .loading import load_llava_next

        return load_llava_next(model_name_or_path, **options)

    @classmethod
    def init(
        cls,
        config: LlavaNextConfig,
        *,
        key: PRNGKeyArray,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        rematerialization: RematerializationPolicy = "full",
        model_id: str = "representax/llava-next",
        revision: str = "random-init",
    ) -> LlavaNextEncoder:
        text_key, vision_key, projector_key, newline_key = jax.random.split(key, 4)
        return cls(
            text=RotaryDecoderTower.init(
                config.text, key=text_key, dtype=parameter_dtype
            ),
            vision=CLIPVisionTower.init(
                config.vision, key=vision_key, dtype=parameter_dtype
            ),
            projector=LlavaNextProjector.init(
                config, key=projector_key, dtype=parameter_dtype
            ),
            image_newline=(config.text.hidden_size**-0.5)
            * jax.random.normal(
                newline_key, (config.text.hidden_size,), dtype=parameter_dtype
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            rematerialization=rematerialization,
        )

    def hidden_states(
        self,
        inputs: LlavaNextBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        compute_dtype = active_compute_dtype(self.compute_dtype)
        embedded = embedding_lookup(self.text.token_embedding, inputs.input_ids).astype(
            compute_dtype
        )
        if inputs.pixel_values is not None:
            assert inputs.tile_valid is not None
            assert inputs.pack_indices is not None
            assert inputs.pack_valid is not None
            assert inputs.visual_token_indices is not None
            pixels = inputs.pixel_values.reshape((-1, *inputs.pixel_values.shape[2:]))
            tokens = self.vision.token_states(
                pixels,
                config=self.config.vision,
                key=key,
                compute_dtype=compute_dtype,
                implementation=self.attention_implementation,
                rematerialization=self.rematerialization,
                layer_count=self.config.selected_vision_layer_count,
            )
            if self.config.vision_feature_select_strategy == "default":
                tokens = tokens[:, 1:]
            projected = self.projector(tokens).reshape(
                (-1, self.config.text.hidden_size)
            )
            sources = jnp.concatenate(
                (projected, self.image_newline.astype(projected.dtype)[None]), axis=0
            )
            visual = sources[inputs.pack_indices]
            embedded = _inject_visual(
                embedded,
                visual,
                inputs.visual_token_indices,
                inputs.pack_valid,
            )
        return self.text(
            inputs.input_ids,
            inputs.attention_mask,
            inputs_embeds=embedded,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: LlavaNextBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        mask = inputs.attention_mask.astype(bool)
        positions = jnp.arange(mask.shape[1])
        index = jnp.max(jnp.where(mask, positions, -1), axis=1)
        pooled = hidden[jnp.arange(hidden.shape[0]), index]
        return l2_normalize(pooled)


__all__ = ["LlavaNextBatch", "LlavaNextEncoder", "LlavaNextProjector"]
