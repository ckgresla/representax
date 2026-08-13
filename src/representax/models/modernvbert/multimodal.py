"""Native ModernVBERT vision-text fusion and representation encoder."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route

from .config import ModernVBERTConfig
from .model import (
    AttentionImplementation,
    Linear,
    ModernVBERTTextBatch,
    ModernVBERTTextEncoder,
    _l2_normalize,
    _mean_pool,
)
from .vision import SigLIPVisionTower, pixel_shuffle


class ModernVBERTBatch(eqx.Module):
    """Fixed-shape text, image, or fused inputs for ModernVBERT."""

    attention_mask: jax.Array
    input_ids: jax.Array | None = None
    inputs_embeds: jax.Array | None = None
    position_ids: jax.Array | None = None
    pixel_values: jax.Array | None = None
    pixel_attention_mask: jax.Array | None = None
    image_valid: jax.Array | None = None

    def __post_init__(self) -> None:
        text = ModernVBERTTextBatch(
            attention_mask=self.attention_mask,
            input_ids=self.input_ids,
            inputs_embeds=self.inputs_embeds,
            position_ids=self.position_ids,
        )
        del text
        if self.pixel_values is None:
            if self.pixel_attention_mask is not None or self.image_valid is not None:
                raise ValueError("image masks require pixel_values")
            return
        if self.input_ids is None:
            raise ValueError("multimodal inputs require input_ids for image merging")
        if self.pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must have shape [batch, images, channels, H, W]"
            )
        if self.pixel_values.shape[0] != self.attention_mask.shape[0]:
            raise ValueError("pixel_values and text inputs must share a batch")
        batch, images, _, height, width = self.pixel_values.shape
        if self.pixel_attention_mask is not None and (
            self.pixel_attention_mask.shape != (batch, images, height, width)
        ):
            raise ValueError("pixel_attention_mask must align with pixel_values")
        if self.image_valid is not None:
            if self.image_valid.shape != (batch, images):
                raise ValueError("image_valid must have shape [batch, images]")
            if self.image_valid.dtype != jnp.bool_:
                raise TypeError("image_valid must be boolean")

    def text_batch(
        self, *, inputs_embeds: jax.Array | None = None
    ) -> ModernVBERTTextBatch:
        return ModernVBERTTextBatch(
            attention_mask=self.attention_mask,
            input_ids=self.input_ids if inputs_embeds is None else None,
            inputs_embeds=self.inputs_embeds
            if inputs_embeds is None
            else inputs_embeds,
            position_ids=self.position_ids,
        )


def merge_image_features(
    input_ids: jax.Array,
    token_embeddings: jax.Array,
    image_features: jax.Array,
    *,
    image_token_id: int,
    image_valid: jax.Array | None = None,
) -> jax.Array:
    """Scatter record-major image features into image-token positions."""

    if image_features.ndim != 4:
        raise ValueError(
            "image_features must have shape [batch, images, tokens, hidden]"
        )
    batch, images, image_tokens, hidden = image_features.shape
    if input_ids.shape != token_embeddings.shape[:2]:
        raise ValueError("input_ids and token_embeddings must align")
    if image_features.shape[0] != input_ids.shape[0]:
        raise ValueError("image features and token inputs must share a batch")
    if hidden != token_embeddings.shape[-1]:
        raise ValueError("image and token hidden dimensions must match")
    if image_valid is None:
        image_valid = jnp.ones((batch, images), dtype=bool)

    # Transformers compacts real images before vision. Stable sorting produces
    # the same per-record ordering while retaining a static JAX shape.
    order = jnp.argsort(~image_valid, axis=1, stable=True)
    ordered = jnp.take_along_axis(
        image_features,
        order[:, :, None, None],
        axis=1,
    ).reshape(batch, images * image_tokens, hidden)
    valid_count = jnp.sum(image_valid, axis=1) * image_tokens
    safe = jnp.concatenate(
        (ordered, jnp.zeros((batch, 1, hidden), dtype=ordered.dtype)),
        axis=1,
    )
    image_mask = input_ids == image_token_id
    local_index = jnp.maximum(jnp.cumsum(image_mask, axis=1) - 1, 0)
    local_index = jnp.where(
        image_mask & (local_index < valid_count[:, None]),
        local_index,
        images * image_tokens,
    )
    gathered = jnp.take_along_axis(safe, local_index[..., None], axis=1)
    return jnp.where(image_mask[..., None], gathered, token_embeddings)


class ModernVBERTEncoder(eqx.Module):
    """Complete native text-image ModernVBERT representation model."""

    text: ModernVBERTTextEncoder
    vision: SigLIPVisionTower
    connector: Linear
    metadata: EncoderMetadata
    config: ModernVBERTConfig = eqx.field(static=True)
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTConfig,
        *,
        key: jax.Array,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        model_id: str = "representax/modernvbert",
        revision: str = "random-init",
    ) -> ModernVBERTEncoder:
        text_key, vision_key, connector_key = jax.random.split(key, 3)
        text = ModernVBERTTextEncoder.init(
            config.text,
            key=text_key,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
            model_id=model_id,
            revision=revision,
        )
        connector_input = config.vision.hidden_size * config.pixel_shuffle_factor**2
        return cls(
            text=text,
            vision=SigLIPVisionTower.init(
                config.vision,
                key=vision_key,
                dtype=parameter_dtype,
            ),
            connector=Linear.init(
                connector_input,
                config.text.hidden_size,
                key=connector_key,
                scale=config.text.initializer_range,
                dtype=parameter_dtype,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FUSED}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def image_features(self, pixel_values: jax.Array) -> jax.Array:
        """Encode record-major image slots into connector token sequences."""

        if pixel_values.ndim != 5:
            raise ValueError(
                "pixel_values must have shape [batch, images, channels, H, W]"
            )
        batch, images = pixel_values.shape[:2]
        flat_pixels = pixel_values.reshape((-1, *pixel_values.shape[-3:]))
        hidden = self.vision(
            flat_pixels,
            compute_dtype=self.compute_dtype,
            attention_implementation=self.attention_implementation,
        )
        features = self.connector(
            pixel_shuffle(hidden, factor=self.config.pixel_shuffle_factor)
        )
        return features.reshape(
            batch,
            images,
            self.config.image_sequence_length,
            self.config.text.hidden_size,
        )

    def hidden_states(self, inputs: ModernVBERTBatch) -> jax.Array:
        if not isinstance(inputs, ModernVBERTBatch):
            raise TypeError("ModernVBERT inputs must be ModernVBERTBatch")
        if inputs.pixel_values is None:
            return self.text.hidden_states(inputs.text_batch())

        features = self.image_features(inputs.pixel_values)
        embeddings = self.text.tower.token_embeddings(inputs.input_ids)
        fused = merge_image_features(
            inputs.input_ids,
            embeddings,
            features,
            image_token_id=self.config.image_token_id,
            image_valid=inputs.image_valid,
        )
        # Transformers 5.3 computes pixel_attention_mask but its SigLIP path
        # does not consume it. Retain the processor tensor without inventing
        # different model semantics.
        return self.text.hidden_states(inputs.text_batch(inputs_embeds=fused))

    def encode(
        self,
        inputs: ModernVBERTBatch,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del route, key
        hidden = self.hidden_states(inputs)
        return _l2_normalize(_mean_pool(hidden, inputs.attention_mask))
