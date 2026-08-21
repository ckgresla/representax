"""Native Equinox text encoder for Jina Embeddings v5 Omni Small."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Route
from representax.models.components import (
    AttentionImplementation,
    Linear,
    RMSNorm,
    dot_product_attention,
    embedding_lookup,
    l2_normalize,
    rematerialize,
)
from representax.planning import RematerializationPolicy
from representax.precision import active_compute_dtype

from .config import JinaV5TextConfig


class JinaV5TextBatch(eqx.Module):
    """Static-shape token tensors consumed by the native Jina text tower."""

    input_ids: Int[Array, "batch sequence"]
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]
    position_ids: Int[Array, "#batch sequence"] | None = None

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("input_ids and attention_mask must align")
        if not jnp.issubdtype(self.input_ids.dtype, jnp.integer):
            raise TypeError("input_ids must have an integer dtype")
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [batch, sequence]")
            if self.position_ids.shape[0] not in {1, self.input_ids.shape[0]}:
                raise ValueError("position_ids batch must be one or match inputs")
            if self.position_ids.shape[1] != self.input_ids.shape[1]:
                raise ValueError("position_ids and input_ids must align")


def _rotate_half(
    value: Float[Array, "batch sequence heads head"],
) -> Float[Array, "batch sequence heads head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def _rotary_embedding(
    head_dimension: int,
    theta: float,
    position_ids: Int[Array, "batch sequence"],
) -> tuple[
    Float[Array, "batch sequence head"],
    Float[Array, "batch sequence head"],
]:
    inverse_frequency = 1.0 / (
        theta ** (jnp.arange(0, head_dimension, 2, dtype=jnp.float32) / head_dimension)
    )
    frequencies = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    embedding = jnp.concatenate((frequencies, frequencies), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


class JinaV5TextLayer(eqx.Module):
    input_norm: RMSNorm
    post_attention_norm: RMSNorm
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    query_norm: RMSNorm
    key_norm: RMSNorm
    gate: Linear
    up: Linear
    down: Linear

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch sequence"],
        cosine: Float[Array, "batch sequence head"],
        sine: Float[Array, "batch sequence head"],
        *,
        config: JinaV5TextConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape
        compute_dtype = hidden.dtype
        residual = hidden
        normalized = self.input_norm(hidden)
        query = self.query(normalized).reshape(
            batch,
            sequence,
            config.num_attention_heads,
            config.head_dimension,
        )
        key = self.key(normalized).reshape(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dimension,
        )
        value = self.value(normalized).reshape(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dimension,
        )
        query = self.query_norm(query)
        key = self.key_norm(key)
        query = query * cosine[:, :, None] + _rotate_half(query) * sine[:, :, None]
        key = key * cosine[:, :, None] + _rotate_half(key) * sine[:, :, None]
        if config.num_attention_heads != config.num_key_value_heads:
            repeats = config.num_attention_heads // config.num_key_value_heads
            key = jnp.repeat(key, repeats, axis=2)
            value = jnp.repeat(value, repeats, axis=2)
        target = jnp.arange(sequence)[:, None]
        source = jnp.arange(sequence)[None, :]
        allowed = (source <= target)[None, None]
        allowed = allowed & attention_mask[:, None, None, :].astype(bool)
        attended = dot_product_attention(
            query.astype(value.dtype),
            key.astype(value.dtype),
            value,
            attention_mask=allowed,
            implementation=implementation,
        ).reshape(batch, sequence, config.num_attention_heads * config.head_dimension)
        hidden = (residual + self.output(attended)).astype(compute_dtype)
        residual = hidden
        normalized = self.post_attention_norm(hidden)
        activated = jax.nn.silu(self.gate(normalized)) * self.up(normalized)
        return (residual + self.down(activated)).astype(compute_dtype)


class JinaV5TextLayerStack(eqx.Module):
    """Depth-major homogeneous decoder layers executed by one scan."""

    blocks: JinaV5TextLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls,
        layers: tuple[JinaV5TextLayer, ...],
    ) -> JinaV5TextLayerStack:
        if not layers:
            raise ValueError("Jina v5 requires at least one text layer")
        blocks = jax.tree.map(lambda *values: jnp.stack(values), *layers)
        return cls(blocks=blocks, depth=len(layers))

    def layer(self, index: int) -> JinaV5TextLayer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.blocks)


class JinaV5TextTower(eqx.Module):
    token_embedding: Float[Array, "vocabulary hidden"]
    layers: JinaV5TextLayerStack
    final_norm: RMSNorm
    config: JinaV5TextConfig = eqx.field(static=True)

    def __call__(
        self,
        batch: JinaV5TextBatch,
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch sequence hidden"]:
        hidden = embedding_lookup(self.token_embedding, batch.input_ids).astype(
            compute_dtype
        )
        batch_size, sequence = batch.input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        position_ids = batch.position_ids
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence, dtype=jnp.int32)[None],
                (batch_size, sequence),
            )
        else:
            position_ids = jnp.broadcast_to(position_ids, (batch_size, sequence))
        cosine, sine = _rotary_embedding(
            self.config.head_dimension,
            self.config.rope_theta,
            position_ids,
        )
        attention_mask = batch.attention_mask.astype(bool)

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            layer: JinaV5TextLayer,
        ) -> tuple[Float[Array, "batch sequence hidden"], None]:
            output = layer(
                carry,
                attention_mask,
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            self.layers.blocks,
        )
        return self.final_norm(hidden)


class JinaV5TextEncoder(eqx.Module):
    """Last-token, Matryoshka-ready Jina v5 text representation encoder."""

    tower: JinaV5TextTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)
    rematerialization: RematerializationPolicy = eqx.field(static=True)

    def hidden_states(
        self,
        inputs: JinaV5TextBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch sequence hidden"]:
        del key
        if not isinstance(inputs, JinaV5TextBatch):
            raise TypeError("Jina v5 text inputs must be JinaV5TextBatch")
        return self.tower(
            inputs,
            compute_dtype=active_compute_dtype(self.compute_dtype),
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def encode(
        self,
        inputs: JinaV5TextBatch,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden = self.hidden_states(inputs, key=key)
        last_token = jnp.maximum(jnp.sum(inputs.attention_mask, axis=-1) - 1, 0)
        pooled = hidden[
            jnp.arange(hidden.shape[0]),
            last_token,
            : self.metadata.output_dimension,
        ]
        return l2_normalize(pooled)


__all__ = [
    "JinaV5TextBatch",
    "JinaV5TextEncoder",
    "JinaV5TextLayer",
    "JinaV5TextLayerStack",
    "JinaV5TextTower",
]
