"""Equinox implementation of the ModernVBERT ModernBERT text tower.

The module graph is written from the published Transformers architecture and
validated against a pinned Transformers oracle.  PyTorch is not part of the
execution path.
"""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route

from .config import AttentionType, ModernVBERTTextConfig

AttentionImplementation = Literal["xla", "cudnn"]


@jax.custom_vjp
def _embedding_lookup(table: jax.Array, indices: jax.Array) -> jax.Array:
    """Gather embeddings while completing each table-gradient scatter."""

    return table[indices]


def _embedding_lookup_forward(
    table: jax.Array,
    indices: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    return table[indices], (table, indices)


def _embedding_lookup_backward(
    residual: tuple[jax.Array, jax.Array],
    cotangent: jax.Array,
) -> tuple[jax.Array, None]:
    table, indices = residual
    gradient = jnp.zeros_like(table).at[indices].add(cotangent)
    return jax.lax.optimization_barrier(gradient), None


_embedding_lookup.defvjp(_embedding_lookup_forward, _embedding_lookup_backward)


class ModernVBERTTextBatch(eqx.Module):
    """Token IDs or input embeddings plus their valid-token mask."""

    attention_mask: jax.Array
    input_ids: jax.Array | None = None
    inputs_embeds: jax.Array | None = None
    position_ids: jax.Array | None = None

    def __post_init__(self) -> None:
        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ValueError("specify exactly one of input_ids or inputs_embeds")
        if self.attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [batch, sequence]")
        if self.input_ids is not None:
            if self.input_ids.ndim != 2:
                raise ValueError("input_ids must have shape [batch, sequence]")
            if not jnp.issubdtype(self.input_ids.dtype, jnp.integer):
                raise TypeError("input_ids must have an integer dtype")
        elif self.inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, sequence, hidden]")
        token_shape = (
            self.input_ids.shape
            if self.input_ids is not None
            else self.inputs_embeds.shape[:2]
        )
        if token_shape != self.attention_mask.shape:
            raise ValueError("token inputs and attention_mask must align")
        if self.position_ids is not None:
            if self.position_ids.ndim != 2:
                raise ValueError("position_ids must have shape [batch, sequence]")
            if self.position_ids.shape[0] not in {
                1,
                self.attention_mask.shape[0],
            }:
                raise ValueError(
                    "position_ids batch must be one or match attention_mask"
                )
            if self.position_ids.shape[1] != self.attention_mask.shape[1]:
                raise ValueError("position_ids and attention_mask must align")
            if not jnp.issubdtype(self.position_ids.dtype, jnp.integer):
                raise TypeError("position_ids must have an integer dtype")


class Linear(eqx.Module):
    """Batched linear projection with Hugging Face weight orientation."""

    weight: jax.Array
    bias: jax.Array | None = None

    @classmethod
    def init(
        cls,
        input_size: int,
        output_size: int,
        *,
        key: jax.Array,
        scale: float,
        dtype: jnp.dtype,
        bias: bool = False,
    ) -> Linear:
        weight = scale * jax.random.normal(key, (output_size, input_size), dtype=dtype)
        return cls(
            weight=weight,
            bias=jnp.zeros((output_size,), dtype=dtype) if bias else None,
        )

    def __call__(self, value: jax.Array) -> jax.Array:
        output = value @ self.weight.T
        return output if self.bias is None else output + self.bias


class LayerNorm(eqx.Module):
    """ModernBERT LayerNorm with FP32 statistics."""

    weight: jax.Array
    bias: jax.Array | None
    epsilon: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        size: int,
        *,
        epsilon: float,
        dtype: jnp.dtype,
        bias: bool = False,
    ) -> LayerNorm:
        return cls(
            weight=jnp.ones((size,), dtype=dtype),
            bias=jnp.zeros((size,), dtype=dtype) if bias else None,
            epsilon=epsilon,
        )

    def __call__(self, value: jax.Array) -> jax.Array:
        source_dtype = value.dtype
        value = value.astype(jnp.float32)
        mean = jnp.mean(value, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
        output = (value - mean) * jax.lax.rsqrt(variance + self.epsilon)
        output = output * self.weight.astype(jnp.float32)
        if self.bias is not None:
            output = output + self.bias.astype(jnp.float32)
        return output.astype(source_dtype)


def _rotary_frequencies(
    head_dimension: int,
    theta: float,
    position_ids: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    inverse_frequency = 1.0 / (
        theta ** (jnp.arange(0, head_dimension, 2, dtype=jnp.float32) / head_dimension)
    )
    frequency = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    embedding = jnp.concatenate((frequency, frequency), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


def _rotate_half(value: jax.Array) -> jax.Array:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def _apply_rope(
    value: jax.Array,
    cosine: jax.Array,
    sine: jax.Array,
) -> jax.Array:
    value32 = value.astype(jnp.float32)
    rotated = value32 * cosine + _rotate_half(value32) * sine
    return rotated.astype(value.dtype)


def _scaled_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array,
    *,
    local_radius: int | None,
    implementation: AttentionImplementation,
) -> jax.Array:
    """Run exact full or symmetric local attention through JAX's primitive."""

    local_window = None if local_radius is None else (local_radius, local_radius)
    return jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=attention_mask[:, None, None, :].astype(bool),
        local_window_size=local_window,
        implementation=implementation,
    )


class FusedSelfAttention(eqx.Module):
    qkv: Linear
    output: Linear
    layer_type: AttentionType = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        layer_type: AttentionType,
        *,
        key: jax.Array,
        dtype: jnp.dtype,
    ) -> FusedSelfAttention:
        qkv_key, output_key = jax.random.split(key)
        return cls(
            qkv=Linear.init(
                config.hidden_size,
                3 * config.hidden_size,
                key=qkv_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            output=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            layer_type=layer_type,
        )

    def __call__(
        self,
        hidden: jax.Array,
        *,
        config: ModernVBERTTextConfig,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        implementation: AttentionImplementation,
    ) -> jax.Array:
        batch, sequence, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(
            batch,
            sequence,
            3,
            config.num_attention_heads,
            config.head_dimension,
        )
        query, key, value = (qkv[:, :, index] for index in range(3))
        cosine, sine = _rotary_frequencies(
            config.head_dimension,
            config.rope_theta(self.layer_type),
            position_ids,
        )
        cosine = cosine[:, :, None, :]
        sine = sine[:, :, None, :]
        query = _apply_rope(query, cosine, sine)
        key = _apply_rope(key, cosine, sine)
        local_radius = (
            config.local_attention // 2
            if self.layer_type == "sliding_attention"
            else None
        )
        attended = _scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
            local_radius=local_radius,
            implementation=implementation,
        )
        return self.output(attended.reshape(batch, sequence, config.hidden_size))


class GatedMLP(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: jax.Array,
        dtype: jnp.dtype,
    ) -> GatedMLP:
        input_key, output_key = jax.random.split(key)
        return cls(
            input=Linear.init(
                config.hidden_size,
                2 * config.intermediate_size,
                key=input_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
            output=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
            ),
        )

    def __call__(self, hidden: jax.Array) -> jax.Array:
        value, gate = jnp.split(self.input(hidden), 2, axis=-1)
        return self.output(jax.nn.gelu(value, approximate=False) * gate)


class ModernVBERTTextBlock(eqx.Module):
    attention: FusedSelfAttention
    attention_norm: LayerNorm | None
    mlp_norm: LayerNorm
    mlp: GatedMLP

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        index: int,
        *,
        key: jax.Array,
        dtype: jnp.dtype,
    ) -> ModernVBERTTextBlock:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention=FusedSelfAttention.init(
                config,
                config.layer_types[index],
                key=attention_key,
                dtype=dtype,
            ),
            # ModernBERT's first layer is the intentional pre-norm exception.
            attention_norm=(
                None
                if index == 0
                else LayerNorm.init(
                    config.hidden_size,
                    epsilon=config.norm_epsilon,
                    dtype=dtype,
                )
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            mlp=GatedMLP.init(config, key=mlp_key, dtype=dtype),
        )

    def __call__(
        self,
        hidden: jax.Array,
        *,
        config: ModernVBERTTextConfig,
        attention_mask: jax.Array,
        position_ids: jax.Array,
        implementation: AttentionImplementation,
    ) -> jax.Array:
        attention_input = (
            hidden if self.attention_norm is None else self.attention_norm(hidden)
        )
        hidden = hidden + self.attention(
            attention_input,
            config=config,
            attention_mask=attention_mask,
            position_ids=position_ids,
            implementation=implementation,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class ModernVBERTTextTower(eqx.Module):
    token_embedding: jax.Array
    embedding_norm: LayerNorm
    layers: tuple[ModernVBERTTextBlock, ...]
    final_norm: LayerNorm
    config: ModernVBERTTextConfig = eqx.field(static=True)

    def token_embeddings(self, input_ids: jax.Array) -> jax.Array:
        """Gather token embeddings with a complete table-gradient scatter."""

        return _embedding_lookup(self.token_embedding, input_ids)

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: jax.Array,
        dtype: jnp.dtype = jnp.float32,
    ) -> ModernVBERTTextTower:
        keys = jax.random.split(key, config.num_hidden_layers + 1)
        token_embedding = config.initializer_range * jax.random.normal(
            keys[0], (config.vocab_size, config.hidden_size), dtype=dtype
        )
        layers = tuple(
            ModernVBERTTextBlock.init(
                config,
                index,
                key=keys[index + 1],
                dtype=dtype,
            )
            for index in range(config.num_hidden_layers)
        )
        return cls(
            token_embedding=token_embedding,
            embedding_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            layers=layers,
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
            ),
            config=config,
        )

    def __call__(
        self,
        batch: ModernVBERTTextBatch,
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
    ) -> jax.Array:
        hidden = (
            self.token_embeddings(batch.input_ids)
            if batch.input_ids is not None
            else batch.inputs_embeds
        )
        hidden = hidden.astype(compute_dtype)
        attention_mask = batch.attention_mask.astype(bool)
        batch_size, sequence, _ = hidden.shape
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "input embedding dimension does not match ModernVBERT hidden_size"
            )
        if sequence > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")
        if batch.position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence)[None, :], (batch_size, sequence)
            )
        elif batch.position_ids.shape[0] == 1 and batch_size != 1:
            position_ids = jnp.broadcast_to(batch.position_ids, (batch_size, sequence))
        else:
            position_ids = batch.position_ids
        hidden = self.embedding_norm(hidden)
        for layer in self.layers:
            hidden = layer(
                hidden,
                config=self.config,
                attention_mask=attention_mask,
                position_ids=position_ids,
                implementation=attention_implementation,
            )
        return self.final_norm(hidden)


def _mean_pool(hidden: jax.Array, attention_mask: jax.Array) -> jax.Array:
    hidden = hidden.astype(jnp.float32)
    mask = attention_mask.astype(bool)[..., None]
    total = jnp.sum(jnp.where(mask, hidden, 0.0), axis=1)
    count = jnp.maximum(jnp.sum(mask, axis=1), 1)
    return total / count


def _l2_normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
    return value / jnp.maximum(norm, jnp.asarray(1e-12, value.dtype))


class ModernVBERTTextEncoder(eqx.Module):
    """Text-only Representax encoder backed by a native ModernBERT tower."""

    tower: ModernVBERTTextTower
    metadata: EncoderMetadata
    compute_dtype: Any = eqx.field(static=True)
    attention_implementation: AttentionImplementation = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTTextConfig,
        *,
        key: jax.Array,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
        model_id: str = "representax/modernvbert-text",
        revision: str = "random-init",
    ) -> ModernVBERTTextEncoder:
        return cls(
            tower=ModernVBERTTextTower.init(config, key=key, dtype=parameter_dtype),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def hidden_states(self, inputs: ModernVBERTTextBatch) -> jax.Array:
        if not isinstance(inputs, ModernVBERTTextBatch):
            raise TypeError("ModernVBERT text inputs must be ModernVBERTTextBatch")
        return self.tower(
            inputs,
            compute_dtype=self.compute_dtype,
            attention_implementation=self.attention_implementation,
        )

    def encode(
        self,
        inputs: ModernVBERTTextBatch,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del route, key
        hidden = self.hidden_states(inputs)
        return _l2_normalize(_mean_pool(hidden, inputs.attention_mask))
