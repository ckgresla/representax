"""Native Qwen2.5-Omni audio convolution and packed transformer tower."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    dot_product_attention,
    rematerialize,
)
from representax.planning import RematerializationPolicy

from .config import Qwen2_5OmniAudioConfig


class Conv1D(eqx.Module):
    """NWC convolution retaining the Hugging Face OIW checkpoint layout."""

    weight: Float[Array, "output input kernel"]
    bias: Float[Array, " output"]
    stride: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        input_size: int,
        output_size: int,
        *,
        kernel_size: int,
        stride: int,
        key: PRNGKeyArray,
        scale: float,
        dtype: jnp.dtype,
    ) -> Conv1D:
        return cls(
            weight=scale
            * jax.random.normal(
                key,
                (output_size, input_size, kernel_size),
                dtype=dtype,
            ),
            bias=jnp.zeros((output_size,), dtype=dtype),
            stride=stride,
        )

    def __call__(
        self,
        value: Float[Array, "batch sequence input"],
    ) -> Float[Array, "batch output_sequence output"]:
        kernel = jnp.transpose(self.weight, (2, 1, 0))
        output = jax.lax.conv_general_dilated(
            value,
            kernel,
            window_strides=(self.stride,),
            padding=((1, 1),),
            dimension_numbers=("NWC", "WIO", "NWC"),
        )
        return output + self.bias


def sinusoidal_position_embedding(
    length: int,
    channels: int,
) -> Float[Array, "length channels"]:
    """Match the upstream fixed sine/cosine audio position table."""

    if channels % 2:
        raise ValueError("audio position channels must be even")
    increment = math.log(10_000.0) / (channels // 2 - 1)
    inverse_timescales = jnp.exp(
        -increment * jnp.arange(channels // 2, dtype=jnp.float32)
    )
    scaled_time = (
        jnp.arange(length, dtype=jnp.float32)[:, None] * inverse_timescales[None, :]
    )
    return jnp.concatenate((jnp.sin(scaled_time), jnp.cos(scaled_time)), axis=1)


class Qwen2_5OmniAudioAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniAudioAttention:
        keys = jax.random.split(key, 4)
        return cls(
            query=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            key=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[1],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            value=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[3],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "sequence hidden"],
        attention_mask: Bool[Array, "one one sequence sequence"],
        *,
        config: Qwen2_5OmniAudioConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "sequence hidden"]:
        sequence = hidden.shape[0]
        shape = (sequence, config.num_attention_heads, config.head_dimension)
        query = self.query(hidden).reshape(shape)
        key = self.key(hidden).reshape(shape)
        value = self.value(hidden).reshape(shape)
        attended = dot_product_attention(
            query[None],
            key[None],
            value[None],
            attention_mask=attention_mask,
            implementation=implementation,
        )[0]
        return self.output(attended.reshape((sequence, config.hidden_size)))


class Qwen2_5OmniAudioLayer(eqx.Module):
    attention_norm: LayerNorm
    attention: Qwen2_5OmniAudioAttention
    mlp_norm: LayerNorm
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniAudioLayer:
        attention_key, up_key, down_key = jax.random.split(key, 3)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=Qwen2_5OmniAudioAttention.init(
                config,
                key=attention_key,
                dtype=dtype,
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            up=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=up_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            down=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=down_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "sequence hidden"],
        attention_mask: Bool[Array, "one one sequence sequence"],
        *,
        config: Qwen2_5OmniAudioConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "sequence hidden"]:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            attention_mask,
            config=config,
            implementation=implementation,
        )
        return hidden + self.down(
            jax.nn.gelu(self.up(self.mlp_norm(hidden)), approximate=False)
        )


class Qwen2_5OmniAudioLayerStack(eqx.Module):
    layers: Qwen2_5OmniAudioLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls,
        layers: tuple[Qwen2_5OmniAudioLayer, ...],
    ) -> Qwen2_5OmniAudioLayerStack:
        if not layers:
            raise ValueError("Qwen2.5-Omni requires at least one audio layer")
        return cls(
            layers=jax.tree.map(lambda *values: jnp.stack(values), *layers),
            depth=len(layers),
        )


class Qwen2_5OmniAudioTower(eqx.Module):
    conv1: Conv1D
    conv2: Conv1D
    layers: Qwen2_5OmniAudioLayerStack
    final_norm: LayerNorm
    projection: Linear
    bos_eos_embedding: Float[Array, "two output"]
    config: Qwen2_5OmniAudioConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniAudioTower:
        keys = jax.random.split(key, config.num_hidden_layers + 4)
        return cls(
            conv1=Conv1D.init(
                config.num_mel_bins,
                config.hidden_size,
                kernel_size=3,
                stride=1,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            conv2=Conv1D.init(
                config.hidden_size,
                config.hidden_size,
                kernel_size=3,
                stride=2,
                key=keys[1],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            layers=Qwen2_5OmniAudioLayerStack.from_layers(
                tuple(
                    Qwen2_5OmniAudioLayer.init(
                        config,
                        key=keys[index + 2],
                        dtype=dtype,
                    )
                    for index in range(config.num_hidden_layers)
                )
            ),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            projection=Linear.init(
                config.hidden_size,
                config.output_size,
                key=keys[-2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            bos_eos_embedding=config.initializer_range
            * jax.random.normal(
                keys[-1],
                (2, config.output_size),
                dtype=dtype,
            ),
            config=config,
        )

    def __call__(
        self,
        input_features: Float[Array, "chunk mel feature"],
        feature_valid: Bool[Array, "chunk feature"],
        after_cnn_valid: Bool[Array, "chunk cnn_sequence"],
        pool_indices: Int[Array, "token pair"],
        pool_valid: Bool[Array, " token"],
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "token output"]:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [chunk, mel, feature]")
        chunks, mel_bins, feature_count = input_features.shape
        if mel_bins != self.config.num_mel_bins:
            raise ValueError("input_features has the wrong mel dimension")
        if feature_valid.shape != (chunks, feature_count):
            raise ValueError("feature_valid must align with input_features")
        if feature_count > 2 * self.config.window_size:
            raise ValueError("audio chunks exceed the configured window")

        hidden = jnp.swapaxes(input_features, 1, 2).astype(compute_dtype)
        hidden = jax.nn.gelu(self.conv1(hidden), approximate=False)
        hidden = hidden * feature_valid[..., None]
        hidden = jax.nn.gelu(self.conv2(hidden), approximate=False)
        cnn_sequence = hidden.shape[1]
        if after_cnn_valid.shape != (chunks, cnn_sequence):
            raise ValueError("after_cnn_valid must align with convolution output")
        positions = sinusoidal_position_embedding(
            cnn_sequence,
            self.config.hidden_size,
        ).astype(hidden.dtype)
        hidden = hidden + positions[None]
        hidden = hidden.reshape((-1, self.config.hidden_size))
        valid = after_cnn_valid.reshape(-1)
        chunk_ids = jnp.repeat(jnp.arange(chunks), cnn_sequence)
        attention_mask = (
            (chunk_ids[:, None] == chunk_ids[None, :]) & valid[:, None] & valid[None, :]
        )[None, None]

        def apply_layer(carry, layer):
            output = layer(
                carry,
                attention_mask,
                config=self.config,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            self.layers.layers,
        )
        if pool_indices.ndim != 2 or pool_indices.shape[1] != 2:
            raise ValueError("pool_indices must have shape [token, 2]")
        if pool_valid.shape != (pool_indices.shape[0],):
            raise ValueError("pool_valid must align with pool_indices")
        pooled = jnp.mean(hidden[pool_indices], axis=1)
        projected = self.projection(self.final_norm(pooled))
        return jnp.where(pool_valid[:, None], projected, 0)


__all__ = [
    "Conv1D",
    "Qwen2_5OmniAudioAttention",
    "Qwen2_5OmniAudioLayer",
    "Qwen2_5OmniAudioLayerStack",
    "Qwen2_5OmniAudioTower",
    "sinusoidal_position_embedding",
]
