"""Native BidirLM Omni convolutional audio encoder."""

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

from .config import BidirLMOmniAudioConfig


class Conv2D(eqx.Module):
    """NCHW/OIHW convolution with explicit PyTorch-compatible same padding."""

    weight: Float[Array, "output input kernel_height kernel_width"]
    bias: Float[Array, " output"]
    stride: int = eqx.field(static=True)

    def __call__(
        self,
        values: Float[Array, "batch input height width"],
    ) -> Float[Array, "batch output new_height new_width"]:
        output = jax.lax.conv_general_dilated(
            values,
            self.weight,
            window_strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return output + self.bias[None, :, None, None]


class BidirLMOmniAudioAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: BidirLMOmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BidirLMOmniAudioAttention:
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
                bias=True,
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
        config: BidirLMOmniAudioConfig,
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


class BidirLMOmniAudioLayer(eqx.Module):
    attention_norm: LayerNorm
    attention: BidirLMOmniAudioAttention
    mlp_norm: LayerNorm
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: BidirLMOmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BidirLMOmniAudioLayer:
        attention_key, up_key, down_key = jax.random.split(key, 3)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=BidirLMOmniAudioAttention.init(
                config, key=attention_key, dtype=dtype
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
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
        config: BidirLMOmniAudioConfig,
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


class BidirLMOmniAudioLayerStack(eqx.Module):
    layers: BidirLMOmniAudioLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls,
        layers: tuple[BidirLMOmniAudioLayer, ...],
    ) -> BidirLMOmniAudioLayerStack:
        if not layers:
            raise ValueError("BidirLM Omni requires at least one audio layer")
        return cls(
            layers=jax.tree.map(lambda *values: jnp.stack(values), *layers),
            depth=len(layers),
        )


def sinusoidal_position_embedding(
    length: int,
    channels: int,
    *,
    maximum_timescale: float = 10_000.0,
) -> Float[Array, "length channels"]:
    """Recompute the reference audio positions in FP32 on every call."""

    increment = math.log(maximum_timescale) / (channels // 2 - 1)
    inverse_timescales = jnp.exp(
        -increment * jnp.arange(channels // 2, dtype=jnp.float32)
    )
    scaled = jnp.arange(length, dtype=jnp.float32)[:, None] * inverse_timescales
    return jnp.concatenate((jnp.sin(scaled), jnp.cos(scaled)), axis=-1)


class BidirLMOmniAudioTower(eqx.Module):
    conv1: Conv2D
    conv2: Conv2D
    conv3: Conv2D
    convolution_projection: Linear
    layers: BidirLMOmniAudioLayerStack
    final_norm: LayerNorm
    projection_up: Linear
    projection_down: Linear
    config: BidirLMOmniAudioConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: BidirLMOmniAudioConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> BidirLMOmniAudioTower:
        keys = jax.random.split(key, config.num_hidden_layers + 7)

        def conv(weight_shape, bias_shape, index):
            return Conv2D(
                weight=config.initializer_range
                * jax.random.normal(keys[index], weight_shape, dtype=dtype),
                bias=jnp.zeros(bias_shape, dtype=dtype),
                stride=2,
            )

        return cls(
            conv1=conv(
                (config.downsample_hidden_size, 1, 3, 3),
                (config.downsample_hidden_size,),
                0,
            ),
            conv2=conv(
                (
                    config.downsample_hidden_size,
                    config.downsample_hidden_size,
                    3,
                    3,
                ),
                (config.downsample_hidden_size,),
                1,
            ),
            conv3=conv(
                (
                    config.downsample_hidden_size,
                    config.downsample_hidden_size,
                    3,
                    3,
                ),
                (config.downsample_hidden_size,),
                2,
            ),
            convolution_projection=Linear.init(
                config.downsample_hidden_size * config.frequency_bins_after_convolution,
                config.hidden_size,
                key=keys[3],
                scale=config.initializer_range,
                dtype=dtype,
                bias=False,
            ),
            layers=BidirLMOmniAudioLayerStack.from_layers(
                tuple(
                    BidirLMOmniAudioLayer.init(
                        config,
                        key=keys[4 + index],
                        dtype=dtype,
                    )
                    for index in range(config.num_hidden_layers)
                )
            ),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            projection_up=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=keys[-3],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            projection_down=Linear.init(
                config.hidden_size,
                config.output_size,
                key=keys[-2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            config=config,
        )

    def __call__(
        self,
        input_features: Float[Array, "chunk mel frame"],
        chunk_lengths: Int[Array, " chunk"],
        output_valid: Bool[Array, " sequence"],
        segment_ids: Int[Array, " sequence"],
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "sequence output"]:
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [chunk, mel, frame]")
        if input_features.shape[1] != self.config.num_mel_bins:
            raise ValueError("input_features has the wrong mel dimension")
        if chunk_lengths.shape != (input_features.shape[0],):
            raise ValueError("chunk_lengths must align with input_features")

        def mask_time(values, lengths):
            valid = jnp.arange(values.shape[-1])[None, :] < lengths[:, None]
            return jnp.where(valid[:, None, None, :], values, 0)

        hidden = input_features[:, None].astype(compute_dtype)
        hidden = jax.nn.gelu(self.conv1(hidden), approximate=False)
        chunk_lengths = (chunk_lengths - 1) // 2 + 1
        hidden = mask_time(hidden, chunk_lengths)
        hidden = jax.nn.gelu(self.conv2(hidden), approximate=False)
        chunk_lengths = (chunk_lengths - 1) // 2 + 1
        hidden = mask_time(hidden, chunk_lengths)
        hidden = jax.nn.gelu(self.conv3(hidden), approximate=False)
        chunk_lengths = (chunk_lengths - 1) // 2 + 1
        hidden = mask_time(hidden, chunk_lengths)
        chunks, channels, frequency, time = hidden.shape
        hidden = jnp.transpose(hidden, (0, 3, 1, 2)).reshape(
            chunks * time, channels * frequency
        )
        hidden = self.convolution_projection(hidden)
        positions = sinusoidal_position_embedding(time, self.config.hidden_size)
        hidden = hidden + jnp.tile(positions, (chunks, 1)).astype(hidden.dtype)
        if output_valid.shape != (hidden.shape[0],):
            raise ValueError("audio output_valid must align with convolved features")
        if segment_ids.shape != output_valid.shape:
            raise ValueError("audio segment_ids must align with convolved features")
        attention_mask = (
            (segment_ids[:, None] == segment_ids[None, :])
            & output_valid[:, None]
            & output_valid[None, :]
        )[None, None]

        def apply_layer(carry, layer):
            return (
                layer(
                    carry,
                    attention_mask,
                    config=self.config,
                    implementation=attention_implementation,
                ),
                None,
            )

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            self.layers.layers,
        )
        hidden = self.final_norm(hidden)
        hidden = jax.nn.gelu(self.projection_up(hidden), approximate=False)
        return self.projection_down(hidden)


__all__ = [
    "BidirLMOmniAudioAttention",
    "BidirLMOmniAudioLayer",
    "BidirLMOmniAudioLayerStack",
    "BidirLMOmniAudioTower",
    "Conv2D",
    "sinusoidal_position_embedding",
]
