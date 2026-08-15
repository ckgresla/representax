"""Native Equinox SigLIP vision path used by ModernVBERT."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .config import ModernVBERTVisionConfig
from .model import AttentionImplementation, LayerNorm, Linear


class PatchEmbedding(eqx.Module):
    """Stride-equals-kernel image patch projection in HF weight layout."""

    weight: Float[Array, "hidden channel patch_height patch_width"]
    bias: Float[Array, " hidden"]
    patch_size: int = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> PatchEmbedding:
        shape = (
            config.hidden_size,
            config.num_channels,
            config.patch_size,
            config.patch_size,
        )
        fan_in = config.num_channels * config.patch_size**2
        weight = jax.random.normal(key, shape, dtype=dtype) / math.sqrt(fan_in)
        return cls(
            weight=weight,
            bias=jnp.zeros((config.hidden_size,), dtype=dtype),
            patch_size=config.patch_size,
        )

    def __call__(
        self,
        pixel_values: Float[Array, "image channel height width"],
    ) -> Float[Array, "image grid_height grid_width hidden"]:
        pixels = jnp.transpose(pixel_values, (0, 2, 3, 1))
        kernel = jnp.transpose(self.weight, (2, 3, 1, 0))
        patches = jax.lax.conv_general_dilated(
            pixels,
            kernel,
            window_strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return patches + self.bias


class SigLIPVisionAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: ModernVBERTVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> SigLIPVisionAttention:
        keys = jax.random.split(key, 4)
        arguments = {
            "input_size": config.hidden_size,
            "output_size": config.hidden_size,
            "scale": config.hidden_size**-0.5,
            "dtype": dtype,
            "bias": True,
        }
        return cls(
            query=Linear.init(key=keys[0], **arguments),
            key=Linear.init(key=keys[1], **arguments),
            value=Linear.init(key=keys[2], **arguments),
            output=Linear.init(key=keys[3], **arguments),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        *,
        config: ModernVBERTVisionConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape

        def project(
            projection: Linear,
        ) -> Float[Array, "batch sequence heads head"]:
            return projection(hidden).reshape(
                batch,
                sequence,
                config.num_attention_heads,
                config.head_dimension,
            )

        attended = jax.nn.dot_product_attention(
            project(self.query),
            project(self.key),
            project(self.value),
            implementation=implementation,
        )
        return self.output(attended.reshape(batch, sequence, config.hidden_size))


class SigLIPVisionMLP(eqx.Module):
    input: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: ModernVBERTVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> SigLIPVisionMLP:
        input_key, output_key = jax.random.split(key)
        return cls(
            input=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=input_key,
                scale=config.hidden_size**-0.5,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=output_key,
                scale=config.intermediate_size**-0.5,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
    ) -> Float[Array, "batch sequence hidden"]:
        return self.output(jax.nn.gelu(self.input(hidden), approximate=True))


class SigLIPVisionLayer(eqx.Module):
    attention_norm: LayerNorm
    attention: SigLIPVisionAttention
    mlp_norm: LayerNorm
    mlp: SigLIPVisionMLP

    @classmethod
    def init(
        cls,
        config: ModernVBERTVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> SigLIPVisionLayer:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=SigLIPVisionAttention.init(
                config,
                key=attention_key,
                dtype=dtype,
            ),
            mlp_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            mlp=SigLIPVisionMLP.init(config, key=mlp_key, dtype=dtype),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        *,
        config: ModernVBERTVisionConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            config=config,
            implementation=implementation,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class SigLIPVisionTower(eqx.Module):
    patch_embedding: PatchEmbedding
    position_embedding: Float[Array, "sequence hidden"]
    layers: tuple[SigLIPVisionLayer, ...]
    final_norm: LayerNorm
    config: ModernVBERTVisionConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: ModernVBERTVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> SigLIPVisionTower:
        keys = jax.random.split(key, config.num_hidden_layers + 2)
        return cls(
            patch_embedding=PatchEmbedding.init(config, key=keys[0], dtype=dtype),
            position_embedding=jax.random.normal(
                keys[1],
                (config.patch_count, config.hidden_size),
                dtype=dtype,
            )
            / math.sqrt(config.hidden_size),
            layers=tuple(
                SigLIPVisionLayer.init(
                    config,
                    key=keys[index + 2],
                    dtype=dtype,
                )
                for index in range(config.num_hidden_layers)
            ),
            final_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            config=config,
        )

    def __call__(
        self,
        pixel_values: Float[Array, "image channel height width"],
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
    ) -> Float[Array, "image sequence hidden"]:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [images, channels, H, W]")
        expected = (
            self.config.num_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if pixel_values.shape[1:] != expected:
            raise ValueError(f"pixel_values must have trailing shape {expected}")
        patches = self.patch_embedding(pixel_values.astype(compute_dtype))
        batch, grid_height, grid_width, hidden_size = patches.shape
        hidden = patches.reshape(batch, grid_height * grid_width, hidden_size)
        hidden = hidden + self.position_embedding[None].astype(compute_dtype)
        for layer in self.layers:
            hidden = layer(
                hidden,
                config=self.config,
                implementation=attention_implementation,
            )
        return self.final_norm(hidden)


def pixel_shuffle(
    hidden: Float[Array, "batch sequence channels"],
    *,
    factor: int,
) -> Float[Array, "batch shuffled_sequence shuffled_channels"]:
    """Apply ModernVBERT's exact connector token ordering."""

    batch, sequence, channels = hidden.shape
    height = math.isqrt(sequence)
    if height * height != sequence or height % factor:
        raise ValueError("vision token grid must be square and divisible by factor")
    value = hidden.reshape(batch, height, height, channels)
    value = value.reshape(batch, height, height // factor, channels * factor)
    value = jnp.transpose(value, (0, 2, 1, 3))
    value = value.reshape(
        batch,
        height // factor,
        height // factor,
        channels * factor**2,
    )
    value = jnp.transpose(value, (0, 2, 1, 3))
    return value.reshape(batch, sequence // factor**2, channels * factor**2)
