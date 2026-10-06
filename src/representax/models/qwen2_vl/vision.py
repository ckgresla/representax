"""Shared native Qwen2-VL and Qwen2.5-VL patch transformer."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    RMSNorm,
    dot_product_attention,
    rematerialize,
)
from representax.planning import RematerializationPolicy

from .config import Qwen2VLVisionConfig


def _rotate_half(value: Float[Array, "*batch head"]) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def vision_rotary_embedding(
    config: Qwen2VLVisionConfig,
    position_ids: Int[Array, "patch coordinate"],
) -> tuple[Float[Array, "patch head"], Float[Array, "patch head"]]:
    """Construct spatial rotary values for merge-major patches."""

    rotary_dimension = config.head_dimension // 2
    inverse_frequency = 1.0 / (
        10_000.0
        ** (jnp.arange(0, rotary_dimension, 2, dtype=jnp.float32) / rotary_dimension)
    )
    frequencies = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    spatial = frequencies.reshape((position_ids.shape[0], -1))
    embedding = jnp.concatenate((spatial, spatial), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


class Qwen2VLVisionAttention(eqx.Module):
    qkv: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2VLVisionAttention:
        qkv_key, output_key = jax.random.split(key)
        return cls(
            qkv=Linear.init(
                config.hidden_size,
                3 * config.hidden_size,
                key=qkv_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                config.hidden_size,
                config.hidden_size,
                key=output_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
        attention_mask: Bool[Array, "one one patch patch"],
        cosine: Float[Array, "patch head"],
        sine: Float[Array, "patch head"],
        *,
        config: Qwen2VLVisionConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "patch hidden"]:
        patch_count = hidden.shape[0]
        qkv = self.qkv(hidden).reshape(
            patch_count,
            3,
            config.num_attention_heads,
            config.head_dimension,
        )
        query, key, value = jnp.moveaxis(qkv, 1, 0)
        cosine = cosine[:, None].astype(jnp.float32)
        sine = sine[:, None].astype(jnp.float32)
        query = (
            query.astype(jnp.float32) * cosine
            + _rotate_half(query.astype(jnp.float32)) * sine
        ).astype(value.dtype)
        key = (
            key.astype(jnp.float32) * cosine
            + _rotate_half(key.astype(jnp.float32)) * sine
        ).astype(value.dtype)
        attended = dot_product_attention(
            query[None],
            key[None],
            value[None],
            attention_mask=attention_mask,
            implementation=implementation,
        )[0]
        return self.output(attended.reshape((patch_count, config.hidden_size)))


class Qwen2VLVisionMLP(eqx.Module):
    first: Linear
    gate: Linear | None
    second: Linear
    activation: str = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen2VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2VLVisionMLP:
        keys = jax.random.split(key, 3)
        return cls(
            first=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            gate=(
                Linear.init(
                    config.hidden_size,
                    config.intermediate_size,
                    key=keys[1],
                    scale=config.initializer_range,
                    dtype=dtype,
                    bias=True,
                )
                if config.mlp == "swiglu"
                else None
            ),
            second=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=keys[2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            activation=config.mlp,
        )

    def __call__(self, hidden: Float[Array, "patch hidden"]):
        if self.activation == "gelu":
            activated = jax.nn.gelu(self.first(hidden), approximate=True)
        elif self.activation == "quick_gelu":
            values = self.first(hidden)
            activated = values * jax.nn.sigmoid(1.702 * values)
        else:
            if self.gate is None:
                raise AssertionError("SwiGLU vision MLP requires a gate")
            activated = jax.nn.silu(self.gate(hidden)) * self.first(hidden)
        return self.second(activated)


class Qwen2VLVisionBlock(eqx.Module):
    attention_norm: Any
    attention: Qwen2VLVisionAttention
    mlp_norm: Any
    mlp: Qwen2VLVisionMLP

    @classmethod
    def init(
        cls,
        config: Qwen2VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2VLVisionBlock:
        attention_key, mlp_key = jax.random.split(key)
        if config.norm == "rms":
            attention_norm: Any = RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            )
            mlp_norm: Any = RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            )
        else:
            attention_norm = LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            )
            mlp_norm = LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            )
        return cls(
            attention_norm=attention_norm,
            attention=Qwen2VLVisionAttention.init(
                config, key=attention_key, dtype=dtype
            ),
            mlp_norm=mlp_norm,
            mlp=Qwen2VLVisionMLP.init(config, key=mlp_key, dtype=dtype),
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
        attention_mask: Bool[Array, "one one patch patch"],
        cosine: Float[Array, "patch head"],
        sine: Float[Array, "patch head"],
        *,
        config: Qwen2VLVisionConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "patch hidden"]:
        hidden = hidden + self.attention(
            self.attention_norm(hidden),
            attention_mask,
            cosine,
            sine,
            config=config,
            implementation=implementation,
        )
        return hidden + self.mlp(self.mlp_norm(hidden))


class Qwen2VLVisionBlockStack(eqx.Module):
    blocks: Qwen2VLVisionBlock
    full_attention: Bool[Array, " depth"]
    depth: int = eqx.field(static=True)

    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[Qwen2VLVisionBlock, ...],
        full_attention_layers: tuple[int, ...],
    ) -> Qwen2VLVisionBlockStack:
        full = frozenset(full_attention_layers)
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *blocks),
            full_attention=jnp.asarray(
                tuple(index in full for index in range(len(blocks)))
            ),
            depth=len(blocks),
        )


class Qwen2VLPatchMerger(eqx.Module):
    norm: Any
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2VLPatchMerger:
        up_key, down_key = jax.random.split(key)
        merged = config.hidden_size * config.spatial_merge_unit
        norm: Any
        if config.norm == "rms":
            norm = RMSNorm(jnp.ones((config.hidden_size,), dtype), config.norm_epsilon)
        else:
            norm = LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            )
        return cls(
            norm=norm,
            up=Linear.init(
                merged,
                merged,
                key=up_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            down=Linear.init(
                merged,
                config.output_size,
                key=down_key,
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(self, hidden: Float[Array, "patch hidden"]):
        hidden = self.norm(hidden)
        hidden = hidden.reshape((-1, self.up.weight.shape[1]))
        return self.down(jax.nn.gelu(self.up(hidden), approximate=False))


class Qwen2VLVisionTower(eqx.Module):
    patch_embedding: Linear
    blocks: Qwen2VLVisionBlockStack
    merger: Qwen2VLPatchMerger
    config: Qwen2VLVisionConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen2VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2VLVisionTower:
        keys = jax.random.split(key, config.depth + 2)
        return cls(
            patch_embedding=Linear.init(
                config.patch_dimension,
                config.hidden_size,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            blocks=Qwen2VLVisionBlockStack.from_blocks(
                tuple(
                    Qwen2VLVisionBlock.init(config, key=keys[index + 1], dtype=dtype)
                    for index in range(config.depth)
                ),
                config.full_attention_layers,
            ),
            merger=Qwen2VLPatchMerger.init(config, key=keys[-1], dtype=dtype),
            config=config,
        )

    def __call__(
        self,
        pixel_values: Float[Array, "patch pixel"],
        patch_valid: Bool[Array, " patch"],
        full_segment_ids: Int[Array, " patch"],
        window_segment_ids: Int[Array, " patch"],
        position_ids: Int[Array, "patch coordinate"],
        reverse_merged_indices: Int[Array, " merged"],
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "visual output"]:
        hidden = self.patch_embedding(pixel_values).astype(compute_dtype)
        cosine, sine = vision_rotary_embedding(self.config, position_ids)

        def make_mask(segment_ids):
            return (
                (segment_ids[:, None] == segment_ids[None, :])
                & patch_valid[:, None]
                & patch_valid[None, :]
            )[None, None]

        full_mask = make_mask(full_segment_ids)
        window_mask = make_mask(window_segment_ids)

        def apply_block(carry, values):
            block, full_attention = values
            output = block(
                carry,
                jnp.where(full_attention, full_mask, window_mask),
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_block, rematerialization),
            hidden,
            (self.blocks.blocks, self.blocks.full_attention),
        )
        return self.merger(hidden)[reverse_merged_indices]


__all__ = [
    "Qwen2VLPatchMerger",
    "Qwen2VLVisionAttention",
    "Qwen2VLVisionBlock",
    "Qwen2VLVisionMLP",
    "Qwen2VLVisionTower",
    "vision_rotary_embedding",
]
