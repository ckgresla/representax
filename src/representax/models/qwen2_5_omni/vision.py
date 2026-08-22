"""Native Qwen2.5-Omni windowed image/video patch transformer."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.models.components import (
    AttentionImplementation,
    Linear,
    RMSNorm,
    dot_product_attention,
    rematerialize,
)
from representax.planning import RematerializationPolicy

from .config import Qwen2_5OmniVisionConfig


def _rotate_half(value: Float[Array, "*batch head"]) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def vision_rotary_embedding(
    config: Qwen2_5OmniVisionConfig,
    position_ids: Int[Array, "patch coordinate"],
) -> tuple[Float[Array, "patch head"], Float[Array, "patch head"]]:
    """Construct Qwen2.5-Omni spatial RoPE for window-reordered patches."""

    if position_ids.ndim != 2 or position_ids.shape[1] != 2:
        raise ValueError("vision position_ids must have shape [patch, 2]")
    rotary_dimension = config.head_dimension // 2
    inverse_frequency = 1.0 / (
        10_000.0
        ** (jnp.arange(0, rotary_dimension, 2, dtype=jnp.float32) / rotary_dimension)
    )
    frequencies = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    spatial = frequencies.reshape((position_ids.shape[0], -1))
    embedding = jnp.concatenate((spatial, spatial), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


class Qwen2_5OmniVisionAttention(eqx.Module):
    query: Linear
    key: Linear
    value: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniVisionAttention:
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
        hidden: Float[Array, "patch hidden"],
        attention_mask: Bool[Array, "one one patch patch"],
        cosine: Float[Array, "patch head"],
        sine: Float[Array, "patch head"],
        *,
        config: Qwen2_5OmniVisionConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "patch hidden"]:
        patch_count = hidden.shape[0]
        shape = (patch_count, config.num_attention_heads, config.head_dimension)
        query = self.query(hidden).reshape(shape)
        key = self.key(hidden).reshape(shape)
        value = self.value(hidden).reshape(shape)
        query_dtype = query.dtype
        key_dtype = key.dtype
        cosine = cosine[:, None].astype(jnp.float32)
        sine = sine[:, None].astype(jnp.float32)
        query = (
            query.astype(jnp.float32) * cosine
            + _rotate_half(query.astype(jnp.float32)) * sine
        ).astype(query_dtype)
        key = (
            key.astype(jnp.float32) * cosine
            + _rotate_half(key.astype(jnp.float32)) * sine
        ).astype(key_dtype)
        attended = dot_product_attention(
            query[None],
            key[None],
            value[None],
            attention_mask=attention_mask,
            implementation=implementation,
        )[0]
        return self.output(attended.reshape((patch_count, config.hidden_size)))


class Qwen2_5OmniVisionMLP(eqx.Module):
    gate: Linear
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniVisionMLP:
        keys = jax.random.split(key, 3)
        return cls(
            gate=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            up=Linear.init(
                config.hidden_size,
                config.intermediate_size,
                key=keys[1],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            down=Linear.init(
                config.intermediate_size,
                config.hidden_size,
                key=keys[2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
    ) -> Float[Array, "patch hidden"]:
        return self.down(jax.nn.silu(self.gate(hidden)) * self.up(hidden))


class Qwen2_5OmniVisionBlock(eqx.Module):
    attention_norm: RMSNorm
    attention: Qwen2_5OmniVisionAttention
    mlp_norm: RMSNorm
    mlp: Qwen2_5OmniVisionMLP

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniVisionBlock:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            ),
            attention=Qwen2_5OmniVisionAttention.init(
                config,
                key=attention_key,
                dtype=dtype,
            ),
            mlp_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            ),
            mlp=Qwen2_5OmniVisionMLP.init(config, key=mlp_key, dtype=dtype),
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
        attention_mask: Bool[Array, "one one patch patch"],
        cosine: Float[Array, "patch head"],
        sine: Float[Array, "patch head"],
        *,
        config: Qwen2_5OmniVisionConfig,
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


class Qwen2_5OmniVisionBlockStack(eqx.Module):
    blocks: Qwen2_5OmniVisionBlock
    full_attention: Bool[Array, " depth"]
    depth: int = eqx.field(static=True)

    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[Qwen2_5OmniVisionBlock, ...],
        full_attention_layers: tuple[int, ...],
    ) -> Qwen2_5OmniVisionBlockStack:
        if not blocks:
            raise ValueError("Qwen2.5-Omni requires at least one vision block")
        full = frozenset(full_attention_layers)
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *blocks),
            full_attention=jnp.asarray(
                tuple(index in full for index in range(len(blocks)))
            ),
            depth=len(blocks),
        )


class Qwen2_5OmniPatchMerger(eqx.Module):
    norm: RMSNorm
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniPatchMerger:
        up_key, down_key = jax.random.split(key)
        merged = config.hidden_size * config.spatial_merge_unit
        return cls(
            norm=RMSNorm(jnp.ones((config.hidden_size,), dtype), config.norm_epsilon),
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

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
    ) -> Float[Array, "visual output"]:
        hidden = self.norm(hidden)
        merged_size = self.up.weight.shape[1]
        if hidden.shape[0] * hidden.shape[1] % merged_size:
            raise ValueError("patch count must divide the spatial merge unit")
        hidden = hidden.reshape((-1, merged_size))
        return self.down(jax.nn.gelu(self.up(hidden), approximate=False))


class Qwen2_5OmniVisionTower(eqx.Module):
    patch_embedding: Linear
    blocks: Qwen2_5OmniVisionBlockStack
    merger: Qwen2_5OmniPatchMerger
    config: Qwen2_5OmniVisionConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniVisionTower:
        keys = jax.random.split(key, config.depth + 2)
        return cls(
            patch_embedding=Linear.init(
                config.patch_dimension,
                config.hidden_size,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            blocks=Qwen2_5OmniVisionBlockStack.from_blocks(
                tuple(
                    Qwen2_5OmniVisionBlock.init(
                        config,
                        key=keys[index + 1],
                        dtype=dtype,
                    )
                    for index in range(config.depth)
                ),
                config.full_attention_layers,
            ),
            merger=Qwen2_5OmniPatchMerger.init(
                config,
                key=keys[-1],
                dtype=dtype,
            ),
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
        patch_count = pixel_values.shape[0]
        if pixel_values.shape[1] != self.config.patch_dimension:
            raise ValueError("pixel_values has the wrong flattened patch size")
        if patch_count % self.config.spatial_merge_unit:
            raise ValueError("patch bucket must divide the spatial merge unit")
        for values in (patch_valid, full_segment_ids, window_segment_ids):
            if values.shape != (patch_count,):
                raise ValueError("vision layout arrays must align with pixel_values")
        merged_count = patch_count // self.config.spatial_merge_unit
        if reverse_merged_indices.shape != (merged_count,):
            raise ValueError("reverse_merged_indices must align with merged patches")

        hidden = self.patch_embedding(pixel_values).astype(compute_dtype)
        cosine, sine = vision_rotary_embedding(self.config, position_ids)

        def mask_for(segment_ids: Int[Array, " patch"]):
            return (
                (segment_ids[:, None] == segment_ids[None, :])
                & patch_valid[:, None]
                & patch_valid[None, :]
            )[None, None]

        full_mask = mask_for(full_segment_ids)
        window_mask = mask_for(window_segment_ids)

        def apply_block(carry, values):
            block, full_attention = values
            attention_mask = jnp.where(full_attention, full_mask, window_mask)
            output = block(
                carry,
                attention_mask,
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
        merged = self.merger(hidden)
        return merged[reverse_merged_indices]


__all__ = [
    "Qwen2_5OmniPatchMerger",
    "Qwen2_5OmniVisionAttention",
    "Qwen2_5OmniVisionBlock",
    "Qwen2_5OmniVisionBlockStack",
    "Qwen2_5OmniVisionMLP",
    "Qwen2_5OmniVisionTower",
    "vision_rotary_embedding",
]
