"""Native Qwen3-VL image/video patch transformer."""

from __future__ import annotations

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

from .config import Qwen3VLVisionConfig


def vision_rotary_embedding(
    config: Qwen3VLVisionConfig,
    position_ids: Int[Array, "patch coordinate"],
) -> tuple[Float[Array, "patch head"], Float[Array, "patch head"]]:
    """Construct the spatial RoPE values for patches in merge-major order."""

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


def interpolate_position_embedding(
    table: Float[Array, "position hidden"],
    indices: Int[Array, "corner patch"],
    weights: Float[Array, "corner patch"],
) -> Float[Array, "patch hidden"]:
    """Apply exact bilinear interpolation from the learned square position grid."""

    if indices.shape != weights.shape or indices.ndim != 2 or indices.shape[0] != 4:
        raise ValueError("position interpolation requires [4, patch] arrays")
    gathered = table[indices]
    return jnp.sum(gathered * weights[..., None].astype(gathered.dtype), axis=0)


def _rotate_half(value: Float[Array, "*batch head"]) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


class Qwen3VLVisionAttention(eqx.Module):
    qkv: Linear
    output: Linear

    @classmethod
    def init(
        cls,
        config: Qwen3VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLVisionAttention:
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
        config: Qwen3VLVisionConfig,
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


class Qwen3VLVisionMLP(eqx.Module):
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen3VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLVisionMLP:
        up_key, down_key = jax.random.split(key)
        return cls(
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
        hidden: Float[Array, "patch hidden"],
    ) -> Float[Array, "patch hidden"]:
        return self.down(jax.nn.gelu(self.up(hidden), approximate=True))


class Qwen3VLVisionBlock(eqx.Module):
    attention_norm: LayerNorm
    attention: Qwen3VLVisionAttention
    mlp_norm: LayerNorm
    mlp: Qwen3VLVisionMLP

    @classmethod
    def init(
        cls,
        config: Qwen3VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLVisionBlock:
        attention_key, mlp_key = jax.random.split(key)
        return cls(
            attention_norm=LayerNorm.init(
                config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
            attention=Qwen3VLVisionAttention.init(
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
            mlp=Qwen3VLVisionMLP.init(config, key=mlp_key, dtype=dtype),
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
        attention_mask: Bool[Array, "one one patch patch"],
        cosine: Float[Array, "patch head"],
        sine: Float[Array, "patch head"],
        *,
        config: Qwen3VLVisionConfig,
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


class Qwen3VLVisionBlockStack(eqx.Module):
    blocks: Qwen3VLVisionBlock
    depth: int = eqx.field(static=True)

    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[Qwen3VLVisionBlock, ...],
    ) -> Qwen3VLVisionBlockStack:
        if not blocks:
            raise ValueError("Qwen3-VL requires at least one vision block")
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *blocks),
            depth=len(blocks),
        )


class Qwen3VLPatchMerger(eqx.Module):
    norm: LayerNorm
    up: Linear
    down: Linear
    postshuffle_norm: bool = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen3VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
        postshuffle_norm: bool,
    ) -> Qwen3VLPatchMerger:
        up_key, down_key = jax.random.split(key)
        merged = config.hidden_size * config.spatial_merge_unit
        return cls(
            norm=LayerNorm.init(
                merged if postshuffle_norm else config.hidden_size,
                epsilon=config.norm_epsilon,
                dtype=dtype,
                bias=True,
            ),
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
            postshuffle_norm=postshuffle_norm,
        )

    def __call__(
        self,
        hidden: Float[Array, "patch hidden"],
    ) -> Float[Array, "visual output"]:
        merged_size = self.up.weight.shape[1]
        if hidden.shape[0] * hidden.shape[1] % merged_size:
            raise ValueError("patch count must divide the spatial merge unit")
        if self.postshuffle_norm:
            hidden = self.norm(hidden.reshape((-1, merged_size)))
        else:
            hidden = self.norm(hidden).reshape((-1, merged_size))
        return self.down(jax.nn.gelu(self.up(hidden), approximate=False))


class Qwen3VLVisionTower(eqx.Module):
    patch_embedding: Linear
    position_embedding: Float[Array, "position hidden"]
    blocks: Qwen3VLVisionBlockStack
    merger: Qwen3VLPatchMerger
    deepstack_mergers: Qwen3VLPatchMerger | None
    config: Qwen3VLVisionConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen3VLVisionConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLVisionTower:
        key_count = config.depth + len(config.deepstack_visual_indexes) + 3
        keys = jax.random.split(key, key_count)
        offset = 0
        patch_embedding = Linear.init(
            config.patch_dimension,
            config.hidden_size,
            key=keys[offset],
            scale=config.initializer_range,
            dtype=dtype,
            bias=True,
        )
        offset += 1
        position_embedding = config.initializer_range * jax.random.normal(
            keys[offset],
            (config.num_position_embeddings, config.hidden_size),
            dtype=dtype,
        )
        offset += 1
        blocks = Qwen3VLVisionBlockStack.from_blocks(
            tuple(
                Qwen3VLVisionBlock.init(config, key=keys[offset + index], dtype=dtype)
                for index in range(config.depth)
            )
        )
        offset += config.depth
        merger = Qwen3VLPatchMerger.init(
            config,
            key=keys[offset],
            dtype=dtype,
            postshuffle_norm=False,
        )
        offset += 1
        deepstack = tuple(
            Qwen3VLPatchMerger.init(
                config,
                key=keys[offset + index],
                dtype=dtype,
                postshuffle_norm=True,
            )
            for index in range(len(config.deepstack_visual_indexes))
        )
        return cls(
            patch_embedding=patch_embedding,
            position_embedding=position_embedding,
            blocks=blocks,
            merger=merger,
            deepstack_mergers=(
                None
                if not deepstack
                else jax.tree.map(lambda *values: jnp.stack(values), *deepstack)
            ),
            config=config,
        )

    def __call__(
        self,
        pixel_values: Float[Array, "patch pixel"],
        patch_valid: Bool[Array, " patch"],
        segment_ids: Int[Array, " patch"],
        position_ids: Int[Array, "patch coordinate"],
        position_indices: Int[Array, "corner patch"],
        position_weights: Float[Array, "corner patch"],
        *,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> tuple[
        Float[Array, "visual output"],
        Float[Array, "depth visual output"],
    ]:
        patch_count = pixel_values.shape[0]
        if pixel_values.shape[1] != self.config.patch_dimension:
            raise ValueError("pixel_values has the wrong flattened patch size")
        if patch_count % self.config.spatial_merge_unit:
            raise ValueError("patch bucket must divide the spatial merge unit")
        if patch_valid.shape != (patch_count,) or segment_ids.shape != (patch_count,):
            raise ValueError("patch masks must align with pixel_values")
        hidden = self.patch_embedding(pixel_values).astype(compute_dtype)
        hidden = hidden + interpolate_position_embedding(
            self.position_embedding,
            position_indices,
            position_weights,
        ).astype(compute_dtype)
        cosine, sine = vision_rotary_embedding(self.config, position_ids)
        attention_mask = (
            (segment_ids[:, None] == segment_ids[None, :])
            & patch_valid[:, None]
            & patch_valid[None, :]
        )[None, None]
        deep_count = len(self.config.deepstack_visual_indexes)
        selected = jnp.zeros(
            (deep_count, patch_count, self.config.hidden_size),
            dtype=hidden.dtype,
        )
        selected_indexes = jnp.asarray(
            self.config.deepstack_visual_indexes,
            dtype=jnp.int32,
        )

        def apply_block(carry, values):
            layer_hidden, deep_hidden = carry
            block, index = values
            layer_hidden = block(
                layer_hidden,
                attention_mask,
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            matches = index == selected_indexes
            deep_hidden = jnp.where(
                matches[:, None, None],
                layer_hidden[None],
                deep_hidden,
            )
            return (layer_hidden, deep_hidden), None

        (hidden, selected), _ = jax.lax.scan(
            rematerialize(apply_block, rematerialization),
            (hidden, selected),
            (self.blocks.blocks, jnp.arange(self.blocks.depth)),
        )
        merged = self.merger(hidden)
        if self.deepstack_mergers is None:
            deepstack = jnp.zeros((0, *merged.shape), dtype=merged.dtype)
        else:
            deepstack = jax.vmap(lambda merger, values: merger(values))(
                self.deepstack_mergers,
                selected,
            )
        return merged, deepstack


__all__ = [
    "Qwen3VLPatchMerger",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionBlockStack",
    "Qwen3VLVisionTower",
    "interpolate_position_embedding",
    "vision_rotary_embedding",
]
