"""Native Qwen3-VL causal language tower with multimodal rotary positions."""

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
    embedding_lookup,
    rematerialize,
)
from representax.planning import RematerializationPolicy

from .config import Qwen3VLTextConfig


def _rotate_half(
    value: Float[Array, "*batch head"],
) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def text_rotary_embedding(
    config: Qwen3VLTextConfig,
    position_ids: Int[Array, "position batch sequence"] | Int[Array, "batch sequence"],
) -> tuple[
    Float[Array, "batch sequence head"],
    Float[Array, "batch sequence head"],
]:
    """Construct Qwen3-VL's interleaved temporal/height/width MRoPE."""

    if position_ids.ndim == 2:
        position_ids = jnp.broadcast_to(position_ids[None], (3, *position_ids.shape))
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError("position_ids must have shape [3, batch, sequence]")
    inverse_frequency = 1.0 / (
        config.rope_theta
        ** (
            jnp.arange(0, config.head_dimension, 2, dtype=jnp.float32)
            / config.head_dimension
        )
    )
    frequencies = (
        position_ids.astype(jnp.float32)[..., None]
        * inverse_frequency[None, None, None, :]
    )
    interleaved = frequencies[0]
    for dimension, offset in ((1, 1), (2, 2)):
        stop = config.mrope_section[dimension] * 3
        interleaved = interleaved.at[..., offset:stop:3].set(
            frequencies[dimension, ..., offset:stop:3]
        )
    embedding = jnp.concatenate((interleaved, interleaved), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


class Qwen3VLTextLayer(eqx.Module):
    """One pre-norm causal GQA/SwiGLU decoder block."""

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

    @classmethod
    def init(
        cls,
        config: Qwen3VLTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLTextLayer:
        keys = jax.random.split(key, 7)
        hidden = config.hidden_size
        attention = config.num_attention_heads * config.head_dimension
        key_value = config.num_key_value_heads * config.head_dimension
        return cls(
            input_norm=RMSNorm(jnp.ones((hidden,), dtype), config.norm_epsilon),
            post_attention_norm=RMSNorm(
                jnp.ones((hidden,), dtype), config.norm_epsilon
            ),
            query=Linear.init(
                hidden,
                attention,
                key=keys[0],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            key=Linear.init(
                hidden,
                key_value,
                key=keys[1],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            value=Linear.init(
                hidden,
                key_value,
                key=keys[2],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            output=Linear.init(
                attention,
                hidden,
                key=keys[3],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            query_norm=RMSNorm(
                jnp.ones((config.head_dimension,), dtype), config.norm_epsilon
            ),
            key_norm=RMSNorm(
                jnp.ones((config.head_dimension,), dtype), config.norm_epsilon
            ),
            gate=Linear.init(
                hidden,
                config.intermediate_size,
                key=keys[4],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            up=Linear.init(
                hidden,
                config.intermediate_size,
                key=keys[5],
                scale=config.initializer_range,
                dtype=dtype,
            ),
            down=Linear.init(
                config.intermediate_size,
                hidden,
                key=keys[6],
                scale=config.initializer_range,
                dtype=dtype,
            ),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch one sequence sequence"],
        cosine: Float[Array, "batch sequence head"],
        sine: Float[Array, "batch sequence head"],
        *,
        config: Qwen3VLTextConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape
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
        attended = dot_product_attention(
            query.astype(value.dtype),
            key.astype(value.dtype),
            value,
            attention_mask=attention_mask,
            implementation=implementation,
        ).reshape(batch, sequence, -1)
        hidden = residual + self.output(attended)
        residual = hidden
        normalized = self.post_attention_norm(hidden)
        activated = jax.nn.silu(self.gate(normalized)) * self.up(normalized)
        return residual + self.down(activated)


class Qwen3VLTextLayerStack(eqx.Module):
    """Depth-major homogeneous decoder layers executed with one scan."""

    blocks: Qwen3VLTextLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls,
        layers: tuple[Qwen3VLTextLayer, ...],
    ) -> Qwen3VLTextLayerStack:
        if not layers:
            raise ValueError("Qwen3-VL requires at least one text layer")
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *layers),
            depth=len(layers),
        )

    def layer(self, index: int) -> Qwen3VLTextLayer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.blocks)


def _inject_visual(
    hidden: Float[Array, "batch sequence hidden"],
    visual: Float[Array, "visual hidden"],
    indices: Int[Array, " visual"],
    valid: Bool[Array, " visual"],
) -> Float[Array, "batch sequence hidden"]:
    flattened = hidden.reshape((-1, hidden.shape[-1]))
    updates = jnp.where(valid[:, None], visual, 0)
    return flattened.at[indices].add(updates).reshape(hidden.shape)


class Qwen3VLTextTower(eqx.Module):
    """Token embedding, scanned decoder, DeepStack injection, and final norm."""

    token_embedding: Float[Array, "vocabulary hidden"]
    layers: Qwen3VLTextLayerStack
    final_norm: RMSNorm
    config: Qwen3VLTextConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen3VLTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen3VLTextTower:
        keys = jax.random.split(key, config.num_hidden_layers + 1)
        return cls(
            token_embedding=(
                config.initializer_range
                * jax.random.normal(
                    keys[0],
                    (config.vocab_size, config.hidden_size),
                    dtype=dtype,
                )
            ),
            layers=Qwen3VLTextLayerStack.from_layers(
                tuple(
                    Qwen3VLTextLayer.init(config, key=keys[index + 1], dtype=dtype)
                    for index in range(config.num_hidden_layers)
                )
            ),
            final_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype),
                config.norm_epsilon,
            ),
            config=config,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        position_ids: Int[Array, "position batch sequence"]
        | Int[Array, "batch sequence"],
        *,
        inputs_embeds: Float[Array, "batch sequence hidden"] | None = None,
        deepstack_visual: Float[Array, "depth visual hidden"] | None = None,
        visual_token_indices: Int[Array, " visual"] | None = None,
        visual_token_valid: Bool[Array, " visual"] | None = None,
        bidirectional: bool = False,
        compute_dtype: jnp.dtype,
        attention_implementation: AttentionImplementation,
        rematerialization: RematerializationPolicy,
    ) -> Float[Array, "batch sequence hidden"]:
        hidden = (
            embedding_lookup(self.token_embedding, input_ids)
            if inputs_embeds is None
            else inputs_embeds
        ).astype(compute_dtype)
        batch, sequence = input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError("sequence exceeds max_position_embeddings")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask and input_ids must align")
        cosine, sine = text_rotary_embedding(self.config, position_ids)
        key_valid = attention_mask[:, None, None, :].astype(bool)
        if bidirectional:
            allowed = key_valid
        else:
            target = jnp.arange(sequence)[:, None]
            source = jnp.arange(sequence)[None, :]
            allowed = (source <= target)[None, None] & key_valid
        has_deepstack = deepstack_visual is not None
        if has_deepstack and (
            visual_token_indices is None or visual_token_valid is None
        ):
            raise ValueError("DeepStack features require visual token placement")

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            values: tuple[Qwen3VLTextLayer, Int[Array, ""]],
        ) -> tuple[Float[Array, "batch sequence hidden"], None]:
            layer, index = values
            output = layer(
                carry,
                allowed,
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            if has_deepstack:
                assert deepstack_visual is not None
                assert visual_token_indices is not None
                assert visual_token_valid is not None
                count = deepstack_visual.shape[0]
                output = jax.lax.cond(
                    index < count,
                    lambda value: _inject_visual(
                        value,
                        jax.lax.dynamic_index_in_dim(
                            deepstack_visual,
                            index,
                            axis=0,
                            keepdims=False,
                        ),
                        visual_token_indices,
                        visual_token_valid,
                    ),
                    lambda value: value,
                    output,
                )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            (self.layers.blocks, jnp.arange(self.layers.depth)),
        )
        if hidden.shape != (batch, sequence, self.config.hidden_size):
            raise AssertionError("text tower changed the hidden-state shape")
        return self.final_norm(hidden)


__all__ = [
    "Qwen3VLTextLayer",
    "Qwen3VLTextLayerStack",
    "Qwen3VLTextTower",
    "text_rotary_embedding",
]
