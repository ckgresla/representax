"""Native Qwen2.5-Omni causal language tower with multimodal rotary positions."""

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

from .config import Qwen2_5OmniTextConfig


def _rotate_half(
    value: Float[Array, "*batch head"],
) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def text_rotary_embedding(
    config: Qwen2_5OmniTextConfig,
    position_ids: Int[Array, "position batch sequence"] | Int[Array, "batch sequence"],
) -> tuple[
    Float[Array, "batch sequence head"],
    Float[Array, "batch sequence head"],
]:
    """Construct Qwen2.5-Omni's sectioned temporal/height/width MRoPE."""

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
    frequency = (
        position_ids.astype(jnp.float32)[..., None]
        * inverse_frequency[None, None, None, :]
    )
    embedding = jnp.concatenate((frequency, frequency), axis=-1)
    sections = (*config.mrope_section, *config.mrope_section)
    boundaries = tuple(sum(sections[:index]) for index in range(1, len(sections)))
    chunks = jnp.split(embedding, boundaries, axis=-1)
    selected = tuple(chunk[index % 3] for index, chunk in enumerate(chunks))
    multimodal = jnp.concatenate(selected, axis=-1)
    return jnp.cos(multimodal), jnp.sin(multimodal)


class Qwen2_5OmniTextLayer(eqx.Module):
    """One pre-norm causal GQA/SwiGLU decoder block."""

    input_norm: RMSNorm
    post_attention_norm: RMSNorm
    query: Linear
    key: Linear
    value: Linear
    output: Linear
    gate: Linear
    up: Linear
    down: Linear

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniTextLayer:
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
                bias=True,
            ),
            key=Linear.init(
                hidden,
                key_value,
                key=keys[1],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            value=Linear.init(
                hidden,
                key_value,
                key=keys[2],
                scale=config.initializer_range,
                dtype=dtype,
                bias=True,
            ),
            output=Linear.init(
                attention,
                hidden,
                key=keys[3],
                scale=config.initializer_range,
                dtype=dtype,
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
        config: Qwen2_5OmniTextConfig,
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


class Qwen2_5OmniTextLayerStack(eqx.Module):
    """Depth-major homogeneous decoder layers executed with one scan."""

    blocks: Qwen2_5OmniTextLayer
    sliding: Bool[Array, " depth"]
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls,
        layers: tuple[Qwen2_5OmniTextLayer, ...],
        layer_types: tuple[str, ...],
    ) -> Qwen2_5OmniTextLayerStack:
        if not layers:
            raise ValueError("Qwen2.5-Omni requires at least one text layer")
        if len(layers) != len(layer_types):
            raise ValueError("layer types must align with text layers")
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *layers),
            sliding=jnp.asarray(
                tuple(item == "sliding_attention" for item in layer_types)
            ),
            depth=len(layers),
        )

    def layer(self, index: int) -> Qwen2_5OmniTextLayer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        return jax.tree.map(lambda value: value[index], self.blocks)


class Qwen2_5OmniTextTower(eqx.Module):
    """Token embedding, scanned causal decoder, and final RMS normalization."""

    token_embedding: Float[Array, "vocabulary hidden"]
    layers: Qwen2_5OmniTextLayerStack
    final_norm: RMSNorm
    config: Qwen2_5OmniTextConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: Qwen2_5OmniTextConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> Qwen2_5OmniTextTower:
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
            layers=Qwen2_5OmniTextLayerStack.from_layers(
                tuple(
                    Qwen2_5OmniTextLayer.init(
                        config,
                        key=keys[index + 1],
                        dtype=dtype,
                    )
                    for index in range(config.num_hidden_layers)
                ),
                config.layer_types,
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
        target = jnp.arange(sequence)[:, None]
        source = jnp.arange(sequence)[None, :]
        allowed = (source <= target)[None, None]
        allowed = allowed & attention_mask[:, None, None, :].astype(bool)
        if self.config.sliding_window is None:
            sliding_allowed = allowed
        else:
            within_window = source > target - self.config.sliding_window
            sliding_allowed = allowed & within_window[None, None]

        def apply_layer(
            carry: Float[Array, "batch sequence hidden"],
            values: tuple[Qwen2_5OmniTextLayer, Bool[Array, ""]],
        ) -> tuple[Float[Array, "batch sequence hidden"], None]:
            layer, sliding = values
            layer_mask = jnp.where(sliding, sliding_allowed, allowed)
            output = layer(
                carry,
                layer_mask,
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            (self.layers.blocks, self.layers.sliding),
        )
        if hidden.shape != (batch, sequence, self.config.hidden_size):
            raise AssertionError("text tower changed the hidden-state shape")
        return self.final_norm(hidden)


__all__ = [
    "Qwen2_5OmniTextLayer",
    "Qwen2_5OmniTextLayerStack",
    "Qwen2_5OmniTextTower",
    "text_rotary_embedding",
]
