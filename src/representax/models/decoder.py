"""Reusable scanned rotary decoder for Llama- and Mistral-family backbones."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from pydantic import model_validator

from representax._config import FrozenConfig
from representax.models.components import (
    AttentionImplementation,
    Linear,
    RMSNorm,
    dot_product_attention,
    embedding_lookup,
    rematerialize,
)
from representax.planning import RematerializationPolicy

RotaryDecoderFamily = Literal["llama", "mistral"]
RotaryAttentionMode = Literal["causal", "bidirectional"]


class Llama3RopeScalingConfig(FrozenConfig):
    """Llama 3 frequency interpolation parameters."""

    factor: float
    low_frequency_factor: float
    high_frequency_factor: float
    original_max_position_embeddings: int

    @model_validator(mode="after")
    def validate_scaling(self) -> Self:
        for name in ("factor", "low_frequency_factor", "high_frequency_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.high_frequency_factor <= self.low_frequency_factor:
            raise ValueError("high_frequency_factor must exceed low_frequency_factor")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("original_max_position_embeddings must be positive")
        return self


class RotaryDecoderConfig(FrozenConfig):
    """Shared Llama/Mistral causal-decoder architecture."""

    family: RotaryDecoderFamily
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dimension: int
    max_position_embeddings: int
    rope_theta: float
    norm_epsilon: float
    attention_bias: bool = False
    initializer_range: float = 0.02
    attention_mode: RotaryAttentionMode = "causal"
    rope_scaling: Llama3RopeScalingConfig | None = None

    @model_validator(mode="after")
    def validate_architecture(self) -> Self:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dimension",
            "max_position_embeddings",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if self.head_dimension % 2:
            raise ValueError("head_dimension must be even")
        if self.num_attention_heads * self.head_dimension != self.hidden_size:
            raise ValueError("attention heads must span hidden_size")
        for name in ("rope_theta", "norm_epsilon", "initializer_range"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        return self

    @classmethod
    def from_hf_config(cls, value: Mapping[str, Any]) -> RotaryDecoderConfig:
        family = str(value.get("model_type", ""))
        bidirectional = family in {"llama_bidirec", "llama_bidirectional"}
        if bidirectional:
            family = "llama"
        if family not in {"llama", "mistral"}:
            raise ValueError("expected a Llama or Mistral text_config")
        hidden_size = int(value.get("hidden_size", 4096))
        heads = int(value.get("num_attention_heads", 32))
        raw_rope = value.get("rope_parameters", value.get("rope_scaling"))
        rope_scaling = None
        if isinstance(raw_rope, Mapping):
            rope_type = str(raw_rope.get("rope_type", raw_rope.get("type", "")))
            if rope_type == "llama3":
                rope_scaling = Llama3RopeScalingConfig(
                    factor=float(raw_rope["factor"]),
                    low_frequency_factor=float(raw_rope["low_freq_factor"]),
                    high_frequency_factor=float(raw_rope["high_freq_factor"]),
                    original_max_position_embeddings=int(
                        raw_rope["original_max_position_embeddings"]
                    ),
                )
            elif rope_type not in {"", "default"}:
                raise ValueError(f"unsupported rotary scaling {rope_type!r}")
        return cls(
            family=family,
            vocab_size=int(value["vocab_size"]),
            hidden_size=hidden_size,
            intermediate_size=int(value.get("intermediate_size", 14336)),
            num_hidden_layers=int(value.get("num_hidden_layers", 32)),
            num_attention_heads=heads,
            num_key_value_heads=int(value.get("num_key_value_heads", heads)),
            head_dimension=int(value.get("head_dim", hidden_size // heads)),
            max_position_embeddings=int(value.get("max_position_embeddings", 4096)),
            rope_theta=float(value.get("rope_theta", 10_000.0)),
            norm_epsilon=float(value.get("rms_norm_eps", 1e-6)),
            attention_bias=bool(value.get("attention_bias", False)),
            initializer_range=float(value.get("initializer_range", 0.02)),
            attention_mode="bidirectional" if bidirectional else "causal",
            rope_scaling=rope_scaling,
        )

    def to_hf_config(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "model_type": self.family,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dimension,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.norm_epsilon,
            "attention_bias": self.attention_bias,
            "initializer_range": self.initializer_range,
            "hidden_act": "silu",
            "use_cache": True,
        }
        if self.rope_scaling is not None:
            value["rope_scaling"] = {
                "rope_type": "llama3",
                "factor": self.rope_scaling.factor,
                "low_freq_factor": self.rope_scaling.low_frequency_factor,
                "high_freq_factor": self.rope_scaling.high_frequency_factor,
                "original_max_position_embeddings": (
                    self.rope_scaling.original_max_position_embeddings
                ),
            }
        return value


def _rotate_half(value: Float[Array, "*batch head"]) -> Float[Array, "*batch head"]:
    first, second = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


def rotary_embedding(
    config: RotaryDecoderConfig,
    position_ids: Int[Array, "batch sequence"],
) -> tuple[
    Float[Array, "batch sequence head"],
    Float[Array, "batch sequence head"],
]:
    inverse_frequency = 1.0 / (
        config.rope_theta
        ** (
            jnp.arange(0, config.head_dimension, 2, dtype=jnp.float32)
            / config.head_dimension
        )
    )
    scaling = config.rope_scaling
    if scaling is not None:
        wavelength = 2 * math.pi / inverse_frequency
        low_wavelength = (
            scaling.original_max_position_embeddings / scaling.low_frequency_factor
        )
        high_wavelength = (
            scaling.original_max_position_embeddings / scaling.high_frequency_factor
        )
        scaled = jnp.where(
            wavelength > low_wavelength,
            inverse_frequency / scaling.factor,
            inverse_frequency,
        )
        smooth = (
            scaling.original_max_position_embeddings / wavelength
            - scaling.low_frequency_factor
        ) / (scaling.high_frequency_factor - scaling.low_frequency_factor)
        interpolated = (1 - smooth) * scaled / scaling.factor + smooth * scaled
        medium = (wavelength >= high_wavelength) & (wavelength <= low_wavelength)
        inverse_frequency = jnp.where(medium, interpolated, scaled)
    frequency = position_ids.astype(jnp.float32)[..., None] * inverse_frequency
    embedding = jnp.concatenate((frequency, frequency), axis=-1)
    return jnp.cos(embedding), jnp.sin(embedding)


class RotaryDecoderLayer(eqx.Module):
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
        config: RotaryDecoderConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> RotaryDecoderLayer:
        keys = jax.random.split(key, 7)
        attention = config.num_attention_heads * config.head_dimension
        key_value = config.num_key_value_heads * config.head_dimension

        def linear(
            input_size: int,
            output_size: int,
            index: int,
            *,
            bias: bool = False,
        ) -> Linear:
            return Linear.init(
                input_size,
                output_size,
                key=keys[index],
                scale=config.initializer_range,
                dtype=dtype,
                bias=bias,
            )

        return cls(
            input_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            ),
            post_attention_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            ),
            query=linear(config.hidden_size, attention, 0, bias=config.attention_bias),
            key=linear(config.hidden_size, key_value, 1, bias=config.attention_bias),
            value=linear(config.hidden_size, key_value, 2, bias=config.attention_bias),
            output=linear(attention, config.hidden_size, 3),
            gate=linear(config.hidden_size, config.intermediate_size, 4),
            up=linear(config.hidden_size, config.intermediate_size, 5),
            down=linear(config.intermediate_size, config.hidden_size, 6),
        )

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch one sequence sequence"],
        cosine: Float[Array, "batch sequence head"],
        sine: Float[Array, "batch sequence head"],
        *,
        config: RotaryDecoderConfig,
        implementation: AttentionImplementation,
    ) -> Float[Array, "batch sequence hidden"]:
        batch, sequence, _ = hidden.shape
        normalized = self.input_norm(hidden)
        query = self.query(normalized).reshape(
            batch, sequence, config.num_attention_heads, config.head_dimension
        )
        key = self.key(normalized).reshape(
            batch, sequence, config.num_key_value_heads, config.head_dimension
        )
        value = self.value(normalized).reshape(
            batch, sequence, config.num_key_value_heads, config.head_dimension
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
        hidden = hidden + self.output(attended)
        normalized = self.post_attention_norm(hidden)
        mlp = jax.nn.silu(self.gate(normalized)) * self.up(normalized)
        return hidden + self.down(mlp)


class RotaryDecoderLayerStack(eqx.Module):
    blocks: RotaryDecoderLayer
    depth: int = eqx.field(static=True)

    @classmethod
    def from_layers(
        cls, layers: tuple[RotaryDecoderLayer, ...]
    ) -> RotaryDecoderLayerStack:
        if not layers:
            raise ValueError("rotary decoders require at least one layer")
        compute_layers = tuple(
            jax.tree.map(
                lambda value: (
                    value.input_major() if isinstance(value, Linear) else value
                ),
                layer,
                is_leaf=lambda value: isinstance(value, Linear),
            )
            for layer in layers
        )
        return cls(
            blocks=jax.tree.map(lambda *values: jnp.stack(values), *compute_layers),
            depth=len(layers),
        )

    def layer(self, index: int) -> RotaryDecoderLayer:
        if not 0 <= index < self.depth:
            raise IndexError(index)
        layer = jax.tree.map(lambda value: value[index], self.blocks)
        return jax.tree.map(
            lambda value: value.output_major() if isinstance(value, Linear) else value,
            layer,
            is_leaf=lambda value: isinstance(value, Linear),
        )


class RotaryDecoderTower(eqx.Module):
    token_embedding: Float[Array, "vocabulary hidden"]
    layers: RotaryDecoderLayerStack
    final_norm: RMSNorm
    config: RotaryDecoderConfig = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        config: RotaryDecoderConfig,
        *,
        key: PRNGKeyArray,
        dtype: jnp.dtype,
    ) -> RotaryDecoderTower:
        keys = jax.random.split(key, config.num_hidden_layers + 1)
        return cls(
            token_embedding=config.initializer_range
            * jax.random.normal(
                keys[0], (config.vocab_size, config.hidden_size), dtype=dtype
            ),
            layers=RotaryDecoderLayerStack.from_layers(
                tuple(
                    RotaryDecoderLayer.init(config, key=keys[index + 1], dtype=dtype)
                    for index in range(config.num_hidden_layers)
                )
            ),
            final_norm=RMSNorm(
                jnp.ones((config.hidden_size,), dtype), config.norm_epsilon
            ),
            config=config,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        *,
        inputs_embeds: Float[Array, "batch sequence hidden"] | None = None,
        position_ids: Int[Array, "batch sequence"] | None = None,
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
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(sequence, dtype=jnp.int32)[None], (batch, sequence)
            )
        cosine, sine = rotary_embedding(self.config, position_ids)
        if self.config.attention_mode == "causal":
            target = jnp.arange(sequence)[:, None]
            source = jnp.arange(sequence)[None, :]
            allowed = (source <= target)[None, None]
        else:
            allowed = jnp.ones((1, 1, sequence, sequence), dtype=bool)
        allowed = allowed & attention_mask[:, None, None, :].astype(bool)

        def apply_layer(carry, index):
            layer = jax.tree.map(
                lambda value: jax.lax.dynamic_index_in_dim(
                    value, index, axis=0, keepdims=False
                ),
                self.layers.blocks,
            )
            output = layer(
                carry,
                allowed,
                cosine,
                sine,
                config=self.config,
                implementation=attention_implementation,
            )
            return output, None

        hidden, _ = jax.lax.scan(
            rematerialize(apply_layer, rematerialization),
            hidden,
            jnp.arange(self.layers.depth, dtype=jnp.int32),
        )
        return self.final_norm(hidden)


__all__ = [
    "Llama3RopeScalingConfig",
    "RotaryDecoderConfig",
    "RotaryDecoderLayer",
    "RotaryDecoderLayerStack",
    "RotaryDecoderTower",
    "rotary_embedding",
]
