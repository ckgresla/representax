"""Checkpoint mapping for the reusable scanned rotary decoder."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from representax.models.components import Linear, RMSNorm
from representax.models.decoder import (
    RotaryDecoderConfig,
    RotaryDecoderLayer,
    RotaryDecoderLayerStack,
    RotaryDecoderTower,
)


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"rotary decoder checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
    return value


def _linear(
    state: Mapping[str, Any],
    prefix: str,
    *,
    input_size: int,
    output_size: int,
    dtype: jnp.dtype,
    bias: bool = False,
) -> Linear:
    return Linear(
        weight=_array(state, prefix + ".weight", (output_size, input_size), dtype),
        bias=(_array(state, prefix + ".bias", (output_size,), dtype) if bias else None),
    )


def rotary_decoder_weight_names(
    config: RotaryDecoderConfig,
    *,
    prefix: str,
) -> frozenset[str]:
    names = {prefix + "embed_tokens.weight", prefix + "norm.weight"}
    for index in range(config.num_hidden_layers):
        layer = prefix + f"layers.{index}."
        names.update(
            layer + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
        if config.attention_bias:
            names.update(
                layer + f"self_attn.{name}_proj.bias" for name in ("q", "k", "v")
            )
    return frozenset(names)


def rotary_decoder_from_state_dict(
    config: RotaryDecoderConfig,
    state: Mapping[str, Any],
    *,
    prefix: str,
    dtype: jnp.dtype,
) -> RotaryDecoderTower:
    hidden = config.hidden_size
    attention = config.num_attention_heads * config.head_dimension
    key_value = config.num_key_value_heads * config.head_dimension
    layers = []
    for index in range(config.num_hidden_layers):
        layer = prefix + f"layers.{index}."
        layers.append(
            RotaryDecoderLayer(
                input_norm=RMSNorm(
                    _array(state, layer + "input_layernorm.weight", (hidden,), dtype),
                    config.norm_epsilon,
                ),
                post_attention_norm=RMSNorm(
                    _array(
                        state,
                        layer + "post_attention_layernorm.weight",
                        (hidden,),
                        dtype,
                    ),
                    config.norm_epsilon,
                ),
                query=_linear(
                    state,
                    layer + "self_attn.q_proj",
                    input_size=hidden,
                    output_size=attention,
                    dtype=dtype,
                    bias=config.attention_bias,
                ),
                key=_linear(
                    state,
                    layer + "self_attn.k_proj",
                    input_size=hidden,
                    output_size=key_value,
                    dtype=dtype,
                    bias=config.attention_bias,
                ),
                value=_linear(
                    state,
                    layer + "self_attn.v_proj",
                    input_size=hidden,
                    output_size=key_value,
                    dtype=dtype,
                    bias=config.attention_bias,
                ),
                output=_linear(
                    state,
                    layer + "self_attn.o_proj",
                    input_size=attention,
                    output_size=hidden,
                    dtype=dtype,
                ),
                gate=_linear(
                    state,
                    layer + "mlp.gate_proj",
                    input_size=hidden,
                    output_size=config.intermediate_size,
                    dtype=dtype,
                ),
                up=_linear(
                    state,
                    layer + "mlp.up_proj",
                    input_size=hidden,
                    output_size=config.intermediate_size,
                    dtype=dtype,
                ),
                down=_linear(
                    state,
                    layer + "mlp.down_proj",
                    input_size=config.intermediate_size,
                    output_size=hidden,
                    dtype=dtype,
                ),
            )
        )
    return RotaryDecoderTower(
        token_embedding=_array(
            state,
            prefix + "embed_tokens.weight",
            (config.vocab_size, hidden),
            dtype,
        ),
        layers=RotaryDecoderLayerStack.from_layers(tuple(layers)),
        final_norm=RMSNorm(
            _array(state, prefix + "norm.weight", (hidden,), dtype),
            config.norm_epsilon,
        ),
        config=config,
    )


def rotary_decoder_state_dict(
    model: RotaryDecoderTower,
    *,
    prefix: str,
) -> dict[str, jax.Array]:
    state = {
        prefix + "embed_tokens.weight": model.token_embedding,
        prefix + "norm.weight": model.final_norm.weight,
    }
    for index in range(model.layers.depth):
        layer = model.layers.layer(index)
        name = prefix + f"layers.{index}."
        state.update(
            {
                name + "input_layernorm.weight": layer.input_norm.weight,
                name
                + "post_attention_layernorm.weight": layer.post_attention_norm.weight,
                name + "self_attn.q_proj.weight": layer.query.weight,
                name + "self_attn.k_proj.weight": layer.key.weight,
                name + "self_attn.v_proj.weight": layer.value.weight,
                name + "self_attn.o_proj.weight": layer.output.weight,
                name + "mlp.gate_proj.weight": layer.gate.weight,
                name + "mlp.up_proj.weight": layer.up.weight,
                name + "mlp.down_proj.weight": layer.down.weight,
            }
        )
        if model.config.attention_bias:
            for projection, value in (
                ("q", layer.query),
                ("k", layer.key),
                ("v", layer.value),
            ):
                if value.bias is None:
                    raise ValueError("attention bias is configured but missing")
                state[name + f"self_attn.{projection}_proj.bias"] = value.bias
    return state


__all__ = [
    "rotary_decoder_from_state_dict",
    "rotary_decoder_state_dict",
    "rotary_decoder_weight_names",
]
