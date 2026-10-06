"""Checkpoint mapping for the shared native SigLIP vision tower."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from representax.models.components import LayerNorm, Linear
from representax.models.modernvbert.config import ModernVBERTVisionConfig
from representax.models.modernvbert.vision import (
    PatchEmbedding,
    SigLIPVisionAttention,
    SigLIPVisionLayer,
    SigLIPVisionMLP,
    SigLIPVisionTower,
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
        raise KeyError(f"SigLIP checkpoint is missing {name}") from error
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
) -> Linear:
    return Linear(
        weight=_array(state, prefix + ".weight", (output_size, input_size), dtype),
        bias=_array(state, prefix + ".bias", (output_size,), dtype),
    )


def siglip_vision_weight_names(
    config: ModernVBERTVisionConfig,
    *,
    prefix: str,
) -> frozenset[str]:
    names = {
        prefix + "embeddings.patch_embedding.weight",
        prefix + "embeddings.patch_embedding.bias",
        prefix + "embeddings.position_embedding.weight",
        prefix + "post_layernorm.weight",
        prefix + "post_layernorm.bias",
    }
    for index in range(config.num_hidden_layers):
        layer = prefix + f"encoder.layers.{index}."
        names.update(
            layer + suffix
            for suffix in (
                "layer_norm1.weight",
                "layer_norm1.bias",
                "layer_norm2.weight",
                "layer_norm2.bias",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "mlp.fc1.weight",
                "mlp.fc1.bias",
                "mlp.fc2.weight",
                "mlp.fc2.bias",
            )
        )
    return frozenset(names)


def siglip_vision_from_state_dict(
    config: ModernVBERTVisionConfig,
    state: Mapping[str, Any],
    *,
    prefix: str,
    dtype: jnp.dtype,
) -> SigLIPVisionTower:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    layers = []
    for index in range(config.num_hidden_layers):
        layer = prefix + f"encoder.layers.{index}."
        layers.append(
            SigLIPVisionLayer(
                attention_norm=LayerNorm(
                    _array(state, layer + "layer_norm1.weight", (hidden,), dtype),
                    _array(state, layer + "layer_norm1.bias", (hidden,), dtype),
                    config.norm_epsilon,
                ),
                attention=SigLIPVisionAttention(
                    query=_linear(
                        state,
                        layer + "self_attn.q_proj",
                        input_size=hidden,
                        output_size=hidden,
                        dtype=dtype,
                    ),
                    key=_linear(
                        state,
                        layer + "self_attn.k_proj",
                        input_size=hidden,
                        output_size=hidden,
                        dtype=dtype,
                    ),
                    value=_linear(
                        state,
                        layer + "self_attn.v_proj",
                        input_size=hidden,
                        output_size=hidden,
                        dtype=dtype,
                    ),
                    output=_linear(
                        state,
                        layer + "self_attn.out_proj",
                        input_size=hidden,
                        output_size=hidden,
                        dtype=dtype,
                    ),
                ),
                mlp_norm=LayerNorm(
                    _array(state, layer + "layer_norm2.weight", (hidden,), dtype),
                    _array(state, layer + "layer_norm2.bias", (hidden,), dtype),
                    config.norm_epsilon,
                ),
                mlp=SigLIPVisionMLP(
                    input=_linear(
                        state,
                        layer + "mlp.fc1",
                        input_size=hidden,
                        output_size=intermediate,
                        dtype=dtype,
                    ),
                    output=_linear(
                        state,
                        layer + "mlp.fc2",
                        input_size=intermediate,
                        output_size=hidden,
                        dtype=dtype,
                    ),
                ),
            )
        )
    return SigLIPVisionTower(
        patch_embedding=PatchEmbedding(
            weight=_array(
                state,
                prefix + "embeddings.patch_embedding.weight",
                (hidden, config.num_channels, config.patch_size, config.patch_size),
                dtype,
            ),
            bias=_array(
                state,
                prefix + "embeddings.patch_embedding.bias",
                (hidden,),
                dtype,
            ),
            patch_size=config.patch_size,
        ),
        position_embedding=_array(
            state,
            prefix + "embeddings.position_embedding.weight",
            (config.patch_count, hidden),
            dtype,
        ),
        layers=tuple(layers),
        final_norm=LayerNorm(
            _array(state, prefix + "post_layernorm.weight", (hidden,), dtype),
            _array(state, prefix + "post_layernorm.bias", (hidden,), dtype),
            config.norm_epsilon,
        ),
        config=config,
    )


def siglip_vision_state_dict(
    model: SigLIPVisionTower,
    *,
    prefix: str,
) -> dict[str, jax.Array]:
    state = {
        prefix + "embeddings.patch_embedding.weight": model.patch_embedding.weight,
        prefix + "embeddings.patch_embedding.bias": model.patch_embedding.bias,
        prefix + "embeddings.position_embedding.weight": model.position_embedding,
        prefix + "post_layernorm.weight": model.final_norm.weight,
        prefix + "post_layernorm.bias": model.final_norm.bias,
    }
    for index, layer in enumerate(model.layers):
        name = prefix + f"encoder.layers.{index}."
        state.update(
            {
                name + "layer_norm1.weight": layer.attention_norm.weight,
                name + "layer_norm1.bias": layer.attention_norm.bias,
                name + "layer_norm2.weight": layer.mlp_norm.weight,
                name + "layer_norm2.bias": layer.mlp_norm.bias,
                name + "self_attn.q_proj.weight": layer.attention.query.weight,
                name + "self_attn.q_proj.bias": layer.attention.query.bias,
                name + "self_attn.k_proj.weight": layer.attention.key.weight,
                name + "self_attn.k_proj.bias": layer.attention.key.bias,
                name + "self_attn.v_proj.weight": layer.attention.value.weight,
                name + "self_attn.v_proj.bias": layer.attention.value.bias,
                name + "self_attn.out_proj.weight": layer.attention.output.weight,
                name + "self_attn.out_proj.bias": layer.attention.output.bias,
                name + "mlp.fc1.weight": layer.mlp.input.weight,
                name + "mlp.fc1.bias": layer.mlp.input.bias,
                name + "mlp.fc2.weight": layer.mlp.output.weight,
                name + "mlp.fc2.bias": layer.mlp.output.bias,
            }
        )
    missing = [name for name, value in state.items() if value is None]
    if missing:
        raise ValueError(f"SigLIP checkpoint requires biases: {missing}")
    return {name: value for name, value in state.items() if value is not None}


__all__ = [
    "siglip_vision_from_state_dict",
    "siglip_vision_state_dict",
    "siglip_vision_weight_names",
]
