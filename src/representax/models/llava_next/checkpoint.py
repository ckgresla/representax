"""Torch-free Hugging Face conversion for native LLaVA-NeXT encoders."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_hf_config, load_safetensor_subset
from representax.models.clip.checkpoint import (
    clip_vision_from_state_dict,
    clip_vision_state_dict,
)
from representax.models.components import AttentionImplementation, Linear, RMSNorm
from representax.models.decoder import (
    RotaryDecoderLayer,
    RotaryDecoderLayerStack,
    RotaryDecoderTower,
)
from representax.planning import RematerializationPolicy

from .config import LlavaNextConfig
from .model import LlavaNextEncoder, LlavaNextProjector


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"LLaVA-NeXT checkpoint is missing {name}") from error
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


def _bias(value: Linear) -> jax.Array:
    if value.bias is None:
        raise ValueError("upstream LLaVA-NeXT projection requires a bias")
    return value.bias


def _layout(checkpoint: Path) -> tuple[str, str]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        names = set(json.loads(index.read_text())["weight_map"])
    else:
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "safetensors is required for checkpoint loading"
            ) from error
        with safe_open(checkpoint / "model.safetensors", framework="np") as handle:
            names = set(handle.keys())
    text_prefix = (
        "language_model.model."
        if "language_model.model.embed_tokens.weight" in names
        else "language_model."
    )
    vision_prefix = (
        "vision_tower.vision_model."
        if "vision_tower.vision_model.embeddings.class_embedding" in names
        else "vision_tower."
    )
    return text_prefix, vision_prefix


def llava_next_weight_names(
    config: LlavaNextConfig,
    *,
    text_prefix: str,
    vision_prefix: str,
) -> frozenset[str]:
    names = {
        "image_newline",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.weight",
        text_prefix + "embed_tokens.weight",
        text_prefix + "norm.weight",
        vision_prefix + "embeddings.class_embedding",
        vision_prefix + "embeddings.patch_embedding.weight",
        vision_prefix + "embeddings.position_embedding.weight",
        vision_prefix + "pre_layrnorm.weight",
        vision_prefix + "pre_layrnorm.bias",
        vision_prefix + "post_layernorm.weight",
        vision_prefix + "post_layernorm.bias",
    }
    if config.projector_bias:
        names.update(
            {
                "multi_modal_projector.linear_1.bias",
                "multi_modal_projector.linear_2.bias",
            }
        )
    for index in range(config.text.num_hidden_layers):
        prefix = text_prefix + f"layers.{index}."
        names.update(
            prefix + suffix
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
        if config.text.attention_bias:
            names.update(
                prefix + f"self_attn.{name}_proj.bias" for name in ("q", "k", "v")
            )
    for index in range(config.vision.num_hidden_layers):
        prefix = vision_prefix + f"encoder.layers.{index}."
        names.update(
            prefix + suffix
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


def _text_from_state_dict(
    config,
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
                    _array(
                        state,
                        layer + "input_layernorm.weight",
                        (hidden,),
                        dtype,
                    ),
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


@dataclass(frozen=True, slots=True)
class LlavaNextCheckpointAdapter:
    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: LlavaNextConfig,
        state: Mapping[str, Any],
        *,
        text_prefix: str,
        vision_prefix: str,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/llava-next",
        revision: str = "local",
    ) -> LlavaNextEncoder:
        return LlavaNextEncoder(
            text=_text_from_state_dict(
                config.text,
                state,
                prefix=text_prefix,
                dtype=parameter_dtype,
            ),
            vision=clip_vision_from_state_dict(
                config.vision,
                state,
                prefix=vision_prefix,
                dtype=parameter_dtype,
            ),
            projector=LlavaNextProjector(
                input=_linear(
                    state,
                    "multi_modal_projector.linear_1",
                    input_size=config.vision.hidden_size,
                    output_size=config.text.hidden_size,
                    dtype=parameter_dtype,
                    bias=config.projector_bias,
                ),
                output=_linear(
                    state,
                    "multi_modal_projector.linear_2",
                    input_size=config.text.hidden_size,
                    output_size=config.text.hidden_size,
                    dtype=parameter_dtype,
                    bias=config.projector_bias,
                ),
            ),
            image_newline=_array(
                state,
                "image_newline",
                (config.text.hidden_size,),
                parameter_dtype,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/llava-next",
        revision: str = "local",
    ) -> LlavaNextEncoder:
        source = Path(checkpoint)
        config = LlavaNextConfig.from_hf_config(load_hf_config(source))
        text_prefix, vision_prefix = _layout(source)
        state = load_safetensor_subset(
            source,
            llava_next_weight_names(
                config,
                text_prefix=text_prefix,
                vision_prefix=vision_prefix,
            ),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            state,
            text_prefix=text_prefix,
            vision_prefix=vision_prefix,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )

    def state_dict(self, model: LlavaNextEncoder) -> dict[str, jax.Array]:
        prefix = "language_model."
        state = {
            "image_newline": model.image_newline,
            "multi_modal_projector.linear_1.weight": (
                model.projector.input.output_major().weight
            ),
            "multi_modal_projector.linear_2.weight": (
                model.projector.output.output_major().weight
            ),
            prefix + "embed_tokens.weight": model.text.token_embedding,
            prefix + "norm.weight": model.text.final_norm.weight,
            **clip_vision_state_dict(model.vision, prefix="vision_tower."),
        }
        if model.config.projector_bias:
            state["multi_modal_projector.linear_1.bias"] = _bias(model.projector.input)
            state["multi_modal_projector.linear_2.bias"] = _bias(model.projector.output)
        for index in range(model.text.layers.depth):
            layer = model.text.layers.layer(index)
            layer_prefix = prefix + f"layers.{index}."
            post_norm_name = layer_prefix + "post_attention_layernorm.weight"
            state.update(
                {
                    layer_prefix + "input_layernorm.weight": layer.input_norm.weight,
                    post_norm_name: layer.post_attention_norm.weight,
                    layer_prefix + "self_attn.q_proj.weight": layer.query.weight,
                    layer_prefix + "self_attn.k_proj.weight": layer.key.weight,
                    layer_prefix + "self_attn.v_proj.weight": layer.value.weight,
                    layer_prefix + "self_attn.o_proj.weight": layer.output.weight,
                    layer_prefix + "mlp.gate_proj.weight": layer.gate.weight,
                    layer_prefix + "mlp.up_proj.weight": layer.up.weight,
                    layer_prefix + "mlp.down_proj.weight": layer.down.weight,
                }
            )
            if model.config.text.attention_bias:
                state[layer_prefix + "self_attn.q_proj.bias"] = _bias(layer.query)
                state[layer_prefix + "self_attn.k_proj.bias"] = _bias(layer.key)
                state[layer_prefix + "self_attn.v_proj.bias"] = _bias(layer.value)
        return state

    def save(self, model: LlavaNextEncoder, directory: str | Path) -> Path:
        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(model.config.to_hf_config(), indent=2, sort_keys=True) + "\n"
        )
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["LlavaNextCheckpointAdapter", "llava_next_weight_names"]
