"""Checkpoint mapping for the executed Jina v5 Omni Small text tower."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_hf_config, load_safetensor_subset
from representax.models.components import AttentionImplementation, Linear, RMSNorm
from representax.planning import RematerializationPolicy

from .config import (
    JINA_V5_SMALL_MODEL_ID,
    JINA_V5_SMALL_REVISION,
    JinaV5TextConfig,
)
from .model import (
    JinaV5TextEncoder,
    JinaV5TextLayer,
    JinaV5TextLayerStack,
    JinaV5TextTower,
)


def jina_v5_text_weight_names(config: JinaV5TextConfig) -> frozenset[str]:
    """Return every upstream tensor consumed by the native text path."""

    names = {
        "language_model.embed_tokens.weight",
        "language_model.norm.weight",
    }
    for index in range(config.num_hidden_layers):
        prefix = f"language_model.layers.{index}."
        names.update(
            {
                prefix + "input_layernorm.weight",
                prefix + "post_attention_layernorm.weight",
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.o_proj.weight",
                prefix + "self_attn.q_norm.weight",
                prefix + "self_attn.k_norm.weight",
                prefix + "mlp.gate_proj.weight",
                prefix + "mlp.up_proj.weight",
                prefix + "mlp.down_proj.weight",
            }
        )
    return frozenset(names)


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"Jina v5 checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"Jina v5 tensor {name} has shape {value.shape}; expected {shape}"
        )
    return value


def _linear(
    state: Mapping[str, Any],
    name: str,
    *,
    input_size: int,
    output_size: int,
    dtype: jnp.dtype,
) -> Linear:
    return Linear(
        weight=_array(state, name + ".weight", (output_size, input_size), dtype),
        bias=None,
    )


@dataclass(frozen=True, slots=True)
class JinaV5TextCheckpointAdapter:
    """Map the pinned Jina v5 Omni Small text tensors into Equinox."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "none"

    def from_state_dict(
        self,
        config: JinaV5TextConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = JINA_V5_SMALL_MODEL_ID,
        revision: str = JINA_V5_SMALL_REVISION,
    ) -> JinaV5TextEncoder:
        hidden = config.hidden_size
        attention = config.num_attention_heads * config.head_dimension
        key_value = config.num_key_value_heads * config.head_dimension
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"language_model.layers.{index}."
            layers.append(
                JinaV5TextLayer(
                    input_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "input_layernorm.weight",
                            (hidden,),
                            parameter_dtype,
                        ),
                        config.norm_epsilon,
                    ),
                    post_attention_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "post_attention_layernorm.weight",
                            (hidden,),
                            parameter_dtype,
                        ),
                        config.norm_epsilon,
                    ),
                    query=_linear(
                        state_dict,
                        prefix + "self_attn.q_proj",
                        input_size=hidden,
                        output_size=attention,
                        dtype=parameter_dtype,
                    ),
                    key=_linear(
                        state_dict,
                        prefix + "self_attn.k_proj",
                        input_size=hidden,
                        output_size=key_value,
                        dtype=parameter_dtype,
                    ),
                    value=_linear(
                        state_dict,
                        prefix + "self_attn.v_proj",
                        input_size=hidden,
                        output_size=key_value,
                        dtype=parameter_dtype,
                    ),
                    output=_linear(
                        state_dict,
                        prefix + "self_attn.o_proj",
                        input_size=attention,
                        output_size=hidden,
                        dtype=parameter_dtype,
                    ),
                    query_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "self_attn.q_norm.weight",
                            (config.head_dimension,),
                            parameter_dtype,
                        ),
                        config.norm_epsilon,
                    ),
                    key_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "self_attn.k_norm.weight",
                            (config.head_dimension,),
                            parameter_dtype,
                        ),
                        config.norm_epsilon,
                    ),
                    gate=_linear(
                        state_dict,
                        prefix + "mlp.gate_proj",
                        input_size=hidden,
                        output_size=config.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    up=_linear(
                        state_dict,
                        prefix + "mlp.up_proj",
                        input_size=hidden,
                        output_size=config.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    down=_linear(
                        state_dict,
                        prefix + "mlp.down_proj",
                        input_size=config.intermediate_size,
                        output_size=hidden,
                        dtype=parameter_dtype,
                    ),
                )
            )
        return JinaV5TextEncoder(
            tower=JinaV5TextTower(
                token_embedding=_array(
                    state_dict,
                    "language_model.embed_tokens.weight",
                    (config.vocab_size, hidden),
                    parameter_dtype,
                ),
                layers=JinaV5TextLayerStack.from_layers(tuple(layers)),
                final_norm=RMSNorm(
                    _array(
                        state_dict,
                        "language_model.norm.weight",
                        (hidden,),
                        parameter_dtype,
                    ),
                    config.norm_epsilon,
                ),
                config=config,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.output_dimension,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
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
        model_id: str = JINA_V5_SMALL_MODEL_ID,
        revision: str = JINA_V5_SMALL_REVISION,
        output_dimension: int | None = None,
    ) -> JinaV5TextEncoder:
        values = load_hf_config(checkpoint)
        config = JinaV5TextConfig.from_hf_config(
            values,
            output_dimension=output_dimension,
        )
        state = load_safetensor_subset(
            checkpoint,
            jina_v5_text_weight_names(config),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )

    def state_dict(self, model: JinaV5TextEncoder) -> dict[str, jax.Array]:
        state = {
            "language_model.embed_tokens.weight": model.tower.token_embedding,
            "language_model.norm.weight": model.tower.final_norm.weight,
        }
        for index in range(model.tower.layers.depth):
            layer = model.tower.layers.layer(index)
            prefix = f"language_model.layers.{index}."
            state.update(
                {
                    prefix + "input_layernorm.weight": layer.input_norm.weight,
                    prefix + "post_attention_layernorm.weight": (
                        layer.post_attention_norm.weight
                    ),
                    prefix + "self_attn.q_proj.weight": layer.query.weight,
                    prefix + "self_attn.k_proj.weight": layer.key.weight,
                    prefix + "self_attn.v_proj.weight": layer.value.weight,
                    prefix + "self_attn.o_proj.weight": layer.output.weight,
                    prefix + "self_attn.q_norm.weight": layer.query_norm.weight,
                    prefix + "self_attn.k_norm.weight": layer.key_norm.weight,
                    prefix + "mlp.gate_proj.weight": layer.gate.weight,
                    prefix + "mlp.up_proj.weight": layer.up.weight,
                    prefix + "mlp.down_proj.weight": layer.down.weight,
                }
            )
        return state


__all__ = ["JinaV5TextCheckpointAdapter", "jina_v5_text_weight_names"]
