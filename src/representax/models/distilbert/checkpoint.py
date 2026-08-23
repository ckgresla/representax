"""Hugging Face checkpoint mapping for native DistilBERT."""

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
from representax.models.bert import (
    BertLayer,
    BertLayerStack,
    BertMLP,
    BertSelfAttention,
)
from representax.models.components import AttentionImplementation, LayerNorm, Linear
from representax.planning import RematerializationPolicy

from .config import CLIP_MULTILINGUAL_MODEL_ID, DistilBertConfig
from .model import DistilBertEmbeddings, DistilBertEncoder, DistilBertTower


def distilbert_weight_names(config: DistilBertConfig) -> frozenset[str]:
    names = {
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
    }
    for index in range(config.num_hidden_layers):
        prefix = f"transformer.layer.{index}."
        names.update(
            {
                prefix + "attention.q_lin.weight",
                prefix + "attention.q_lin.bias",
                prefix + "attention.k_lin.weight",
                prefix + "attention.k_lin.bias",
                prefix + "attention.v_lin.weight",
                prefix + "attention.v_lin.bias",
                prefix + "attention.out_lin.weight",
                prefix + "attention.out_lin.bias",
                prefix + "sa_layer_norm.weight",
                prefix + "sa_layer_norm.bias",
                prefix + "ffn.lin1.weight",
                prefix + "ffn.lin1.bias",
                prefix + "ffn.lin2.weight",
                prefix + "ffn.lin2.bias",
                prefix + "output_layer_norm.weight",
                prefix + "output_layer_norm.bias",
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
        raise KeyError(f"DistilBERT checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"DistilBERT tensor {name} has shape {value.shape}; expected {shape}"
        )
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


def _norm(
    state: Mapping[str, Any],
    prefix: str,
    *,
    hidden_size: int,
    epsilon: float,
    dtype: jnp.dtype,
) -> LayerNorm:
    return LayerNorm(
        weight=_array(state, prefix + ".weight", (hidden_size,), dtype),
        bias=_array(state, prefix + ".bias", (hidden_size,), dtype),
        epsilon=epsilon,
    )


def _required_bias(value: Linear | LayerNorm) -> jax.Array:
    if value.bias is None:
        raise ValueError("native DistilBERT tree is missing an upstream bias")
    return value.bias


@dataclass(frozen=True, slots=True)
class DistilBertCheckpointAdapter:
    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: DistilBertConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = CLIP_MULTILINGUAL_MODEL_ID,
        revision: str = "local",
    ) -> DistilBertEncoder:
        hidden = config.hidden_size
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"transformer.layer.{index}."
            layers.append(
                BertLayer(
                    attention=BertSelfAttention(
                        query=_linear(
                            state_dict,
                            prefix + "attention.q_lin",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        key=_linear(
                            state_dict,
                            prefix + "attention.k_lin",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        value=_linear(
                            state_dict,
                            prefix + "attention.v_lin",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "attention.out_lin",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                    ),
                    attention_norm=_norm(
                        state_dict,
                        prefix + "sa_layer_norm",
                        hidden_size=hidden,
                        epsilon=config.norm_epsilon,
                        dtype=parameter_dtype,
                    ),
                    mlp=BertMLP(
                        input=_linear(
                            state_dict,
                            prefix + "ffn.lin1",
                            input_size=hidden,
                            output_size=config.intermediate_size,
                            dtype=parameter_dtype,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "ffn.lin2",
                            input_size=config.intermediate_size,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                    ),
                    output_norm=_norm(
                        state_dict,
                        prefix + "output_layer_norm",
                        hidden_size=hidden,
                        epsilon=config.norm_epsilon,
                        dtype=parameter_dtype,
                    ),
                )
            )
        return DistilBertEncoder(
            tower=DistilBertTower(
                embeddings=DistilBertEmbeddings(
                    word=_array(
                        state_dict,
                        "embeddings.word_embeddings.weight",
                        (config.vocab_size, hidden),
                        parameter_dtype,
                    ),
                    position=_array(
                        state_dict,
                        "embeddings.position_embeddings.weight",
                        (config.max_position_embeddings, hidden),
                        parameter_dtype,
                    ),
                    norm=_norm(
                        state_dict,
                        "embeddings.LayerNorm",
                        hidden_size=hidden,
                        epsilon=config.norm_epsilon,
                        dtype=parameter_dtype,
                    ),
                ),
                layers=BertLayerStack.from_layers(tuple(layers)),
                config=config,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=hidden,
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
        model_id: str = CLIP_MULTILINGUAL_MODEL_ID,
        revision: str = "local",
    ) -> DistilBertEncoder:
        config = DistilBertConfig.from_hf_config(load_hf_config(checkpoint))
        state = load_safetensor_subset(
            checkpoint,
            distilbert_weight_names(config),
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

    def state_dict(self, model: DistilBertEncoder) -> dict[str, jax.Array]:
        state = {
            "embeddings.word_embeddings.weight": model.tower.embeddings.word,
            "embeddings.position_embeddings.weight": model.tower.embeddings.position,
            "embeddings.LayerNorm.weight": model.tower.embeddings.norm.weight,
            "embeddings.LayerNorm.bias": _required_bias(model.tower.embeddings.norm),
        }
        for index in range(model.tower.config.num_hidden_layers):
            layer = model.tower.layers.layer(index)
            prefix = f"transformer.layer.{index}."
            linears = {
                "attention.q_lin": layer.attention.query,
                "attention.k_lin": layer.attention.key,
                "attention.v_lin": layer.attention.value,
                "attention.out_lin": layer.attention.output,
                "ffn.lin1": layer.mlp.input,
                "ffn.lin2": layer.mlp.output,
            }
            for name, linear in linears.items():
                state[prefix + name + ".weight"] = linear.weight
                state[prefix + name + ".bias"] = _required_bias(linear)
            norms = {
                "sa_layer_norm": layer.attention_norm,
                "output_layer_norm": layer.output_norm,
            }
            for name, norm in norms.items():
                state[prefix + name + ".weight"] = norm.weight
                state[prefix + name + ".bias"] = _required_bias(norm)
        return state

    def save(self, model: DistilBertEncoder, directory: str | Path) -> Path:
        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(model.tower.config.to_hf_config(), indent=2, sort_keys=True)
            + "\n"
        )
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["DistilBertCheckpointAdapter", "distilbert_weight_names"]
