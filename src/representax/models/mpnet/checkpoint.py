"""Bidirectional Hugging Face checkpoint mapping for native MPNet."""

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
from representax.models.components import AttentionImplementation, LayerNorm, Linear
from representax.planning import RematerializationPolicy

from .config import MPNET_MODEL_ID, MPNetConfig
from .model import (
    MPNetEmbeddings,
    MPNetEncoder,
    MPNetLayer,
    MPNetLayerStack,
    MPNetMLP,
    MPNetSelfAttention,
    MPNetTower,
)


def mpnet_weight_names(config: MPNetConfig) -> frozenset[str]:
    """Return every upstream tensor consumed by the native MPNet model."""

    names = {
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "encoder.relative_attention_bias.weight",
        "pooler.dense.weight",
        "pooler.dense.bias",
    }
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layer.{index}."
        names.update(
            {
                prefix + "attention.attn.q.weight",
                prefix + "attention.attn.q.bias",
                prefix + "attention.attn.k.weight",
                prefix + "attention.attn.k.bias",
                prefix + "attention.attn.v.weight",
                prefix + "attention.attn.v.bias",
                prefix + "attention.attn.o.weight",
                prefix + "attention.attn.o.bias",
                prefix + "attention.LayerNorm.weight",
                prefix + "attention.LayerNorm.bias",
                prefix + "intermediate.dense.weight",
                prefix + "intermediate.dense.bias",
                prefix + "output.dense.weight",
                prefix + "output.dense.bias",
                prefix + "output.LayerNorm.weight",
                prefix + "output.LayerNorm.bias",
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
        raise KeyError(f"MPNet checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"MPNet tensor {name} has shape {value.shape}; expected {shape}"
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
        weight=_array(
            state,
            prefix + ".weight",
            (output_size, input_size),
            dtype,
        ),
        bias=_array(state, prefix + ".bias", (output_size,), dtype),
    )


def _norm(
    state: Mapping[str, Any],
    prefix: str,
    *,
    config: MPNetConfig,
    dtype: jnp.dtype,
) -> LayerNorm:
    return LayerNorm(
        weight=_array(state, prefix + ".weight", (config.hidden_size,), dtype),
        bias=_array(state, prefix + ".bias", (config.hidden_size,), dtype),
        epsilon=config.norm_epsilon,
    )


@dataclass(frozen=True, slots=True)
class MPNetCheckpointAdapter:
    """Load and export the standard bidirectional MPNet base-model tree."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: MPNetConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = MPNET_MODEL_ID,
        revision: str = "local",
    ) -> MPNetEncoder:
        hidden = config.hidden_size
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"encoder.layer.{index}."
            layers.append(
                MPNetLayer(
                    attention=MPNetSelfAttention.from_projections(
                        _linear(
                            state_dict,
                            prefix + "attention.attn.q",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.attn.k",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.attn.v",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.attn.o",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                    ),
                    attention_norm=_norm(
                        state_dict,
                        prefix + "attention.LayerNorm",
                        config=config,
                        dtype=parameter_dtype,
                    ),
                    mlp=MPNetMLP(
                        input=_linear(
                            state_dict,
                            prefix + "intermediate.dense",
                            input_size=hidden,
                            output_size=config.intermediate_size,
                            dtype=parameter_dtype,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "output.dense",
                            input_size=config.intermediate_size,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                    ),
                    output_norm=_norm(
                        state_dict,
                        prefix + "output.LayerNorm",
                        config=config,
                        dtype=parameter_dtype,
                    ),
                )
            )
        tower = MPNetTower(
            embeddings=MPNetEmbeddings(
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
                    config=config,
                    dtype=parameter_dtype,
                ),
            ),
            layers=MPNetLayerStack.from_layers(tuple(layers)),
            relative_attention_bias=_array(
                state_dict,
                "encoder.relative_attention_bias.weight",
                (
                    config.relative_attention_num_buckets,
                    config.num_attention_heads,
                ),
                parameter_dtype,
            ),
            pooler=_linear(
                state_dict,
                "pooler.dense",
                input_size=hidden,
                output_size=hidden,
                dtype=parameter_dtype,
            ),
            config=config,
        )
        return MPNetEncoder(
            tower=tower,
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
    ) -> MPNetEncoder:
        values = load_hf_config(checkpoint)
        config = MPNetConfig.from_hf_config(values)
        state = load_safetensor_subset(
            checkpoint,
            mpnet_weight_names(config),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=str(values.get("_name_or_path", MPNET_MODEL_ID)),
            revision=str(values.get("_commit_hash", "local")),
        )

    def state_dict(self, model: MPNetEncoder) -> dict[str, jax.Array]:
        config = model.tower.config
        embeddings = model.tower.embeddings
        if embeddings.norm.bias is None or model.tower.pooler.bias is None:
            raise ValueError("native MPNet tree is missing an upstream bias")
        state: dict[str, jax.Array] = {
            "embeddings.word_embeddings.weight": embeddings.word,
            "embeddings.position_embeddings.weight": embeddings.position,
            "embeddings.LayerNorm.weight": embeddings.norm.weight,
            "embeddings.LayerNorm.bias": embeddings.norm.bias,
            "encoder.relative_attention_bias.weight": (
                model.tower.relative_attention_bias
            ),
            "pooler.dense.weight": model.tower.pooler.weight,
            "pooler.dense.bias": model.tower.pooler.bias,
        }
        for index in range(config.num_hidden_layers):
            layer = model.tower.layers.layer(index)
            prefix = f"encoder.layer.{index}."
            linears = {
                "attention.attn.q": layer.attention.query,
                "attention.attn.k": layer.attention.key,
                "attention.attn.v": layer.attention.value,
                "attention.attn.o": layer.attention.output,
                "intermediate.dense": layer.mlp.input,
                "output.dense": layer.mlp.output,
            }
            for name, linear in linears.items():
                if linear.bias is None:
                    raise ValueError("native MPNet tree is missing an upstream bias")
                state[prefix + name + ".weight"] = linear.output_major().weight
                state[prefix + name + ".bias"] = linear.bias
            norms = {
                "attention.LayerNorm": layer.attention_norm,
                "output.LayerNorm": layer.output_norm,
            }
            for name, norm in norms.items():
                if norm.bias is None:
                    raise ValueError("native MPNet tree is missing an upstream bias")
                state[prefix + name + ".weight"] = norm.weight
                state[prefix + name + ".bias"] = norm.bias
        return state

    def save(self, model: MPNetEncoder, directory: str | Path) -> Path:
        """Export a native MPNet tree as a reloadable Transformers checkpoint."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(model.tower.config.to_hf_config(), indent=2, sort_keys=True)
            + "\n"
        )
        state = {
            name: np.asarray(value) for name, value in self.state_dict(model).items()
        }
        save_file(state, target / "model.safetensors")
        return target


__all__ = ["MPNetCheckpointAdapter", "mpnet_weight_names"]
