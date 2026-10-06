"""Bidirectional Hugging Face checkpoint mapping for native BERT."""

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
from representax.integrations.huggingface import (
    load_hf_config,
    load_safetensor_subset,
    safetensor_names,
)
from representax.models.components import AttentionImplementation, LayerNorm, Linear
from representax.planning import RematerializationPolicy

from .config import BERT_MODEL_ID, BertConfig
from .model import (
    BertEmbeddings,
    BertEncoder,
    BertLayer,
    BertLayerStack,
    BertMLP,
    BertSelfAttention,
    BertTower,
)


def bert_weight_names(config: BertConfig) -> frozenset[str]:
    """Return every upstream tensor consumed by the native BERT base model."""

    names = {
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "pooler.dense.weight",
        "pooler.dense.bias",
    }
    for index in range(config.num_hidden_layers):
        prefix = f"encoder.layer.{index}."
        names.update(
            {
                prefix + "attention.self.query.weight",
                prefix + "attention.self.query.bias",
                prefix + "attention.self.key.weight",
                prefix + "attention.self.key.bias",
                prefix + "attention.self.value.weight",
                prefix + "attention.self.value.bias",
                prefix + "attention.output.dense.weight",
                prefix + "attention.output.dense.bias",
                prefix + "attention.output.LayerNorm.weight",
                prefix + "attention.output.LayerNorm.bias",
                prefix + "intermediate.dense.weight",
                prefix + "intermediate.dense.bias",
                prefix + "output.dense.weight",
                prefix + "output.dense.bias",
                prefix + "output.LayerNorm.weight",
                prefix + "output.LayerNorm.bias",
            }
        )
    return frozenset(names)


def _checkpoint_name_map(
    config: BertConfig,
    available: frozenset[str],
) -> dict[str, str]:
    mapping = {}
    for name in bert_weight_names(config):
        source = name
        if source not in available and ".LayerNorm." in name:
            suffix = ".gamma" if name.endswith(".weight") else ".beta"
            source = name.rsplit(".", 1)[0] + suffix
        if source not in available:
            raise KeyError(f"BERT checkpoint is missing {name}")
        mapping[name] = source
    return mapping


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"BERT checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"BERT tensor {name} has shape {value.shape}; expected {shape}"
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
    config: BertConfig,
    dtype: jnp.dtype,
) -> LayerNorm:
    return LayerNorm(
        weight=_array(state, prefix + ".weight", (config.hidden_size,), dtype),
        bias=_array(state, prefix + ".bias", (config.hidden_size,), dtype),
        epsilon=config.norm_epsilon,
    )


@dataclass(frozen=True, slots=True)
class BertCheckpointAdapter:
    """Load and export the standard bidirectional BERT base-model tree."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: BertConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = BERT_MODEL_ID,
        revision: str = "local",
    ) -> BertEncoder:
        hidden = config.hidden_size
        layers = []
        for index in range(config.num_hidden_layers):
            prefix = f"encoder.layer.{index}."
            layers.append(
                BertLayer(
                    attention=BertSelfAttention.from_projections(
                        _linear(
                            state_dict,
                            prefix + "attention.self.query",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.self.key",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.self.value",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                        _linear(
                            state_dict,
                            prefix + "attention.output.dense",
                            input_size=hidden,
                            output_size=hidden,
                            dtype=parameter_dtype,
                        ),
                    ),
                    attention_norm=_norm(
                        state_dict,
                        prefix + "attention.output.LayerNorm",
                        config=config,
                        dtype=parameter_dtype,
                    ),
                    mlp=BertMLP(
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
        tower = BertTower(
            embeddings=BertEmbeddings(
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
                token_type=_array(
                    state_dict,
                    "embeddings.token_type_embeddings.weight",
                    (config.type_vocab_size, hidden),
                    parameter_dtype,
                ),
                norm=_norm(
                    state_dict,
                    "embeddings.LayerNorm",
                    config=config,
                    dtype=parameter_dtype,
                ),
            ),
            layers=BertLayerStack.from_layers(tuple(layers)),
            pooler=_linear(
                state_dict,
                "pooler.dense",
                input_size=hidden,
                output_size=hidden,
                dtype=parameter_dtype,
            ),
            config=config,
        )
        return BertEncoder(
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
    ) -> BertEncoder:
        values = load_hf_config(checkpoint)
        config = BertConfig.from_hf_config(values)
        names = _checkpoint_name_map(config, safetensor_names(checkpoint))
        loaded = load_safetensor_subset(
            checkpoint,
            frozenset(names.values()),
            dtype=parameter_dtype,
        )
        state = {name: loaded[source] for name, source in names.items()}
        return self.from_state_dict(
            config,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=str(values.get("_name_or_path", BERT_MODEL_ID)),
            revision=str(values.get("_commit_hash", "local")),
        )

    def state_dict(self, model: BertEncoder) -> dict[str, jax.Array]:
        config = model.tower.config
        embeddings = model.tower.embeddings
        if embeddings.norm.bias is None or model.tower.pooler.bias is None:
            raise ValueError("native BERT tree is missing an upstream bias")
        state: dict[str, jax.Array] = {
            "embeddings.word_embeddings.weight": embeddings.word,
            "embeddings.position_embeddings.weight": embeddings.position,
            "embeddings.token_type_embeddings.weight": embeddings.token_type,
            "embeddings.LayerNorm.weight": embeddings.norm.weight,
            "embeddings.LayerNorm.bias": embeddings.norm.bias,
            "pooler.dense.weight": model.tower.pooler.weight,
            "pooler.dense.bias": model.tower.pooler.bias,
        }
        for index in range(config.num_hidden_layers):
            layer = model.tower.layers.layer(index)
            prefix = f"encoder.layer.{index}."
            linears = {
                "attention.self.query": layer.attention.query,
                "attention.self.key": layer.attention.key,
                "attention.self.value": layer.attention.value,
                "attention.output.dense": layer.attention.output,
                "intermediate.dense": layer.mlp.input,
                "output.dense": layer.mlp.output,
            }
            for name, linear in linears.items():
                if linear.bias is None:
                    raise ValueError("native BERT tree is missing an upstream bias")
                state[prefix + name + ".weight"] = linear.output_major().weight
                state[prefix + name + ".bias"] = linear.bias
            norms = {
                "attention.output.LayerNorm": layer.attention_norm,
                "output.LayerNorm": layer.output_norm,
            }
            for name, norm in norms.items():
                if norm.bias is None:
                    raise ValueError("native BERT tree is missing an upstream bias")
                state[prefix + name + ".weight"] = norm.weight
                state[prefix + name + ".bias"] = norm.bias
        return state

    def save(self, model: BertEncoder, directory: str | Path) -> Path:
        """Export a native BERT tree as a reloadable Transformers checkpoint."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "config.json"
        config_path.write_text(
            json.dumps(model.tower.config.to_hf_config(), indent=2, sort_keys=True)
            + "\n"
        )
        state = {
            name: np.asarray(value) for name, value in self.state_dict(model).items()
        }
        save_file(state, target / "model.safetensors")
        return target


__all__ = ["BertCheckpointAdapter", "bert_weight_names"]
