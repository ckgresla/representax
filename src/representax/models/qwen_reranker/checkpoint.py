"""Torch-free Hugging Face conversion for native Qwen text rerankers."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_safetensor_subset
from representax.models.components import AttentionImplementation, Linear, RMSNorm
from representax.models.qwen2_5_omni import (
    Qwen2_5OmniTextLayer,
    Qwen2_5OmniTextLayerStack,
    Qwen2_5OmniTextTower,
)
from representax.models.qwen3_vl import (
    Qwen3VLTextLayer,
    Qwen3VLTextLayerStack,
    Qwen3VLTextTower,
)
from representax.planning import RematerializationPolicy

from .config import QWEN3_RERANKER_0_6B_MODEL_ID, QwenRerankerConfig
from .model import QwenReranker


def qwen_reranker_weight_names(config: QwenRerankerConfig) -> frozenset[str]:
    """Return every upstream tensor represented by the native execution graph."""

    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if not config.tie_word_embeddings:
        names.add("lm_head.weight")
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        suffixes = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ]
        if config.generation == "qwen2":
            suffixes.extend(
                (
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                )
            )
        else:
            suffixes.extend(("self_attn.q_norm.weight", "self_attn.k_norm.weight"))
        names.update(prefix + suffix for suffix in suffixes)
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
        raise KeyError(f"Qwen reranker checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"Qwen reranker tensor {name} has shape {value.shape}; expected {shape}"
        )
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
        raise AssertionError("Qwen2 attention projections require biases")
    return value.bias


@dataclass(frozen=True, slots=True)
class QwenRerankerCheckpointAdapter:
    """Convert one Qwen2/Qwen3 causal scorer without importing Transformers."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: QwenRerankerConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = QWEN3_RERANKER_0_6B_MODEL_ID,
        revision: str = "unknown",
    ) -> QwenReranker:
        hidden = config.hidden_size
        attention = config.num_attention_heads * config.head_dimension
        key_value = config.num_key_value_heads * config.head_dimension
        if config.generation == "qwen2":
            tower_config = config.qwen2_tower_config()
            layers = []
            for index in range(config.num_hidden_layers):
                prefix = f"model.layers.{index}."
                layers.append(
                    Qwen2_5OmniTextLayer(
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
                            bias=True,
                        ),
                        key=_linear(
                            state_dict,
                            prefix + "self_attn.k_proj",
                            input_size=hidden,
                            output_size=key_value,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        value=_linear(
                            state_dict,
                            prefix + "self_attn.v_proj",
                            input_size=hidden,
                            output_size=key_value,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "self_attn.o_proj",
                            input_size=attention,
                            output_size=hidden,
                            dtype=parameter_dtype,
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
            text: Qwen2_5OmniTextTower | Qwen3VLTextTower = Qwen2_5OmniTextTower(
                token_embedding=_array(
                    state_dict,
                    "model.embed_tokens.weight",
                    (config.vocab_size, hidden),
                    parameter_dtype,
                ),
                layers=Qwen2_5OmniTextLayerStack.from_layers(
                    tuple(layers), tower_config.layer_types
                ),
                final_norm=RMSNorm(
                    _array(
                        state_dict,
                        "model.norm.weight",
                        (hidden,),
                        parameter_dtype,
                    ),
                    config.norm_epsilon,
                ),
                config=tower_config,
            )
        else:
            qwen3_config = config.qwen3_tower_config()
            qwen3_layers = []
            for index in range(config.num_hidden_layers):
                prefix = f"model.layers.{index}."
                qwen3_layers.append(
                    Qwen3VLTextLayer(
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
            text = Qwen3VLTextTower(
                token_embedding=_array(
                    state_dict,
                    "model.embed_tokens.weight",
                    (config.vocab_size, hidden),
                    parameter_dtype,
                ),
                layers=Qwen3VLTextLayerStack.from_layers(tuple(qwen3_layers)),
                final_norm=RMSNorm(
                    _array(
                        state_dict,
                        "model.norm.weight",
                        (hidden,),
                        parameter_dtype,
                    ),
                    config.norm_epsilon,
                ),
                config=qwen3_config,
            )
        lm_head = (
            None
            if config.tie_word_embeddings
            else _array(
                state_dict,
                "lm_head.weight",
                (config.vocab_size, hidden),
                parameter_dtype,
            )
        )
        return QwenReranker(
            text=text,
            lm_head=lm_head,
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=1,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
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
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = QWEN3_RERANKER_0_6B_MODEL_ID,
        revision: str = "unknown",
    ) -> QwenReranker:
        config = QwenRerankerConfig.from_checkpoint(checkpoint)
        state = load_safetensor_subset(
            checkpoint,
            qwen_reranker_weight_names(config),
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

    def state_dict(self, model: QwenReranker) -> dict[str, jax.Array]:
        state = {
            "model.embed_tokens.weight": model.text.token_embedding,
            "model.norm.weight": model.text.final_norm.weight,
        }
        if model.lm_head is not None:
            state["lm_head.weight"] = model.lm_head
        for index in range(model.config.num_hidden_layers):
            layer = model.text.layers.layer(index)
            prefix = f"model.layers.{index}."
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
                    prefix + "mlp.gate_proj.weight": layer.gate.weight,
                    prefix + "mlp.up_proj.weight": layer.up.weight,
                    prefix + "mlp.down_proj.weight": layer.down.weight,
                }
            )
            if model.config.generation == "qwen2":
                if not isinstance(layer, Qwen2_5OmniTextLayer):
                    raise TypeError("Qwen2 config requires a Qwen2 text layer")
                state.update(
                    {
                        prefix + "self_attn.q_proj.bias": _bias(layer.query),
                        prefix + "self_attn.k_proj.bias": _bias(layer.key),
                        prefix + "self_attn.v_proj.bias": _bias(layer.value),
                    }
                )
            else:
                if not isinstance(layer, Qwen3VLTextLayer):
                    raise TypeError("Qwen3 config requires a Qwen3 text layer")
                state.update(
                    {
                        prefix + "self_attn.q_norm.weight": layer.query_norm.weight,
                        prefix + "self_attn.k_norm.weight": layer.key_norm.weight,
                    }
                )
        return state

    def save(
        self,
        model: QwenReranker,
        directory: str | Path,
        *,
        source_checkpoint: str | Path,
    ) -> Path:
        """Export native weights and processor assets for upstream fresh reload."""

        from safetensors.numpy import save_file

        source = Path(source_checkpoint)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if (
                path.name == "model.safetensors"
                or path.name.startswith("model-")
                or path.name == "model.safetensors.index.json"
            ):
                continue
            destination = target / path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copy2(path, destination)
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["QwenRerankerCheckpointAdapter", "qwen_reranker_weight_names"]
