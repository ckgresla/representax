"""Exact Safetensors mapping for native Qwen3-VL models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_hf_config, load_safetensor_subset
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    RMSNorm,
)
from representax.planning import RematerializationPolicy

from .config import (
    QWEN3_VL_EMBEDDING_2B_MODEL_ID,
    QWEN3_VL_EMBEDDING_2B_REVISION,
    Qwen3VLConfig,
)
from .model import Qwen3VLEncoder, Qwen3VLReranker
from .text import Qwen3VLTextLayer, Qwen3VLTextLayerStack, Qwen3VLTextTower
from .vision import (
    Qwen3VLPatchMerger,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionBlockStack,
    Qwen3VLVisionMLP,
    Qwen3VLVisionTower,
)


def qwen3_vl_weight_names(config: Qwen3VLConfig) -> frozenset[str]:
    """Return the complete native tensor inventory for one Qwen3-VL model."""

    names = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.pos_embed.weight",
    }
    for index in range(config.text.num_hidden_layers):
        prefix = f"model.language_model.layers.{index}."
        names.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            )
        )
    for index in range(config.vision.depth):
        prefix = f"model.visual.blocks.{index}."
        names.update(
            prefix + suffix
            for suffix in (
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
            )
        )
    names.update(_merger_names("model.visual.merger"))
    for index in range(len(config.vision.deepstack_visual_indexes)):
        names.update(_merger_names(f"model.visual.deepstack_merger_list.{index}"))
    return frozenset(names)


def _merger_names(prefix: str) -> frozenset[str]:
    return frozenset(
        prefix + suffix
        for suffix in (
            ".norm.weight",
            ".norm.bias",
            ".linear_fc1.weight",
            ".linear_fc1.bias",
            ".linear_fc2.weight",
            ".linear_fc2.bias",
        )
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
        raise KeyError(f"Qwen3-VL checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"Qwen3-VL tensor {name} has shape {value.shape}; expected {shape}"
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
        weight=_array(
            state,
            prefix + ".weight",
            (output_size, input_size),
            dtype,
        ),
        bias=(_array(state, prefix + ".bias", (output_size,), dtype) if bias else None),
    )


def _layer_norm(
    state: Mapping[str, Any],
    prefix: str,
    size: int,
    epsilon: float,
    dtype: jnp.dtype,
) -> LayerNorm:
    return LayerNorm(
        weight=_array(state, prefix + ".weight", (size,), dtype),
        bias=_array(state, prefix + ".bias", (size,), dtype),
        epsilon=epsilon,
    )


def _bias(layer: Linear | LayerNorm) -> jax.Array:
    if layer.bias is None:
        raise AssertionError("Qwen3-VL vision projections always have biases")
    return layer.bias


def _merger(
    state: Mapping[str, Any],
    prefix: str,
    config: Qwen3VLConfig,
    dtype: jnp.dtype,
    *,
    postshuffle_norm: bool,
) -> Qwen3VLPatchMerger:
    merged = config.vision.hidden_size * config.vision.spatial_merge_unit
    return Qwen3VLPatchMerger(
        norm=_layer_norm(
            state,
            prefix + ".norm",
            merged if postshuffle_norm else config.vision.hidden_size,
            config.vision.norm_epsilon,
            dtype,
        ),
        up=_linear(
            state,
            prefix + ".linear_fc1",
            input_size=merged,
            output_size=merged,
            dtype=dtype,
            bias=True,
        ),
        down=_linear(
            state,
            prefix + ".linear_fc2",
            input_size=merged,
            output_size=config.vision.output_size,
            dtype=dtype,
            bias=True,
        ),
        postshuffle_norm=postshuffle_norm,
    )


@dataclass(frozen=True, slots=True)
class Qwen3VLCheckpointAdapter:
    """Convert pinned Hugging Face Qwen3-VL weights without an upstream runtime."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: Qwen3VLConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = QWEN3_VL_EMBEDDING_2B_MODEL_ID,
        revision: str = QWEN3_VL_EMBEDDING_2B_REVISION,
    ) -> Qwen3VLEncoder:
        text = config.text
        attention = text.num_attention_heads * text.head_dimension
        key_value = text.num_key_value_heads * text.head_dimension
        text_layers = []
        for index in range(text.num_hidden_layers):
            prefix = f"model.language_model.layers.{index}."
            text_layers.append(
                Qwen3VLTextLayer(
                    input_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "input_layernorm.weight",
                            (text.hidden_size,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    post_attention_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "post_attention_layernorm.weight",
                            (text.hidden_size,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    query=_linear(
                        state_dict,
                        prefix + "self_attn.q_proj",
                        input_size=text.hidden_size,
                        output_size=attention,
                        dtype=parameter_dtype,
                    ),
                    key=_linear(
                        state_dict,
                        prefix + "self_attn.k_proj",
                        input_size=text.hidden_size,
                        output_size=key_value,
                        dtype=parameter_dtype,
                    ),
                    value=_linear(
                        state_dict,
                        prefix + "self_attn.v_proj",
                        input_size=text.hidden_size,
                        output_size=key_value,
                        dtype=parameter_dtype,
                    ),
                    output=_linear(
                        state_dict,
                        prefix + "self_attn.o_proj",
                        input_size=attention,
                        output_size=text.hidden_size,
                        dtype=parameter_dtype,
                    ),
                    query_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "self_attn.q_norm.weight",
                            (text.head_dimension,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    key_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "self_attn.k_norm.weight",
                            (text.head_dimension,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    gate=_linear(
                        state_dict,
                        prefix + "mlp.gate_proj",
                        input_size=text.hidden_size,
                        output_size=text.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    up=_linear(
                        state_dict,
                        prefix + "mlp.up_proj",
                        input_size=text.hidden_size,
                        output_size=text.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    down=_linear(
                        state_dict,
                        prefix + "mlp.down_proj",
                        input_size=text.intermediate_size,
                        output_size=text.hidden_size,
                        dtype=parameter_dtype,
                    ),
                )
            )

        vision = config.vision
        vision_blocks = []
        for index in range(vision.depth):
            prefix = f"model.visual.blocks.{index}."
            vision_blocks.append(
                Qwen3VLVisionBlock(
                    attention_norm=_layer_norm(
                        state_dict,
                        prefix + "norm1",
                        vision.hidden_size,
                        vision.norm_epsilon,
                        parameter_dtype,
                    ),
                    attention=Qwen3VLVisionAttention(
                        qkv=_linear(
                            state_dict,
                            prefix + "attn.qkv",
                            input_size=vision.hidden_size,
                            output_size=3 * vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "attn.proj",
                            input_size=vision.hidden_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                    mlp_norm=_layer_norm(
                        state_dict,
                        prefix + "norm2",
                        vision.hidden_size,
                        vision.norm_epsilon,
                        parameter_dtype,
                    ),
                    mlp=Qwen3VLVisionMLP(
                        up=_linear(
                            state_dict,
                            prefix + "mlp.linear_fc1",
                            input_size=vision.hidden_size,
                            output_size=vision.intermediate_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        down=_linear(
                            state_dict,
                            prefix + "mlp.linear_fc2",
                            input_size=vision.intermediate_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                )
            )
        deepstack = tuple(
            _merger(
                state_dict,
                f"model.visual.deepstack_merger_list.{index}",
                config,
                parameter_dtype,
                postshuffle_norm=True,
            )
            for index in range(len(vision.deepstack_visual_indexes))
        )
        return Qwen3VLEncoder(
            text=Qwen3VLTextTower(
                token_embedding=_array(
                    state_dict,
                    "model.language_model.embed_tokens.weight",
                    (text.vocab_size, text.hidden_size),
                    parameter_dtype,
                ),
                layers=Qwen3VLTextLayerStack.from_layers(tuple(text_layers)),
                final_norm=RMSNorm(
                    _array(
                        state_dict,
                        "model.language_model.norm.weight",
                        (text.hidden_size,),
                        parameter_dtype,
                    ),
                    text.norm_epsilon,
                ),
                config=text,
            ),
            vision=Qwen3VLVisionTower(
                patch_embedding=Linear(
                    weight=_array(
                        state_dict,
                        "model.visual.patch_embed.proj.weight",
                        (
                            vision.hidden_size,
                            vision.in_channels,
                            vision.temporal_patch_size,
                            vision.patch_size,
                            vision.patch_size,
                        ),
                        parameter_dtype,
                    ).reshape((vision.hidden_size, vision.patch_dimension)),
                    bias=_array(
                        state_dict,
                        "model.visual.patch_embed.proj.bias",
                        (vision.hidden_size,),
                        parameter_dtype,
                    ),
                ),
                position_embedding=_array(
                    state_dict,
                    "model.visual.pos_embed.weight",
                    (vision.num_position_embeddings, vision.hidden_size),
                    parameter_dtype,
                ),
                blocks=Qwen3VLVisionBlockStack.from_blocks(tuple(vision_blocks)),
                merger=_merger(
                    state_dict,
                    "model.visual.merger",
                    config,
                    parameter_dtype,
                    postshuffle_norm=False,
                ),
                deepstack_mergers=(
                    None
                    if not deepstack
                    else jax.tree.map(lambda *values: jnp.stack(values), *deepstack)
                ),
                config=vision,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO}),
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
        model_id: str = QWEN3_VL_EMBEDDING_2B_MODEL_ID,
        revision: str = QWEN3_VL_EMBEDDING_2B_REVISION,
    ) -> Qwen3VLEncoder:
        config = Qwen3VLConfig.from_hf_config(load_hf_config(checkpoint))
        state = load_safetensor_subset(
            checkpoint,
            qwen3_vl_weight_names(config),
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

    def load_reranker(
        self,
        checkpoint: str | Path,
        **options: Any,
    ) -> Qwen3VLReranker:
        return Qwen3VLReranker(self.load(checkpoint, **options))

    def state_dict(self, model: Qwen3VLEncoder) -> dict[str, jax.Array]:
        state: dict[str, jax.Array] = {
            "model.language_model.embed_tokens.weight": model.text.token_embedding,
            "model.language_model.norm.weight": model.text.final_norm.weight,
            "model.visual.patch_embed.proj.weight": (
                model.vision.patch_embedding.weight.reshape(
                    (
                        model.config.vision.hidden_size,
                        model.config.vision.in_channels,
                        model.config.vision.temporal_patch_size,
                        model.config.vision.patch_size,
                        model.config.vision.patch_size,
                    )
                )
            ),
            "model.visual.patch_embed.proj.bias": _bias(model.vision.patch_embedding),
            "model.visual.pos_embed.weight": model.vision.position_embedding,
        }
        for index in range(model.text.layers.depth):
            layer = model.text.layers.layer(index)
            prefix = f"model.language_model.layers.{index}."
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
        for index in range(model.vision.blocks.depth):
            block = jax.tree.map(
                lambda value, index=index: value[index], model.vision.blocks.blocks
            )
            prefix = f"model.visual.blocks.{index}."
            state.update(
                {
                    prefix + "norm1.weight": block.attention_norm.weight,
                    prefix + "norm1.bias": _bias(block.attention_norm),
                    prefix + "norm2.weight": block.mlp_norm.weight,
                    prefix + "norm2.bias": _bias(block.mlp_norm),
                    prefix + "attn.qkv.weight": block.attention.qkv.weight,
                    prefix + "attn.qkv.bias": _bias(block.attention.qkv),
                    prefix + "attn.proj.weight": block.attention.output.weight,
                    prefix + "attn.proj.bias": _bias(block.attention.output),
                    prefix + "mlp.linear_fc1.weight": block.mlp.up.weight,
                    prefix + "mlp.linear_fc1.bias": _bias(block.mlp.up),
                    prefix + "mlp.linear_fc2.weight": block.mlp.down.weight,
                    prefix + "mlp.linear_fc2.bias": _bias(block.mlp.down),
                }
            )
        _export_merger(state, "model.visual.merger", model.vision.merger)
        if model.vision.deepstack_mergers is not None:
            for index in range(len(model.config.vision.deepstack_visual_indexes)):
                merger = jax.tree.map(
                    lambda value, index=index: value[index],
                    model.vision.deepstack_mergers,
                )
                _export_merger(
                    state,
                    f"model.visual.deepstack_merger_list.{index}",
                    merger,
                )
        return state


def _export_merger(
    state: dict[str, jax.Array],
    prefix: str,
    merger: Qwen3VLPatchMerger,
) -> None:
    state.update(
        {
            prefix + ".norm.weight": merger.norm.weight,
            prefix + ".norm.bias": _bias(merger.norm),
            prefix + ".linear_fc1.weight": merger.up.weight,
            prefix + ".linear_fc1.bias": _bias(merger.up),
            prefix + ".linear_fc2.weight": merger.down.weight,
            prefix + ".linear_fc2.bias": _bias(merger.down),
        }
    )


__all__ = ["Qwen3VLCheckpointAdapter", "qwen3_vl_weight_names"]
