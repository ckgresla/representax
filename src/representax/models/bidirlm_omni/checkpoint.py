"""Torch-free Hugging Face checkpoint conversion for BidirLM Omni."""

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
from representax.integrations.huggingface import (
    load_hf_config,
    load_safetensor_subset,
)
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    RMSNorm,
)
from representax.models.qwen3_vl.text import (
    Qwen3VLTextLayer,
    Qwen3VLTextLayerStack,
    Qwen3VLTextTower,
)
from representax.models.qwen3_vl.vision import (
    Qwen3VLPatchMerger,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionBlockStack,
    Qwen3VLVisionMLP,
    Qwen3VLVisionTower,
)
from representax.planning import RematerializationPolicy

from .audio import (
    BidirLMOmniAudioAttention,
    BidirLMOmniAudioLayer,
    BidirLMOmniAudioLayerStack,
    BidirLMOmniAudioTower,
    Conv2D,
)
from .config import (
    BIDIRLM_OMNI_2_5B_MODEL_ID,
    BIDIRLM_OMNI_2_5B_REVISION,
    BidirLMOmniConfig,
)
from .model import BidirLMOmniEncoder


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"BidirLM Omni checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"BidirLM Omni tensor {name} has shape {value.shape}; expected {shape}"
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


def _required_bias(layer: Linear | LayerNorm) -> jax.Array:
    if layer.bias is None:
        raise AssertionError("BidirLM Omni projection requires a bias")
    return layer.bias


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


def _merger(
    state: Mapping[str, Any],
    prefix: str,
    config: BidirLMOmniConfig,
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


def bidirlm_omni_weight_names(config: BidirLMOmniConfig) -> frozenset[str]:
    """Return exactly the source tensors executed by the native forward."""

    names = {
        "language_model.embed_tokens.weight",
        "language_model.norm.weight",
        "visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.bias",
        "visual.pos_embed.weight",
        "audio_tower.conv2d1.weight",
        "audio_tower.conv2d1.bias",
        "audio_tower.conv2d2.weight",
        "audio_tower.conv2d2.bias",
        "audio_tower.conv2d3.weight",
        "audio_tower.conv2d3.bias",
        "audio_tower.conv_out.weight",
        "audio_tower.ln_post.weight",
        "audio_tower.ln_post.bias",
        "audio_tower.proj1.weight",
        "audio_tower.proj1.bias",
        "audio_tower.proj2.weight",
        "audio_tower.proj2.bias",
    }
    for index in range(config.text.num_hidden_layers):
        prefix = f"language_model.layers.{index}."
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
        prefix = f"visual.blocks.{index}."
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
    names.update(_merger_names("visual.merger"))
    for index in range(len(config.vision.deepstack_visual_indexes)):
        names.update(_merger_names(f"visual.deepstack_merger_list.{index}"))
    for index in range(config.audio.num_hidden_layers):
        prefix = f"audio_tower.layers.{index}."
        names.update(
            prefix + suffix
            for suffix in (
                "self_attn_layer_norm.weight",
                "self_attn_layer_norm.bias",
                "final_layer_norm.weight",
                "final_layer_norm.bias",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
            )
        )
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class BidirLMOmniCheckpointAdapter:
    """Convert the pinned Hugging Face model without importing Torch."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: BidirLMOmniConfig,
        state: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = BIDIRLM_OMNI_2_5B_MODEL_ID,
        revision: str = BIDIRLM_OMNI_2_5B_REVISION,
    ) -> BidirLMOmniEncoder:
        text = config.text
        text_layers = []
        for index in range(text.num_hidden_layers):
            prefix = f"language_model.layers.{index}."
            text_layers.append(
                Qwen3VLTextLayer(
                    input_norm=RMSNorm(
                        _array(
                            state,
                            prefix + "input_layernorm.weight",
                            (text.hidden_size,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    post_attention_norm=RMSNorm(
                        _array(
                            state,
                            prefix + "post_attention_layernorm.weight",
                            (text.hidden_size,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    query=_linear(
                        state,
                        prefix + "self_attn.q_proj",
                        input_size=text.hidden_size,
                        output_size=text.num_attention_heads * text.head_dimension,
                        dtype=parameter_dtype,
                    ),
                    key=_linear(
                        state,
                        prefix + "self_attn.k_proj",
                        input_size=text.hidden_size,
                        output_size=text.num_key_value_heads * text.head_dimension,
                        dtype=parameter_dtype,
                    ),
                    value=_linear(
                        state,
                        prefix + "self_attn.v_proj",
                        input_size=text.hidden_size,
                        output_size=text.num_key_value_heads * text.head_dimension,
                        dtype=parameter_dtype,
                    ),
                    output=_linear(
                        state,
                        prefix + "self_attn.o_proj",
                        input_size=text.num_attention_heads * text.head_dimension,
                        output_size=text.hidden_size,
                        dtype=parameter_dtype,
                    ),
                    query_norm=RMSNorm(
                        _array(
                            state,
                            prefix + "self_attn.q_norm.weight",
                            (text.head_dimension,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    key_norm=RMSNorm(
                        _array(
                            state,
                            prefix + "self_attn.k_norm.weight",
                            (text.head_dimension,),
                            parameter_dtype,
                        ),
                        text.norm_epsilon,
                    ),
                    gate=_linear(
                        state,
                        prefix + "mlp.gate_proj",
                        input_size=text.hidden_size,
                        output_size=text.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    up=_linear(
                        state,
                        prefix + "mlp.up_proj",
                        input_size=text.hidden_size,
                        output_size=text.intermediate_size,
                        dtype=parameter_dtype,
                    ),
                    down=_linear(
                        state,
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
            prefix = f"visual.blocks.{index}."
            vision_blocks.append(
                Qwen3VLVisionBlock(
                    attention_norm=_layer_norm(
                        state,
                        prefix + "norm1",
                        vision.hidden_size,
                        vision.norm_epsilon,
                        parameter_dtype,
                    ),
                    attention=Qwen3VLVisionAttention(
                        qkv=_linear(
                            state,
                            prefix + "attn.qkv",
                            input_size=vision.hidden_size,
                            output_size=3 * vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        output=_linear(
                            state,
                            prefix + "attn.proj",
                            input_size=vision.hidden_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                    mlp_norm=_layer_norm(
                        state,
                        prefix + "norm2",
                        vision.hidden_size,
                        vision.norm_epsilon,
                        parameter_dtype,
                    ),
                    mlp=Qwen3VLVisionMLP(
                        up=_linear(
                            state,
                            prefix + "mlp.linear_fc1",
                            input_size=vision.hidden_size,
                            output_size=vision.intermediate_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        down=_linear(
                            state,
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
                state,
                f"visual.deepstack_merger_list.{index}",
                config,
                parameter_dtype,
                postshuffle_norm=True,
            )
            for index in range(len(vision.deepstack_visual_indexes))
        )

        audio = config.audio
        audio_layers = []
        for index in range(audio.num_hidden_layers):
            prefix = f"audio_tower.layers.{index}."
            audio_layers.append(
                BidirLMOmniAudioLayer(
                    attention_norm=_layer_norm(
                        state,
                        prefix + "self_attn_layer_norm",
                        audio.hidden_size,
                        audio.norm_epsilon,
                        parameter_dtype,
                    ),
                    attention=BidirLMOmniAudioAttention(
                        query=_linear(
                            state,
                            prefix + "self_attn.q_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        key=_linear(
                            state,
                            prefix + "self_attn.k_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        value=_linear(
                            state,
                            prefix + "self_attn.v_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        output=_linear(
                            state,
                            prefix + "self_attn.out_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                    mlp_norm=_layer_norm(
                        state,
                        prefix + "final_layer_norm",
                        audio.hidden_size,
                        audio.norm_epsilon,
                        parameter_dtype,
                    ),
                    up=_linear(
                        state,
                        prefix + "fc1",
                        input_size=audio.hidden_size,
                        output_size=audio.intermediate_size,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                    down=_linear(
                        state,
                        prefix + "fc2",
                        input_size=audio.intermediate_size,
                        output_size=audio.hidden_size,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                )
            )
        downsample = audio.downsample_hidden_size
        return BidirLMOmniEncoder(
            text=Qwen3VLTextTower(
                token_embedding=_array(
                    state,
                    "language_model.embed_tokens.weight",
                    (text.vocab_size, text.hidden_size),
                    parameter_dtype,
                ),
                layers=Qwen3VLTextLayerStack.from_layers(tuple(text_layers)),
                final_norm=RMSNorm(
                    _array(
                        state,
                        "language_model.norm.weight",
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
                        state,
                        "visual.patch_embed.proj.weight",
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
                        state,
                        "visual.patch_embed.proj.bias",
                        (vision.hidden_size,),
                        parameter_dtype,
                    ),
                ),
                position_embedding=_array(
                    state,
                    "visual.pos_embed.weight",
                    (vision.num_position_embeddings, vision.hidden_size),
                    parameter_dtype,
                ),
                blocks=Qwen3VLVisionBlockStack.from_blocks(tuple(vision_blocks)),
                merger=_merger(
                    state,
                    "visual.merger",
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
            audio=BidirLMOmniAudioTower(
                conv1=Conv2D(
                    _array(
                        state,
                        "audio_tower.conv2d1.weight",
                        (downsample, 1, 3, 3),
                        parameter_dtype,
                    ),
                    _array(
                        state,
                        "audio_tower.conv2d1.bias",
                        (downsample,),
                        parameter_dtype,
                    ),
                    2,
                ),
                conv2=Conv2D(
                    _array(
                        state,
                        "audio_tower.conv2d2.weight",
                        (downsample, downsample, 3, 3),
                        parameter_dtype,
                    ),
                    _array(
                        state,
                        "audio_tower.conv2d2.bias",
                        (downsample,),
                        parameter_dtype,
                    ),
                    2,
                ),
                conv3=Conv2D(
                    _array(
                        state,
                        "audio_tower.conv2d3.weight",
                        (downsample, downsample, 3, 3),
                        parameter_dtype,
                    ),
                    _array(
                        state,
                        "audio_tower.conv2d3.bias",
                        (downsample,),
                        parameter_dtype,
                    ),
                    2,
                ),
                convolution_projection=_linear(
                    state,
                    "audio_tower.conv_out",
                    input_size=downsample * audio.frequency_bins_after_convolution,
                    output_size=audio.hidden_size,
                    dtype=parameter_dtype,
                ),
                layers=BidirLMOmniAudioLayerStack.from_layers(tuple(audio_layers)),
                final_norm=_layer_norm(
                    state,
                    "audio_tower.ln_post",
                    audio.hidden_size,
                    audio.norm_epsilon,
                    parameter_dtype,
                ),
                projection_up=_linear(
                    state,
                    "audio_tower.proj1",
                    input_size=audio.hidden_size,
                    output_size=audio.hidden_size,
                    dtype=parameter_dtype,
                    bias=True,
                ),
                projection_down=_linear(
                    state,
                    "audio_tower.proj2",
                    input_size=audio.hidden_size,
                    output_size=audio.output_size,
                    dtype=parameter_dtype,
                    bias=True,
                ),
                config=audio,
            ),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset(
                    {Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}
                ),
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
        model_id: str = BIDIRLM_OMNI_2_5B_MODEL_ID,
        revision: str = BIDIRLM_OMNI_2_5B_REVISION,
    ) -> BidirLMOmniEncoder:
        config = BidirLMOmniConfig.from_hf_config(load_hf_config(checkpoint))
        state = load_safetensor_subset(
            checkpoint,
            bidirlm_omni_weight_names(config),
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

    def state_dict(self, model: BidirLMOmniEncoder) -> dict[str, jax.Array]:
        state: dict[str, jax.Array] = {
            "language_model.embed_tokens.weight": model.text.token_embedding,
            "language_model.norm.weight": model.text.final_norm.weight,
            "visual.patch_embed.proj.weight": (
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
            "visual.patch_embed.proj.bias": _required_bias(
                model.vision.patch_embedding
            ),
            "visual.pos_embed.weight": model.vision.position_embedding,
            "audio_tower.conv2d1.weight": model.audio.conv1.weight,
            "audio_tower.conv2d1.bias": model.audio.conv1.bias,
            "audio_tower.conv2d2.weight": model.audio.conv2.weight,
            "audio_tower.conv2d2.bias": model.audio.conv2.bias,
            "audio_tower.conv2d3.weight": model.audio.conv3.weight,
            "audio_tower.conv2d3.bias": model.audio.conv3.bias,
            "audio_tower.conv_out.weight": model.audio.convolution_projection.weight,
            "audio_tower.ln_post.weight": model.audio.final_norm.weight,
            "audio_tower.ln_post.bias": _required_bias(model.audio.final_norm),
            "audio_tower.proj1.weight": model.audio.projection_up.weight,
            "audio_tower.proj1.bias": _required_bias(model.audio.projection_up),
            "audio_tower.proj2.weight": model.audio.projection_down.weight,
            "audio_tower.proj2.bias": _required_bias(model.audio.projection_down),
        }
        for index in range(model.text.layers.depth):
            layer = model.text.layers.layer(index)
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
        for index in range(model.vision.blocks.depth):
            block = jax.tree.map(
                lambda value, index=index: value[index], model.vision.blocks.blocks
            )
            prefix = f"visual.blocks.{index}."
            state.update(
                {
                    prefix + "norm1.weight": block.attention_norm.weight,
                    prefix + "norm1.bias": _required_bias(block.attention_norm),
                    prefix + "norm2.weight": block.mlp_norm.weight,
                    prefix + "norm2.bias": _required_bias(block.mlp_norm),
                    prefix + "attn.qkv.weight": block.attention.qkv.weight,
                    prefix + "attn.qkv.bias": _required_bias(block.attention.qkv),
                    prefix + "attn.proj.weight": block.attention.output.weight,
                    prefix + "attn.proj.bias": _required_bias(block.attention.output),
                    prefix + "mlp.linear_fc1.weight": block.mlp.up.weight,
                    prefix + "mlp.linear_fc1.bias": _required_bias(block.mlp.up),
                    prefix + "mlp.linear_fc2.weight": block.mlp.down.weight,
                    prefix + "mlp.linear_fc2.bias": _required_bias(block.mlp.down),
                }
            )
        _export_merger(state, "visual.merger", model.vision.merger)
        if model.vision.deepstack_mergers is not None:
            for index in range(len(model.config.vision.deepstack_visual_indexes)):
                merger = jax.tree.map(
                    lambda value, index=index: value[index],
                    model.vision.deepstack_mergers,
                )
                _export_merger(state, f"visual.deepstack_merger_list.{index}", merger)
        for index in range(model.audio.layers.depth):
            layer = jax.tree.map(
                lambda value, index=index: value[index], model.audio.layers.layers
            )
            prefix = f"audio_tower.layers.{index}."
            state.update(
                {
                    prefix + "self_attn_layer_norm.weight": layer.attention_norm.weight,
                    prefix + "self_attn_layer_norm.bias": _required_bias(
                        layer.attention_norm
                    ),
                    prefix + "final_layer_norm.weight": layer.mlp_norm.weight,
                    prefix + "final_layer_norm.bias": _required_bias(layer.mlp_norm),
                    prefix + "self_attn.q_proj.weight": layer.attention.query.weight,
                    prefix + "self_attn.q_proj.bias": _required_bias(
                        layer.attention.query
                    ),
                    prefix + "self_attn.k_proj.weight": layer.attention.key.weight,
                    prefix + "self_attn.k_proj.bias": _required_bias(
                        layer.attention.key
                    ),
                    prefix + "self_attn.v_proj.weight": layer.attention.value.weight,
                    prefix + "self_attn.v_proj.bias": _required_bias(
                        layer.attention.value
                    ),
                    prefix + "self_attn.out_proj.weight": layer.attention.output.weight,
                    prefix + "self_attn.out_proj.bias": _required_bias(
                        layer.attention.output
                    ),
                    prefix + "fc1.weight": layer.up.weight,
                    prefix + "fc1.bias": _required_bias(layer.up),
                    prefix + "fc2.weight": layer.down.weight,
                    prefix + "fc2.bias": _required_bias(layer.down),
                }
            )
        return state

    def save(
        self,
        model: BidirLMOmniEncoder,
        directory: str | Path,
        *,
        source_checkpoint: str | Path,
    ) -> Path:
        """Export a self-contained checkpoint reloadable by the upstream runtime."""

        from safetensors.numpy import save_file

        source = Path(source_checkpoint)
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if path.name == "model.safetensors":
                continue
            destination = target / path.name
            if path.is_dir():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(path, destination)
            elif path.is_file():
                shutil.copy2(path, destination)
        exported = self.state_dict(model)
        # Preserve the source checkpoint's unreachable third DeepStack merger so
        # Transformers reports a fully initialized artifact. It never enters the
        # native or upstream executed graph for this 24-block checkpoint.
        unused_names = _merger_names("visual.deepstack_merger_list.2")
        if unused_names.isdisjoint(exported):
            exported.update(
                load_safetensor_subset(
                    source, unused_names, dtype=model.text.token_embedding.dtype
                )
            )
        save_file(
            {name: np.array(value, copy=True) for name, value in exported.items()},
            target / "model.safetensors",
        )
        return target


def _export_merger(
    state: dict[str, jax.Array],
    prefix: str,
    merger: Qwen3VLPatchMerger,
) -> None:
    state.update(
        {
            prefix + ".norm.weight": merger.norm.weight,
            prefix + ".norm.bias": _required_bias(merger.norm),
            prefix + ".linear_fc1.weight": merger.up.weight,
            prefix + ".linear_fc1.bias": _required_bias(merger.up),
            prefix + ".linear_fc2.weight": merger.down.weight,
            prefix + ".linear_fc2.bias": _required_bias(merger.down),
        }
    )


__all__ = ["BidirLMOmniCheckpointAdapter", "bidirlm_omni_weight_names"]
