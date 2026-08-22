"""Exact Safetensors mapping for native Qwen2.5-Omni representation models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import load_hf_config, load_safetensor_subset
from representax.models.components import (
    AttentionImplementation,
    LayerNorm,
    Linear,
    RMSNorm,
)
from representax.planning import RematerializationPolicy

from .audio import (
    Conv1D,
    Qwen2_5OmniAudioAttention,
    Qwen2_5OmniAudioLayer,
    Qwen2_5OmniAudioLayerStack,
    Qwen2_5OmniAudioTower,
)
from .config import (
    LCO_OMNI_3B_2605_MODEL_ID,
    LCO_OMNI_3B_2605_REVISION,
    Qwen2_5OmniConfig,
)
from .model import Qwen2_5OmniEncoder
from .text import (
    Qwen2_5OmniTextLayer,
    Qwen2_5OmniTextLayerStack,
    Qwen2_5OmniTextTower,
)
from .vision import (
    Qwen2_5OmniPatchMerger,
    Qwen2_5OmniVisionAttention,
    Qwen2_5OmniVisionBlock,
    Qwen2_5OmniVisionBlockStack,
    Qwen2_5OmniVisionMLP,
    Qwen2_5OmniVisionTower,
)


def qwen2_5_omni_weight_names(config: Qwen2_5OmniConfig) -> frozenset[str]:
    """Return every tensor consumed by the native thinker encoder."""

    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "visual.patch_embed.proj.weight",
        "visual.merger.ln_q.weight",
        "visual.merger.mlp.0.weight",
        "visual.merger.mlp.0.bias",
        "visual.merger.mlp.2.weight",
        "visual.merger.mlp.2.bias",
        "audio_tower.conv1.weight",
        "audio_tower.conv1.bias",
        "audio_tower.conv2.weight",
        "audio_tower.conv2.bias",
        "audio_tower.ln_post.weight",
        "audio_tower.ln_post.bias",
        "audio_tower.proj.weight",
        "audio_tower.proj.bias",
        "audio_tower.audio_bos_eos_token.weight",
    }
    for index in range(config.text.num_hidden_layers):
        prefix = f"model.layers.{index}."
        names.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.weight",
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
                "norm2.weight",
                "attn.q.weight",
                "attn.q.bias",
                "attn.k.weight",
                "attn.k.bias",
                "attn.v.weight",
                "attn.v.bias",
                "attn.proj.weight",
                "attn.proj.bias",
                "mlp.gate_proj.weight",
                "mlp.gate_proj.bias",
                "mlp.up_proj.weight",
                "mlp.up_proj.bias",
                "mlp.down_proj.weight",
                "mlp.down_proj.bias",
            )
        )
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


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"Qwen2.5-Omni checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"Qwen2.5-Omni tensor {name} has shape {value.shape}; expected {shape}"
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
        raise AssertionError("checkpoint-compatible projection requires a bias")
    return layer.bias


@dataclass(frozen=True, slots=True)
class Qwen2_5OmniCheckpointAdapter:
    """Convert pinned Hugging Face thinker weights without an upstream runtime."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"
    nvidia_text_attention: Literal["causal", "bidirectional"] = "causal"

    def from_state_dict(
        self,
        config: Qwen2_5OmniConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = LCO_OMNI_3B_2605_MODEL_ID,
        revision: str = LCO_OMNI_3B_2605_REVISION,
        config_model_type: str = "qwen2_5_omni_thinker",
    ) -> Qwen2_5OmniEncoder:
        text = config.text
        attention = text.num_attention_heads * text.head_dimension
        key_value = text.num_key_value_heads * text.head_dimension
        text_layers = []
        for index in range(text.num_hidden_layers):
            prefix = f"model.layers.{index}."
            text_layers.append(
                Qwen2_5OmniTextLayer(
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
                        bias=True,
                    ),
                    key=_linear(
                        state_dict,
                        prefix + "self_attn.k_proj",
                        input_size=text.hidden_size,
                        output_size=key_value,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                    value=_linear(
                        state_dict,
                        prefix + "self_attn.v_proj",
                        input_size=text.hidden_size,
                        output_size=key_value,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                    output=_linear(
                        state_dict,
                        prefix + "self_attn.o_proj",
                        input_size=attention,
                        output_size=text.hidden_size,
                        dtype=parameter_dtype,
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
        text_tower = Qwen2_5OmniTextTower(
            token_embedding=_array(
                state_dict,
                "model.embed_tokens.weight",
                (text.vocab_size, text.hidden_size),
                parameter_dtype,
            ),
            layers=Qwen2_5OmniTextLayerStack.from_layers(
                tuple(text_layers), text.layer_types
            ),
            final_norm=RMSNorm(
                _array(
                    state_dict,
                    "model.norm.weight",
                    (text.hidden_size,),
                    parameter_dtype,
                ),
                text.norm_epsilon,
            ),
            config=text,
        )

        vision = config.vision
        vision_blocks = []
        for index in range(vision.depth):
            prefix = f"visual.blocks.{index}."
            vision_blocks.append(
                Qwen2_5OmniVisionBlock(
                    attention_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "norm1.weight",
                            (vision.hidden_size,),
                            parameter_dtype,
                        ),
                        vision.norm_epsilon,
                    ),
                    attention=Qwen2_5OmniVisionAttention(
                        query=_linear(
                            state_dict,
                            prefix + "attn.q",
                            input_size=vision.hidden_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        key=_linear(
                            state_dict,
                            prefix + "attn.k",
                            input_size=vision.hidden_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        value=_linear(
                            state_dict,
                            prefix + "attn.v",
                            input_size=vision.hidden_size,
                            output_size=vision.hidden_size,
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
                    mlp_norm=RMSNorm(
                        _array(
                            state_dict,
                            prefix + "norm2.weight",
                            (vision.hidden_size,),
                            parameter_dtype,
                        ),
                        vision.norm_epsilon,
                    ),
                    mlp=Qwen2_5OmniVisionMLP(
                        gate=_linear(
                            state_dict,
                            prefix + "mlp.gate_proj",
                            input_size=vision.hidden_size,
                            output_size=vision.intermediate_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        up=_linear(
                            state_dict,
                            prefix + "mlp.up_proj",
                            input_size=vision.hidden_size,
                            output_size=vision.intermediate_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        down=_linear(
                            state_dict,
                            prefix + "mlp.down_proj",
                            input_size=vision.intermediate_size,
                            output_size=vision.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                )
            )
        merged = vision.hidden_size * vision.spatial_merge_unit
        vision_tower = Qwen2_5OmniVisionTower(
            patch_embedding=Linear(
                weight=_array(
                    state_dict,
                    "visual.patch_embed.proj.weight",
                    (
                        vision.hidden_size,
                        vision.in_channels,
                        vision.temporal_patch_size,
                        vision.patch_size,
                        vision.patch_size,
                    ),
                    parameter_dtype,
                ).reshape(vision.hidden_size, vision.patch_dimension),
                bias=None,
            ),
            blocks=Qwen2_5OmniVisionBlockStack.from_blocks(
                tuple(vision_blocks), vision.full_attention_layers
            ),
            merger=Qwen2_5OmniPatchMerger(
                norm=RMSNorm(
                    _array(
                        state_dict,
                        "visual.merger.ln_q.weight",
                        (vision.hidden_size,),
                        parameter_dtype,
                    ),
                    vision.norm_epsilon,
                ),
                up=_linear(
                    state_dict,
                    "visual.merger.mlp.0",
                    input_size=merged,
                    output_size=merged,
                    dtype=parameter_dtype,
                    bias=True,
                ),
                down=_linear(
                    state_dict,
                    "visual.merger.mlp.2",
                    input_size=merged,
                    output_size=vision.output_size,
                    dtype=parameter_dtype,
                    bias=True,
                ),
            ),
            config=vision,
        )

        audio = config.audio
        audio_layers = []
        for index in range(audio.num_hidden_layers):
            prefix = f"audio_tower.layers.{index}."
            audio_layers.append(
                Qwen2_5OmniAudioLayer(
                    attention_norm=_layer_norm(
                        state_dict,
                        prefix + "self_attn_layer_norm",
                        audio.hidden_size,
                        audio.layer_norm_epsilon,
                        parameter_dtype,
                    ),
                    attention=Qwen2_5OmniAudioAttention(
                        query=_linear(
                            state_dict,
                            prefix + "self_attn.q_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        key=_linear(
                            state_dict,
                            prefix + "self_attn.k_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                        ),
                        value=_linear(
                            state_dict,
                            prefix + "self_attn.v_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                        output=_linear(
                            state_dict,
                            prefix + "self_attn.out_proj",
                            input_size=audio.hidden_size,
                            output_size=audio.hidden_size,
                            dtype=parameter_dtype,
                            bias=True,
                        ),
                    ),
                    mlp_norm=_layer_norm(
                        state_dict,
                        prefix + "final_layer_norm",
                        audio.hidden_size,
                        audio.layer_norm_epsilon,
                        parameter_dtype,
                    ),
                    up=_linear(
                        state_dict,
                        prefix + "fc1",
                        input_size=audio.hidden_size,
                        output_size=audio.intermediate_size,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                    down=_linear(
                        state_dict,
                        prefix + "fc2",
                        input_size=audio.intermediate_size,
                        output_size=audio.hidden_size,
                        dtype=parameter_dtype,
                        bias=True,
                    ),
                )
            )
        audio_tower = Qwen2_5OmniAudioTower(
            conv1=Conv1D(
                weight=_array(
                    state_dict,
                    "audio_tower.conv1.weight",
                    (audio.hidden_size, audio.num_mel_bins, 3),
                    parameter_dtype,
                ),
                bias=_array(
                    state_dict,
                    "audio_tower.conv1.bias",
                    (audio.hidden_size,),
                    parameter_dtype,
                ),
                stride=1,
            ),
            conv2=Conv1D(
                weight=_array(
                    state_dict,
                    "audio_tower.conv2.weight",
                    (audio.hidden_size, audio.hidden_size, 3),
                    parameter_dtype,
                ),
                bias=_array(
                    state_dict,
                    "audio_tower.conv2.bias",
                    (audio.hidden_size,),
                    parameter_dtype,
                ),
                stride=2,
            ),
            layers=Qwen2_5OmniAudioLayerStack.from_layers(tuple(audio_layers)),
            final_norm=_layer_norm(
                state_dict,
                "audio_tower.ln_post",
                audio.hidden_size,
                audio.layer_norm_epsilon,
                parameter_dtype,
            ),
            projection=_linear(
                state_dict,
                "audio_tower.proj",
                input_size=audio.hidden_size,
                output_size=audio.output_size,
                dtype=parameter_dtype,
                bias=True,
            ),
            bos_eos_embedding=_array(
                state_dict,
                "audio_tower.audio_bos_eos_token.weight",
                (2, audio.output_size),
                parameter_dtype,
            ),
            config=audio,
        )

        return Qwen2_5OmniEncoder(
            text=text_tower,
            vision=vision_tower,
            audio=audio_tower,
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset(
                    {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
                ),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
            text_attention=(
                self.nvidia_text_attention
                if config_model_type == "nvomniembed"
                else "causal"
            ),
            pooling="mean" if config_model_type == "nvomniembed" else "last",
            source_model_type=(
                "nvomniembed"
                if config_model_type == "nvomniembed"
                else "qwen2_5_omni_thinker"
            ),
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = LCO_OMNI_3B_2605_MODEL_ID,
        revision: str = LCO_OMNI_3B_2605_REVISION,
    ) -> Qwen2_5OmniEncoder:
        hf_config = load_hf_config(checkpoint)
        config = Qwen2_5OmniConfig.from_hf_config(hf_config)
        state = load_safetensor_subset(
            checkpoint,
            qwen2_5_omni_weight_names(config),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
            config_model_type=str(hf_config.get("model_type", "")),
        )

    def state_dict(self, model: Qwen2_5OmniEncoder) -> dict[str, jax.Array]:
        """Map native leaves back to the source Hugging Face tensor layout."""

        state: dict[str, jax.Array] = {
            "model.embed_tokens.weight": model.text.token_embedding,
            "model.norm.weight": model.text.final_norm.weight,
            "lm_head.weight": model.text.token_embedding,
            "visual.patch_embed.proj.weight": (
                model.vision.patch_embedding.weight.reshape(
                    model.config.vision.hidden_size,
                    model.config.vision.in_channels,
                    model.config.vision.temporal_patch_size,
                    model.config.vision.patch_size,
                    model.config.vision.patch_size,
                )
            ),
            "visual.merger.ln_q.weight": model.vision.merger.norm.weight,
            "visual.merger.mlp.0.weight": model.vision.merger.up.weight,
            "visual.merger.mlp.0.bias": _required_bias(model.vision.merger.up),
            "visual.merger.mlp.2.weight": model.vision.merger.down.weight,
            "visual.merger.mlp.2.bias": _required_bias(model.vision.merger.down),
            "audio_tower.conv1.weight": model.audio.conv1.weight,
            "audio_tower.conv1.bias": model.audio.conv1.bias,
            "audio_tower.conv2.weight": model.audio.conv2.weight,
            "audio_tower.conv2.bias": model.audio.conv2.bias,
            "audio_tower.ln_post.weight": model.audio.final_norm.weight,
            "audio_tower.ln_post.bias": _required_bias(model.audio.final_norm),
            "audio_tower.proj.weight": model.audio.projection.weight,
            "audio_tower.proj.bias": _required_bias(model.audio.projection),
            "audio_tower.audio_bos_eos_token.weight": model.audio.bos_eos_embedding,
        }
        for index in range(model.text.layers.depth):
            layer = model.text.layers.layer(index)
            prefix = f"model.layers.{index}."
            state.update(
                {
                    prefix + "input_layernorm.weight": layer.input_norm.weight,
                    prefix + "post_attention_layernorm.weight": (
                        layer.post_attention_norm.weight
                    ),
                    prefix + "self_attn.q_proj.weight": layer.query.weight,
                    prefix + "self_attn.q_proj.bias": _required_bias(layer.query),
                    prefix + "self_attn.k_proj.weight": layer.key.weight,
                    prefix + "self_attn.k_proj.bias": _required_bias(layer.key),
                    prefix + "self_attn.v_proj.weight": layer.value.weight,
                    prefix + "self_attn.v_proj.bias": _required_bias(layer.value),
                    prefix + "self_attn.o_proj.weight": layer.output.weight,
                    prefix + "mlp.gate_proj.weight": layer.gate.weight,
                    prefix + "mlp.up_proj.weight": layer.up.weight,
                    prefix + "mlp.down_proj.weight": layer.down.weight,
                }
            )
        for index in range(model.vision.blocks.depth):
            block = jax.tree.map(
                lambda value, index=index: value[index],
                model.vision.blocks.blocks,
            )
            prefix = f"visual.blocks.{index}."
            state.update(
                {
                    prefix + "norm1.weight": block.attention_norm.weight,
                    prefix + "norm2.weight": block.mlp_norm.weight,
                    prefix + "attn.q.weight": block.attention.query.weight,
                    prefix + "attn.q.bias": _required_bias(block.attention.query),
                    prefix + "attn.k.weight": block.attention.key.weight,
                    prefix + "attn.k.bias": _required_bias(block.attention.key),
                    prefix + "attn.v.weight": block.attention.value.weight,
                    prefix + "attn.v.bias": _required_bias(block.attention.value),
                    prefix + "attn.proj.weight": block.attention.output.weight,
                    prefix + "attn.proj.bias": _required_bias(block.attention.output),
                    prefix + "mlp.gate_proj.weight": block.mlp.gate.weight,
                    prefix + "mlp.gate_proj.bias": _required_bias(block.mlp.gate),
                    prefix + "mlp.up_proj.weight": block.mlp.up.weight,
                    prefix + "mlp.up_proj.bias": _required_bias(block.mlp.up),
                    prefix + "mlp.down_proj.weight": block.mlp.down.weight,
                    prefix + "mlp.down_proj.bias": _required_bias(block.mlp.down),
                }
            )
        for index in range(model.audio.layers.depth):
            layer = jax.tree.map(
                lambda value, index=index: value[index],
                model.audio.layers.layers,
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

    def save(self, model: Qwen2_5OmniEncoder, directory: str | Path) -> Path:
        """Export native weights as a reloadable Transformers thinker."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        hf_config = model.config.to_hf_config()
        if model.source_model_type == "nvomniembed":
            hf_config.update(
                {
                    "architectures": ["NVOmniEmbedModel"],
                    "model_type": "nvomniembed",
                    "auto_map": {
                        "AutoConfig": "modeling_nv_omni_embed.NVOmniEmbedConfig",
                        "AutoModel": "modeling_nv_omni_embed.NVOmniEmbedModel",
                    },
                }
            )
        (target / "config.json").write_text(
            json.dumps(hf_config, indent=2, sort_keys=True) + "\n"
        )
        state = {
            name: np.array(value, copy=True)
            for name, value in self.state_dict(model).items()
        }
        save_file(state, target / "model.safetensors")
        return target


__all__ = ["Qwen2_5OmniCheckpointAdapter", "qwen2_5_omni_weight_names"]
