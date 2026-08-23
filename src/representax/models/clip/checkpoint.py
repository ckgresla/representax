"""Exact Hugging Face Safetensors mapping for native CLIP models."""

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

from .config import CLIPConfig
from .model import (
    CLIPMLP,
    CLIPAttention,
    CLIPEncoder,
    CLIPLayer,
    CLIPLayerStack,
    CLIPTextTower,
    CLIPVisionTower,
)


def _required_bias(layer: Linear | LayerNorm) -> jax.Array:
    if layer.bias is None:
        raise ValueError("native CLIP tree is missing an upstream bias")
    return layer.bias


def clip_checkpoint_directory(checkpoint: str | Path) -> Path:
    """Resolve ordinary and legacy Sentence Transformers CLIP layouts."""

    root = Path(checkpoint)
    if (root / "config.json").is_file():
        return root
    nested = root / "0_CLIPModel"
    if (nested / "config.json").is_file():
        return nested
    raise FileNotFoundError(f"no CLIP config under {root}")


def clip_weight_names(config: CLIPConfig) -> frozenset[str]:
    names = {
        "logit_scale",
        "text_projection.weight",
        "visual_projection.weight",
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.pre_layrnorm.weight",
        "vision_model.pre_layrnorm.bias",
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
    }
    for tower, depth in (
        ("text_model", config.text.num_hidden_layers),
        ("vision_model", config.vision.num_hidden_layers),
    ):
        for index in range(depth):
            prefix = f"{tower}.encoder.layers.{index}."
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


def _array(
    state: Mapping[str, Any],
    name: str,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
) -> jax.Array:
    try:
        value = jnp.asarray(state[name], dtype=dtype)
    except KeyError as error:
        raise KeyError(f"CLIP checkpoint is missing {name}") from error
    if value.shape != shape:
        raise ValueError(
            f"CLIP tensor {name} has shape {value.shape}; expected {shape}"
        )
    return value


def _linear(
    state: Mapping[str, Any],
    prefix: str,
    *,
    input_size: int,
    output_size: int,
    dtype: jnp.dtype,
    bias: bool,
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


def _norm(
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


def _layer(
    state: Mapping[str, Any],
    prefix: str,
    *,
    hidden_size: int,
    intermediate_size: int,
    epsilon: float,
    dtype: jnp.dtype,
) -> CLIPLayer:
    return CLIPLayer(
        attention_norm=_norm(
            state,
            prefix + "layer_norm1",
            hidden_size,
            epsilon,
            dtype,
        ),
        attention=CLIPAttention(
            query=_linear(
                state,
                prefix + "self_attn.q_proj",
                input_size=hidden_size,
                output_size=hidden_size,
                dtype=dtype,
                bias=True,
            ),
            key=_linear(
                state,
                prefix + "self_attn.k_proj",
                input_size=hidden_size,
                output_size=hidden_size,
                dtype=dtype,
                bias=True,
            ),
            value=_linear(
                state,
                prefix + "self_attn.v_proj",
                input_size=hidden_size,
                output_size=hidden_size,
                dtype=dtype,
                bias=True,
            ),
            output=_linear(
                state,
                prefix + "self_attn.out_proj",
                input_size=hidden_size,
                output_size=hidden_size,
                dtype=dtype,
                bias=True,
            ),
        ),
        mlp_norm=_norm(
            state,
            prefix + "layer_norm2",
            hidden_size,
            epsilon,
            dtype,
        ),
        mlp=CLIPMLP(
            up=_linear(
                state,
                prefix + "mlp.fc1",
                input_size=hidden_size,
                output_size=intermediate_size,
                dtype=dtype,
                bias=True,
            ),
            down=_linear(
                state,
                prefix + "mlp.fc2",
                input_size=intermediate_size,
                output_size=hidden_size,
                dtype=dtype,
                bias=True,
            ),
        ),
    )


def _layers(
    state: Mapping[str, Any],
    tower: str,
    *,
    depth: int,
    hidden_size: int,
    intermediate_size: int,
    epsilon: float,
    dtype: jnp.dtype,
) -> CLIPLayerStack:
    return CLIPLayerStack.from_layers(
        tuple(
            _layer(
                state,
                f"{tower}.encoder.layers.{index}.",
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                epsilon=epsilon,
                dtype=dtype,
            )
            for index in range(depth)
        )
    )


def clip_vision_from_state_dict(
    config,
    state: Mapping[str, Any],
    *,
    prefix: str = "vision_model.",
    dtype: jnp.dtype = jnp.float32,
) -> CLIPVisionTower:
    """Construct the reusable CLIP vision tower from a prefixed state dict."""

    return CLIPVisionTower(
        patch_weight=_array(
            state,
            prefix + "embeddings.patch_embedding.weight",
            (
                config.hidden_size,
                config.num_channels,
                config.patch_size,
                config.patch_size,
            ),
            dtype,
        ),
        class_embedding=_array(
            state,
            prefix + "embeddings.class_embedding",
            (config.hidden_size,),
            dtype,
        ),
        position_embedding=_array(
            state,
            prefix + "embeddings.position_embedding.weight",
            (config.patch_count + 1, config.hidden_size),
            dtype,
        ),
        pre_norm=_norm(
            state,
            prefix + "pre_layrnorm",
            config.hidden_size,
            config.layer_norm_epsilon,
            dtype,
        ),
        layers=_layers(
            state,
            prefix.removesuffix("."),
            depth=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            epsilon=config.layer_norm_epsilon,
            dtype=dtype,
        ),
        post_norm=_norm(
            state,
            prefix + "post_layernorm",
            config.hidden_size,
            config.layer_norm_epsilon,
            dtype,
        ),
    )


def clip_vision_state_dict(
    tower: CLIPVisionTower,
    *,
    prefix: str = "vision_model.",
) -> dict[str, jax.Array]:
    """Export one native CLIP vision tower under an arbitrary HF prefix."""

    state: dict[str, jax.Array] = {
        prefix + "embeddings.class_embedding": tower.class_embedding,
        prefix + "embeddings.patch_embedding.weight": tower.patch_weight,
        prefix + "embeddings.position_embedding.weight": tower.position_embedding,
        prefix + "pre_layrnorm.weight": tower.pre_norm.weight,
        prefix + "pre_layrnorm.bias": _required_bias(tower.pre_norm),
        prefix + "post_layernorm.weight": tower.post_norm.weight,
        prefix + "post_layernorm.bias": _required_bias(tower.post_norm),
    }
    for index in range(tower.layers.depth):
        layer = tower.layers.layer(index)
        layer_prefix = prefix + f"encoder.layers.{index}."
        state.update(
            {
                layer_prefix + "layer_norm1.weight": layer.attention_norm.weight,
                layer_prefix + "layer_norm1.bias": _required_bias(layer.attention_norm),
                layer_prefix + "layer_norm2.weight": layer.mlp_norm.weight,
                layer_prefix + "layer_norm2.bias": _required_bias(layer.mlp_norm),
                layer_prefix + "self_attn.q_proj.weight": layer.attention.query.weight,
                layer_prefix + "self_attn.q_proj.bias": _required_bias(
                    layer.attention.query
                ),
                layer_prefix + "self_attn.k_proj.weight": layer.attention.key.weight,
                layer_prefix + "self_attn.k_proj.bias": _required_bias(
                    layer.attention.key
                ),
                layer_prefix + "self_attn.v_proj.weight": layer.attention.value.weight,
                layer_prefix + "self_attn.v_proj.bias": _required_bias(
                    layer.attention.value
                ),
                layer_prefix
                + "self_attn.out_proj.weight": layer.attention.output.weight,
                layer_prefix + "self_attn.out_proj.bias": _required_bias(
                    layer.attention.output
                ),
                layer_prefix + "mlp.fc1.weight": layer.mlp.up.weight,
                layer_prefix + "mlp.fc1.bias": _required_bias(layer.mlp.up),
                layer_prefix + "mlp.fc2.weight": layer.mlp.down.weight,
                layer_prefix + "mlp.fc2.bias": _required_bias(layer.mlp.down),
            }
        )
    return state


@dataclass(frozen=True, slots=True)
class CLIPCheckpointAdapter:
    """Bidirectional conversion between Hugging Face and native CLIP trees."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"
    normalize_output: bool = False

    def from_state_dict(
        self,
        config: CLIPConfig,
        state: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/clip",
        revision: str = "local",
    ) -> CLIPEncoder:
        text = config.text
        vision = config.vision
        return CLIPEncoder(
            text=CLIPTextTower(
                token_embedding=_array(
                    state,
                    "text_model.embeddings.token_embedding.weight",
                    (text.vocab_size, text.hidden_size),
                    parameter_dtype,
                ),
                position_embedding=_array(
                    state,
                    "text_model.embeddings.position_embedding.weight",
                    (text.max_position_embeddings, text.hidden_size),
                    parameter_dtype,
                ),
                layers=_layers(
                    state,
                    "text_model",
                    depth=text.num_hidden_layers,
                    hidden_size=text.hidden_size,
                    intermediate_size=text.intermediate_size,
                    epsilon=text.layer_norm_epsilon,
                    dtype=parameter_dtype,
                ),
                final_norm=_norm(
                    state,
                    "text_model.final_layer_norm",
                    text.hidden_size,
                    text.layer_norm_epsilon,
                    parameter_dtype,
                ),
            ),
            vision=CLIPVisionTower(
                patch_weight=_array(
                    state,
                    "vision_model.embeddings.patch_embedding.weight",
                    (
                        vision.hidden_size,
                        vision.num_channels,
                        vision.patch_size,
                        vision.patch_size,
                    ),
                    parameter_dtype,
                ),
                class_embedding=_array(
                    state,
                    "vision_model.embeddings.class_embedding",
                    (vision.hidden_size,),
                    parameter_dtype,
                ),
                position_embedding=_array(
                    state,
                    "vision_model.embeddings.position_embedding.weight",
                    (vision.patch_count + 1, vision.hidden_size),
                    parameter_dtype,
                ),
                pre_norm=_norm(
                    state,
                    "vision_model.pre_layrnorm",
                    vision.hidden_size,
                    vision.layer_norm_epsilon,
                    parameter_dtype,
                ),
                layers=_layers(
                    state,
                    "vision_model",
                    depth=vision.num_hidden_layers,
                    hidden_size=vision.hidden_size,
                    intermediate_size=vision.intermediate_size,
                    epsilon=vision.layer_norm_epsilon,
                    dtype=parameter_dtype,
                ),
                post_norm=_norm(
                    state,
                    "vision_model.post_layernorm",
                    vision.hidden_size,
                    vision.layer_norm_epsilon,
                    parameter_dtype,
                ),
            ),
            text_projection=_linear(
                state,
                "text_projection",
                input_size=text.hidden_size,
                output_size=config.projection_dimension,
                dtype=parameter_dtype,
                bias=False,
            ),
            vision_projection=_linear(
                state,
                "visual_projection",
                input_size=vision.hidden_size,
                output_size=config.projection_dimension,
                dtype=parameter_dtype,
                bias=False,
            ),
            logit_scale=_array(state, "logit_scale", (), parameter_dtype),
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.projection_dimension,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
            ),
            config=config,
            normalize_output=self.normalize_output,
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
        model_id: str = "representax/clip",
        revision: str = "local",
    ) -> CLIPEncoder:
        source = clip_checkpoint_directory(checkpoint)
        config = CLIPConfig.from_hf_config(load_hf_config(source))
        state = load_safetensor_subset(
            source,
            clip_weight_names(config),
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

    def state_dict(self, model: CLIPEncoder) -> dict[str, jax.Array]:
        state: dict[str, jax.Array] = {
            "logit_scale": model.logit_scale,
            "text_projection.weight": model.text_projection.weight,
            "visual_projection.weight": model.vision_projection.weight,
            "text_model.embeddings.token_embedding.weight": model.text.token_embedding,
            "text_model.embeddings.position_embedding.weight": (
                model.text.position_embedding
            ),
            "text_model.final_layer_norm.weight": model.text.final_norm.weight,
            "text_model.final_layer_norm.bias": _required_bias(model.text.final_norm),
            "vision_model.embeddings.class_embedding": model.vision.class_embedding,
            "vision_model.embeddings.patch_embedding.weight": model.vision.patch_weight,
            "vision_model.embeddings.position_embedding.weight": (
                model.vision.position_embedding
            ),
            "vision_model.pre_layrnorm.weight": model.vision.pre_norm.weight,
            "vision_model.pre_layrnorm.bias": _required_bias(model.vision.pre_norm),
            "vision_model.post_layernorm.weight": model.vision.post_norm.weight,
            "vision_model.post_layernorm.bias": _required_bias(model.vision.post_norm),
        }
        for tower_name, stack in (
            ("text_model", model.text.layers),
            ("vision_model", model.vision.layers),
        ):
            for index in range(stack.depth):
                layer = stack.layer(index)
                prefix = f"{tower_name}.encoder.layers.{index}."
                state.update(
                    {
                        prefix + "layer_norm1.weight": layer.attention_norm.weight,
                        prefix + "layer_norm1.bias": _required_bias(
                            layer.attention_norm
                        ),
                        prefix + "layer_norm2.weight": layer.mlp_norm.weight,
                        prefix + "layer_norm2.bias": _required_bias(layer.mlp_norm),
                        prefix
                        + "self_attn.q_proj.weight": layer.attention.query.weight,
                        prefix + "self_attn.q_proj.bias": _required_bias(
                            layer.attention.query
                        ),
                        prefix + "self_attn.k_proj.weight": layer.attention.key.weight,
                        prefix + "self_attn.k_proj.bias": _required_bias(
                            layer.attention.key
                        ),
                        prefix
                        + "self_attn.v_proj.weight": layer.attention.value.weight,
                        prefix + "self_attn.v_proj.bias": _required_bias(
                            layer.attention.value
                        ),
                        prefix
                        + "self_attn.out_proj.weight": layer.attention.output.weight,
                        prefix + "self_attn.out_proj.bias": _required_bias(
                            layer.attention.output
                        ),
                        prefix + "mlp.fc1.weight": layer.mlp.up.weight,
                        prefix + "mlp.fc1.bias": _required_bias(layer.mlp.up),
                        prefix + "mlp.fc2.weight": layer.mlp.down.weight,
                        prefix + "mlp.fc2.bias": _required_bias(layer.mlp.down),
                    }
                )
        return state

    def save(self, model: CLIPEncoder, directory: str | Path) -> Path:
        from safetensors.numpy import save_file

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        try:
            target = clip_checkpoint_directory(root)
        except FileNotFoundError:
            target = root
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
        return root


__all__ = [
    "CLIPCheckpointAdapter",
    "clip_vision_from_state_dict",
    "clip_vision_state_dict",
    "clip_checkpoint_directory",
    "clip_weight_names",
]
