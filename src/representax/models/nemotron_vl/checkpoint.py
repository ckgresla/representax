"""Torch-free Hugging Face conversion for Llama Nemotron VL."""

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
from representax.models.decoder_checkpoint import (
    rotary_decoder_from_state_dict,
    rotary_decoder_state_dict,
    rotary_decoder_weight_names,
)
from representax.models.siglip_checkpoint import (
    siglip_vision_from_state_dict,
    siglip_vision_state_dict,
    siglip_vision_weight_names,
)
from representax.planning import RematerializationPolicy

from .config import LlamaNemotronVLConfig
from .model import (
    LlamaNemotronVLBackbone,
    LlamaNemotronVLEncoder,
    LlamaNemotronVLReranker,
    NemotronVLProjector,
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
        raise KeyError(f"Llama Nemotron VL checkpoint is missing {name}") from error
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
) -> Linear:
    return Linear(
        weight=_array(state, prefix + ".weight", (output_size, input_size), dtype),
        bias=_array(state, prefix + ".bias", (output_size,), dtype),
    )


def _prefixes(mode: str) -> tuple[str, str, str]:
    root = "model." if mode == "reranking" else ""
    return (
        root + "language_model.",
        root + "vision_model.vision_model.",
        root + "mlp1.",
    )


def nemotron_vl_weight_names(config: LlamaNemotronVLConfig) -> frozenset[str]:
    text, vision, projector = _prefixes(config.mode)
    names = set(
        rotary_decoder_weight_names(config.text, prefix=text)
        | siglip_vision_weight_names(config.vision, prefix=vision)
    )
    names.update(
        {
            projector + "0.weight",
            projector + "0.bias",
            projector + "1.weight",
            projector + "1.bias",
            projector + "3.weight",
            projector + "3.bias",
        }
    )
    if config.mode == "reranking":
        names.add("score.weight")
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class LlamaNemotronVLCheckpointAdapter:
    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: LlamaNemotronVLConfig,
        state: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/llama-nemotron-vl",
        revision: str = "local",
    ) -> LlamaNemotronVLEncoder | LlamaNemotronVLReranker:
        text_prefix, vision_prefix, projector_prefix = _prefixes(config.mode)
        expanded = config.vision.hidden_size * config.pixel_shuffle_factor**2
        backbone = LlamaNemotronVLBackbone(
            text=rotary_decoder_from_state_dict(
                config.text,
                state,
                prefix=text_prefix,
                dtype=parameter_dtype,
            ),
            vision=siglip_vision_from_state_dict(
                config.vision,
                state,
                prefix=vision_prefix,
                dtype=parameter_dtype,
            ),
            projector=NemotronVLProjector(
                norm=LayerNorm(
                    _array(
                        state,
                        projector_prefix + "0.weight",
                        (expanded,),
                        parameter_dtype,
                    ),
                    _array(
                        state,
                        projector_prefix + "0.bias",
                        (expanded,),
                        parameter_dtype,
                    ),
                    1e-5,
                ),
                input=_linear(
                    state,
                    projector_prefix + "1",
                    input_size=expanded,
                    output_size=config.text.hidden_size,
                    dtype=parameter_dtype,
                ),
                output=_linear(
                    state,
                    projector_prefix + "3",
                    input_size=config.text.hidden_size,
                    output_size=config.text.hidden_size,
                    dtype=parameter_dtype,
                ),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        )
        metadata = EncoderMetadata(
            model_id=model_id,
            revision=revision,
            output_dimension=config.output_dimension,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT, Modality.IMAGE}),
        )
        if config.mode == "embedding":
            return LlamaNemotronVLEncoder(model=backbone, metadata=metadata)
        return LlamaNemotronVLReranker(
            model=backbone,
            score=Linear(
                weight=_array(
                    state,
                    "score.weight",
                    (config.output_dimension, config.text.hidden_size),
                    jnp.float32,
                ),
                bias=None,
            ),
            metadata=metadata,
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        model_id: str = "representax/llama-nemotron-vl",
        revision: str = "local",
    ) -> LlamaNemotronVLEncoder | LlamaNemotronVLReranker:
        source = Path(checkpoint)
        config = LlamaNemotronVLConfig.from_hf_config(load_hf_config(source))
        names = nemotron_vl_weight_names(config)
        if config.mode == "reranking":
            names = names.difference({"score.weight"})
        state = load_safetensor_subset(
            source,
            names,
            dtype=parameter_dtype,
        )
        if config.mode == "reranking":
            # The upstream score head intentionally remains FP32.
            score = load_safetensor_subset(source, {"score.weight"}, dtype=jnp.float32)
            state["score.weight"] = score["score.weight"]
        return self.from_state_dict(
            config,
            state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )

    def state_dict(
        self,
        model: LlamaNemotronVLEncoder | LlamaNemotronVLReranker,
    ) -> dict[str, jax.Array]:
        text_prefix, vision_prefix, projector_prefix = _prefixes(
            model.model.config.mode
        )
        projector = model.model.projector
        state = {
            **rotary_decoder_state_dict(model.model.text, prefix=text_prefix),
            **siglip_vision_state_dict(model.model.vision, prefix=vision_prefix),
            projector_prefix + "0.weight": projector.norm.weight,
            projector_prefix + "0.bias": projector.norm.bias,
            projector_prefix + "1.weight": projector.input.weight,
            projector_prefix + "1.bias": projector.input.bias,
            projector_prefix + "3.weight": projector.output.weight,
            projector_prefix + "3.bias": projector.output.bias,
        }
        missing = [name for name, value in state.items() if value is None]
        if missing:
            raise ValueError(f"Llama Nemotron VL checkpoint requires biases: {missing}")
        output = {name: value for name, value in state.items() if value is not None}
        if isinstance(model, LlamaNemotronVLReranker):
            output["score.weight"] = model.score.weight.astype(jnp.float32)
        return output

    def save(
        self,
        model: LlamaNemotronVLEncoder | LlamaNemotronVLReranker,
        directory: str | Path,
    ) -> Path:
        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(model.model.config.to_hf_config(), indent=2, sort_keys=True)
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


__all__ = ["LlamaNemotronVLCheckpointAdapter", "nemotron_vl_weight_names"]
