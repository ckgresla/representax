"""Checkpoint mapping for both released Jina Embeddings v5 Omni sizes."""

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
from representax.models.components import AttentionImplementation
from representax.models.qwen2_5_omni.checkpoint import (
    qwen2_5_omni_audio_from_state_dict,
    qwen2_5_omni_audio_state_dict,
)
from representax.models.qwen3_vl.checkpoint import (
    Qwen3VLCheckpointAdapter,
    qwen3_vl_weight_names,
)
from representax.models.qwen3_vl.config import Qwen3VLConfig
from representax.models.qwen3_vl.model import Qwen3VLEncoder
from representax.planning import RematerializationPolicy

from .config import (
    JINA_V5_SMALL_MODEL_ID,
    JINA_V5_SMALL_REVISION,
    JinaV5OmniConfig,
)
from .omni import JinaV5OmniEncoder


def _pad_token_id(checkpoint: str | Path) -> int:
    root = Path(checkpoint)
    tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
    pad_token = tokenizer_config.get("pad_token")
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    for token in tokenizer.get("added_tokens", ()):
        if token.get("content") == pad_token:
            return int(token["id"])
    vocabulary = tokenizer.get("model", {}).get("vocab", {})
    if pad_token in vocabulary:
        return int(vocabulary[pad_token])
    raise ValueError(f"Jina v5 tokenizer does not define pad token {pad_token!r}")


def _qwen3_config(config: JinaV5OmniConfig) -> Qwen3VLConfig:
    """Describe only the shared text/vision towers for their existing adapter."""

    return Qwen3VLConfig(
        text=config.text,
        vision=config.vision,
        image_token_id=0,
        video_token_id=1,
        vision_start_token_id=2,
        vision_end_token_id=3,
    )


def _source_name(config: JinaV5OmniConfig, qwen_name: str) -> str:
    if qwen_name.startswith("model.language_model."):
        return qwen_name.removeprefix("model.")
    if not qwen_name.startswith("model.visual."):
        raise ValueError(f"unexpected Qwen3-VL tensor: {qwen_name}")
    suffix = qwen_name.removeprefix("model.visual.")
    if config.variant == "small":
        return "visual." + suffix
    if suffix.startswith("merger."):
        return suffix
    return "vision_tower." + suffix


def _audio_weight_names(config: JinaV5OmniConfig) -> set[str]:
    names = {
        "audio_tower.conv1.weight",
        "audio_tower.conv1.bias",
        "audio_tower.conv2.weight",
        "audio_tower.conv2.bias",
        "audio_tower.ln_post.weight",
        "audio_tower.ln_post.bias",
        "audio_tower.audio_bos_eos_token.weight",
        "audio_projector.weight",
        "audio_projector.bias",
    }
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
    return names


def jina_v5_omni_weight_names(config: JinaV5OmniConfig) -> frozenset[str]:
    qwen = _qwen3_config(config)
    names = {_source_name(config, name) for name in qwen3_vl_weight_names(qwen)}
    names.update(_audio_weight_names(config))
    return frozenset(names)


@dataclass(frozen=True, slots=True)
class JinaV5OmniCheckpointAdapter:
    """Map Nano and Small checkpoints onto one native four-modality model."""

    attention_implementation: AttentionImplementation = "xla"
    rematerialization: RematerializationPolicy = "full"

    def from_state_dict(
        self,
        config: JinaV5OmniConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.bfloat16,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        model_id: str = JINA_V5_SMALL_MODEL_ID,
        revision: str = JINA_V5_SMALL_REVISION,
    ) -> JinaV5OmniEncoder:
        qwen_config = _qwen3_config(config)
        qwen_state = {
            name: state_dict[_source_name(config, name)]
            for name in qwen3_vl_weight_names(qwen_config)
        }
        towers = Qwen3VLCheckpointAdapter(
            attention_implementation=self.attention_implementation,
            rematerialization=self.rematerialization,
        ).from_state_dict(
            qwen_config,
            qwen_state,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=model_id,
            revision=revision,
        )
        audio = qwen2_5_omni_audio_from_state_dict(
            config.audio,
            state_dict,
            parameter_dtype=parameter_dtype,
            projection_prefix="audio_projector",
        )
        return JinaV5OmniEncoder(
            text=towers.text,
            vision=towers.vision,
            audio=audio,
            metadata=EncoderMetadata(
                model_id=model_id,
                revision=revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset(
                    {Modality.TEXT, Modality.IMAGE, Modality.AUDIO, Modality.VIDEO}
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
        model_id: str = JINA_V5_SMALL_MODEL_ID,
        revision: str = JINA_V5_SMALL_REVISION,
    ) -> JinaV5OmniEncoder:
        config = JinaV5OmniConfig.from_hf_config(load_hf_config(checkpoint))
        config = config.model_copy(
            update={
                "text": config.text.model_copy(
                    update={"pad_token_id": _pad_token_id(checkpoint)}
                )
            }
        )
        state = load_safetensor_subset(
            checkpoint,
            jina_v5_omni_weight_names(config),
            dtype=parameter_dtype,
        )
        host = jax.local_devices(backend="cpu")[0]
        with jax.default_device(host):
            return self.from_state_dict(
                config,
                state,
                parameter_dtype=parameter_dtype,
                compute_dtype=compute_dtype,
                model_id=model_id,
                revision=revision,
            )

    def state_dict(self, model: JinaV5OmniEncoder) -> dict[str, jax.Array]:
        qwen_config = _qwen3_config(model.config)
        towers = Qwen3VLEncoder(
            text=model.text,
            vision=model.vision,
            metadata=model.metadata,
            config=qwen_config,
            compute_dtype=model.compute_dtype,
            attention_implementation=model.attention_implementation,
            rematerialization=model.rematerialization,
        )
        qwen_state = Qwen3VLCheckpointAdapter().state_dict(towers)
        state = {
            _source_name(model.config, name): value
            for name, value in qwen_state.items()
        }
        state.update(
            qwen2_5_omni_audio_state_dict(
                model.audio,
                projection_prefix="audio_projector",
            )
        )
        return state

    def save(self, model: JinaV5OmniEncoder, directory: str | Path) -> Path:
        """Replace weights inside a copied upstream checkpoint."""

        from safetensors.numpy import save_file

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "config.json").is_file():
            raise FileNotFoundError(
                "Jina v5 HF export must copy its source checkpoint before saving"
            )
        save_file(
            {
                name: np.array(value, copy=True)
                for name, value in self.state_dict(model).items()
            },
            target / "model.safetensors",
        )
        return target


__all__ = ["JinaV5OmniCheckpointAdapter", "jina_v5_omni_weight_names"]
