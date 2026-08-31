"""Pinned, Torch-free Jina Embeddings v5 integration."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from representax.models.components import AttentionImplementation
from representax.models.jina_v5 import (
    JINA_V5_SMALL_MODEL_ID,
    JINA_V5_SMALL_REVISION,
    JinaV5TextCheckpointAdapter,
    JinaV5TextEncoder,
)
from representax.planning import RematerializationPolicy

from .huggingface import resolve_hf_checkpoint

_ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.*",
    "merges.txt",
)


def load_jina_v5_text_encoder(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    parameter_dtype: str = "bfloat16",
    compute_dtype: str = "bfloat16",
    output_dimension: int | None = None,
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "none",
) -> JinaV5TextEncoder:
    """Load a compatible Jina v5 text checkpoint from the Hub or a local path."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=_ALLOW_PATTERNS,
    )
    return JinaV5TextCheckpointAdapter(
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    ).load(
        resolved.path,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        model_id=resolved.model_id,
        revision=resolved.revision,
        output_dimension=output_dimension,
    )


def load_jina_v5_small_text_encoder(
    model_name_or_path: str | Path = JINA_V5_SMALL_MODEL_ID,
    *,
    revision: str = JINA_V5_SMALL_REVISION,
    local_files_only: bool = False,
    parameter_dtype: str = "bfloat16",
    compute_dtype: str = "bfloat16",
    output_dimension: int = 1024,
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "none",
) -> JinaV5TextEncoder:
    """Load the paper-pinned Jina v5 Omni Small text checkpoint."""

    return load_jina_v5_text_encoder(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        output_dimension=output_dimension,
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )


__all__ = ["load_jina_v5_small_text_encoder", "load_jina_v5_text_encoder"]
