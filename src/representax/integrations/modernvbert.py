"""Pinned, Torch-free ModernVBERT text integration."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from representax.models.components import AttentionImplementation
from representax.models.modernvbert import (
    MODERNVBERT_MODEL_ID,
    MODERNVBERT_REVISION,
    ModernVBERTCheckpointAdapter,
    ModernVBERTEncoder,
    ModernVBERTTextCheckpointAdapter,
    ModernVBERTTextEncoder,
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


def load_modernvbert_text_encoder(
    model_name_or_path: str | Path = MODERNVBERT_MODEL_ID,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    parameter_dtype: str = "float32",
    compute_dtype: str = "float32",
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "full",
) -> ModernVBERTTextEncoder:
    """Load a compatible ModernVBERT text path from the Hub or a local path."""

    if revision is None and str(model_name_or_path) == MODERNVBERT_MODEL_ID:
        revision = MODERNVBERT_REVISION

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=_ALLOW_PATTERNS,
    )
    return ModernVBERTTextCheckpointAdapter(
        model_id=resolved.model_id,
        revision=resolved.revision,
    ).load(
        resolved.path,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )


def load_modernvbert_encoder(
    model_name_or_path: str | Path = MODERNVBERT_MODEL_ID,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    parameter_dtype: str = "float32",
    compute_dtype: str = "float32",
    attention_implementation: AttentionImplementation = "xla",
    rematerialization: RematerializationPolicy = "full",
) -> ModernVBERTEncoder:
    """Load a complete text-image ModernVBERT from the Hub or a local path."""

    if revision is None and str(model_name_or_path) == MODERNVBERT_MODEL_ID:
        revision = MODERNVBERT_REVISION
    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        allow_patterns=_ALLOW_PATTERNS,
    )
    return ModernVBERTCheckpointAdapter(
        model_id=resolved.model_id,
        revision=resolved.revision,
    ).load(
        resolved.path,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        attention_implementation=attention_implementation,
        rematerialization=rematerialization,
    )


__all__ = ["load_modernvbert_encoder", "load_modernvbert_text_encoder"]
