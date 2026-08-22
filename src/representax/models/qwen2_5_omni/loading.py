"""One-shot model and processor loading for Qwen2.5-Omni artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import Qwen2_5OmniCheckpointAdapter
from .config import (
    LCO_OMNI_3B_2605_MODEL_ID,
    LCO_OMNI_3B_2605_REVISION,
    NVIDIA_OMNI_EMBED_3B_MODEL_ID,
    NVIDIA_OMNI_EMBED_3B_REVISION,
)
from .model import Qwen2_5OmniEncoder
from .processing import make_qwen2_5_omni_processor


def load_qwen2_5_omni(
    model_name_or_path: str | Path = LCO_OMNI_3B_2605_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (128, 512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    audio_chunk_count_buckets: Sequence[int] = (1, 4, 16, 64, 256),
    audio_token_count_buckets: Sequence[int] = (64, 256, 1024, 4096),
    **adapter_options,
) -> tuple[Qwen2_5OmniEncoder, Processor]:
    """Resolve one immutable checkpoint and construct model plus processor once."""

    if revision is None:
        revision = (
            NVIDIA_OMNI_EMBED_3B_REVISION
            if str(model_name_or_path) == NVIDIA_OMNI_EMBED_3B_MODEL_ID
            else LCO_OMNI_3B_2605_REVISION
        )
    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    model = Qwen2_5OmniCheckpointAdapter(**adapter_options).load(
        resolved.path,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    processor = make_qwen2_5_omni_processor(
        resolved.path,
        model.config,
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
        audio_chunk_count_buckets=audio_chunk_count_buckets,
        audio_token_count_buckets=audio_token_count_buckets,
    )
    return model, processor


__all__ = ["load_qwen2_5_omni"]
