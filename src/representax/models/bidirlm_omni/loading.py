"""One-shot model and processor loading for BidirLM Omni artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import BidirLMOmniCheckpointAdapter
from .config import BIDIRLM_OMNI_2_5B_MODEL_ID, BIDIRLM_OMNI_2_5B_REVISION
from .model import BidirLMOmniEncoder
from .processing import make_bidirlm_omni_processor


def load_bidirlm_omni(
    model_name_or_path: str | Path = BIDIRLM_OMNI_2_5B_MODEL_ID,
    *,
    revision: str = BIDIRLM_OMNI_2_5B_REVISION,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 1024, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    audio_chunk_buckets: Sequence[int] = (1, 4, 8, 16),
    **adapter_options,
) -> tuple[BidirLMOmniEncoder, Processor]:
    """Resolve one immutable checkpoint and construct model plus processor once."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    model = BidirLMOmniCheckpointAdapter(**adapter_options).load(
        resolved.path,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    processor = make_bidirlm_omni_processor(
        resolved.path,
        model.config,
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
        audio_chunk_buckets=audio_chunk_buckets,
    )
    return model, processor


__all__ = ["load_bidirlm_omni"]
