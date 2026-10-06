"""One-shot model and processor loading for Jina Embeddings v5 Omni."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .config import (
    JINA_V5_NANO_MODEL_ID,
    JINA_V5_NANO_REVISION,
    JINA_V5_SMALL_MODEL_ID,
    JINA_V5_SMALL_REVISION,
)
from .omni import JinaV5OmniEncoder
from .omni_checkpoint import JinaV5OmniCheckpointAdapter
from .processing import make_jina_v5_omni_processor


def _default_revision(model_name_or_path: str | Path) -> str | None:
    model = str(model_name_or_path)
    if model == JINA_V5_NANO_MODEL_ID:
        return JINA_V5_NANO_REVISION
    if model == JINA_V5_SMALL_MODEL_ID:
        return JINA_V5_SMALL_REVISION
    return None


def load_jina_v5_omni(
    model_name_or_path: str | Path = JINA_V5_SMALL_MODEL_ID,
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
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    **adapter_options,
) -> tuple[JinaV5OmniEncoder, Processor]:
    """Resolve either released size and construct its model and processor."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=_default_revision(model_name_or_path)
        if revision is None
        else revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        model = JinaV5OmniCheckpointAdapter(**adapter_options).load(
            resolved.path,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=resolved.model_id,
            revision=resolved.revision,
        )
    processor = make_jina_v5_omni_processor(
        resolved.path,
        model.config,
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
        audio_chunk_count_buckets=audio_chunk_count_buckets,
        audio_token_count_buckets=audio_token_count_buckets,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
    )
    return model, processor


__all__ = ["load_jina_v5_omni"]
