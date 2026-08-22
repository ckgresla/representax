"""One-shot model and processor loading for Qwen3-VL artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import Qwen3VLCheckpointAdapter
from .config import (
    QWEN3_VL_EMBEDDING_2B_MODEL_ID,
    QWEN3_VL_EMBEDDING_2B_REVISION,
    QWEN3_VL_RERANKER_2B_MODEL_ID,
    QWEN3_VL_RERANKER_2B_REVISION,
)
from .model import Qwen3VLEncoder, Qwen3VLReranker
from .processing import make_qwen3_vl_processor


def load_qwen3_vl_embedding(
    model_name_or_path: str | Path = QWEN3_VL_EMBEDDING_2B_MODEL_ID,
    *,
    revision: str = QWEN3_VL_EMBEDDING_2B_REVISION,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    **adapter_options,
) -> tuple[Qwen3VLEncoder, Processor]:
    """Resolve one immutable checkpoint and construct model plus processor once."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    model = Qwen3VLCheckpointAdapter(**adapter_options).load(
        resolved.path,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    processor = make_qwen3_vl_processor(
        resolved.path,
        model.config,
        mode="embedding",
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
    )
    return model, processor


def load_qwen3_vl_reranker(
    model_name_or_path: str | Path = QWEN3_VL_RERANKER_2B_MODEL_ID,
    *,
    revision: str = QWEN3_VL_RERANKER_2B_REVISION,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    **adapter_options,
) -> tuple[Qwen3VLReranker, Processor]:
    """Resolve the reranker backbone, tied score, and left-padded processor."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    model = Qwen3VLCheckpointAdapter(**adapter_options).load_reranker(
        resolved.path,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    processor = make_qwen3_vl_processor(
        resolved.path,
        model.model.config,
        mode="reranking",
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
    )
    return model, processor


__all__ = ["load_qwen3_vl_embedding", "load_qwen3_vl_reranker"]
