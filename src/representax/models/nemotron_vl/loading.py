"""One-shot native Llama Nemotron VL model and processor loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import LlamaNemotronVLCheckpointAdapter
from .config import (
    LLAMA_NEMOTRON_EMBED_VL_MODEL_ID,
    LLAMA_NEMOTRON_EMBED_VL_REVISION,
    LLAMA_NEMOTRON_RERANK_VL_MODEL_ID,
    LLAMA_NEMOTRON_RERANK_VL_REVISION,
)
from .model import LlamaNemotronVLEncoder, LlamaNemotronVLReranker
from .processing import make_nemotron_vl_processor

_REVISIONS = {
    LLAMA_NEMOTRON_EMBED_VL_MODEL_ID: LLAMA_NEMOTRON_EMBED_VL_REVISION,
    LLAMA_NEMOTRON_RERANK_VL_MODEL_ID: LLAMA_NEMOTRON_RERANK_VL_REVISION,
}


def load_nemotron_vl(
    model_name_or_path: str | Path = LLAMA_NEMOTRON_EMBED_VL_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 1024, 2048, 4096, 8192),
    tile_count_buckets: Sequence[int] = (1, 3, 7, 14, 28, 56),
) -> tuple[LlamaNemotronVLEncoder | LlamaNemotronVLReranker, Processor]:
    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision or _REVISIONS.get(str(model_name_or_path)),
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        model = LlamaNemotronVLCheckpointAdapter().load(
            resolved.path,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=resolved.model_id,
            revision=resolved.revision,
        )
    processor_path = resolved.path / "processor_config.json"
    processor_config = (
        json.loads(processor_path.read_text()) if processor_path.is_file() else {}
    )
    processor = make_nemotron_vl_processor(
        resolved.path,
        model.model.config,
        sequence_length_buckets=sequence_length_buckets,
        tile_count_buckets=tile_count_buckets,
        max_input_tiles=int(processor_config.get("max_input_tiles", 6)),
        use_thumbnail=bool(processor_config.get("use_thumbnail", True)),
    )
    return model, processor


__all__ = ["load_nemotron_vl"]
