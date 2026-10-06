"""One-shot native LLaVA-NeXT model and processor loading."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import LlavaNextCheckpointAdapter
from .config import (
    BGE_VL_MLLM_S1_MODEL_ID,
    BGE_VL_MLLM_S1_REVISION,
    BGE_VL_MLLM_S2_MODEL_ID,
    BGE_VL_MLLM_S2_REVISION,
    BGE_VL_V15_MMEB_MODEL_ID,
    BGE_VL_V15_MMEB_REVISION,
    BGE_VL_V15_ZS_MODEL_ID,
    BGE_VL_V15_ZS_REVISION,
    E5_V_MODEL_ID,
    E5_V_REVISION,
)
from .model import LlavaNextEncoder
from .processing import make_llava_next_processor

_REVISIONS = {
    BGE_VL_MLLM_S1_MODEL_ID: BGE_VL_MLLM_S1_REVISION,
    BGE_VL_MLLM_S2_MODEL_ID: BGE_VL_MLLM_S2_REVISION,
    BGE_VL_V15_ZS_MODEL_ID: BGE_VL_V15_ZS_REVISION,
    BGE_VL_V15_MMEB_MODEL_ID: BGE_VL_V15_MMEB_REVISION,
    E5_V_MODEL_ID: E5_V_REVISION,
}


def load_llava_next(
    model_name_or_path: str | Path = BGE_VL_V15_MMEB_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (2048, 4096, 8192),
    image_count_buckets: Sequence[int] = (1, 2, 4, 8, 16, 32),
    tile_count_buckets: Sequence[int] = (1, 3, 5, 10),
) -> tuple[LlavaNextEncoder, Processor]:
    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision or _REVISIONS.get(str(model_name_or_path)),
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        model = LlavaNextCheckpointAdapter().load(
            resolved.path,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=resolved.model_id,
            revision=resolved.revision,
        )
    processor = make_llava_next_processor(
        resolved.path,
        model.config,
        mode="e5" if model.config.text.family == "llama" else "bge",
        sequence_length_buckets=sequence_length_buckets,
        image_count_buckets=image_count_buckets,
        tile_count_buckets=tile_count_buckets,
    )
    return model, processor


__all__ = ["load_llava_next"]
