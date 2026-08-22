"""One-shot native model and processor loading for Qwen2/Qwen2.5-VL."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import Qwen2VLCheckpointAdapter
from .config import (
    BGE_VL_SCREENSHOT_MODEL_ID,
    BGE_VL_SCREENSHOT_REVISION,
    JINA_RERANKER_M0_MODEL_ID,
    JINA_RERANKER_M0_REVISION,
    NOMIC_MULTIMODAL_3B_MODEL_ID,
    NOMIC_MULTIMODAL_3B_REVISION,
    NOMIC_MULTIMODAL_7B_MODEL_ID,
    NOMIC_MULTIMODAL_7B_REVISION,
    QWEN2_5_VL_3B_REVISION,
    QWEN2_5_VL_7B_REVISION,
)
from .model import Qwen2VLEncoder, Qwen2VLReranker
from .processing import Qwen2VLProcessorMode, make_qwen2_vl_processor

_REVISIONS = {
    BGE_VL_SCREENSHOT_MODEL_ID: BGE_VL_SCREENSHOT_REVISION,
    NOMIC_MULTIMODAL_3B_MODEL_ID: NOMIC_MULTIMODAL_3B_REVISION,
    NOMIC_MULTIMODAL_7B_MODEL_ID: NOMIC_MULTIMODAL_7B_REVISION,
    JINA_RERANKER_M0_MODEL_ID: JINA_RERANKER_M0_REVISION,
}
_BASE_REVISIONS = {
    "Qwen/Qwen2.5-VL-3B-Instruct": QWEN2_5_VL_3B_REVISION,
    "Qwen/Qwen2.5-VL-7B-Instruct": QWEN2_5_VL_7B_REVISION,
}


def _resolve(
    model_name_or_path: str | Path,
    *,
    revision: str | None,
    cache_directory: str | Path | None,
    local_files_only: bool,
):
    return resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision or _REVISIONS.get(str(model_name_or_path)),
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )


def _embedding_mode(model_id: str, checkpoint: Path) -> Qwen2VLProcessorMode:
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    exported_mode = config.get("representax_processor_mode")
    if exported_mode in {"embedding", "bge_embedding", "nomic_embedding"}:
        return exported_mode
    if (checkpoint / "adapter_config.json").is_file():
        return "nomic_embedding"
    if (
        model_id == BGE_VL_SCREENSHOT_MODEL_ID
        or (checkpoint / "modeling_bge_vl_screenshot.py").is_file()
    ):
        return "bge_embedding"
    return "embedding"


def load_qwen2_vl_embedding(
    model_name_or_path: str | Path = BGE_VL_SCREENSHOT_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    processor_mode: Qwen2VLProcessorMode | None = None,
    **adapter_options: Any,
) -> tuple[Qwen2VLEncoder, Processor]:
    """Resolve full or PEFT artifacts and construct one native model/processor pair."""

    resolved = _resolve(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    adapter = Qwen2VLCheckpointAdapter(**adapter_options)
    adapter_path = resolved.path / "adapter_config.json"
    has_full_checkpoint = (resolved.path / "model.safetensors").is_file() or (
        resolved.path / "model.safetensors.index.json"
    ).is_file()
    # Checkpoint conversion may stack or transpose multi-GiB parameter arrays.
    # Keep that entire transformation on host; accelerator placement is a
    # separate runtime concern handled after the model and processor exist.
    with jax.default_device(jax.devices("cpu")[0]):
        if adapter_path.is_file() and not has_full_checkpoint:
            adapter_config = json.loads(adapter_path.read_text())
            base_id = str(adapter_config["base_model_name_or_path"])
            base = resolve_hf_checkpoint(
                base_id,
                revision=_BASE_REVISIONS.get(base_id),
                cache_directory=cache_directory,
                local_files_only=local_files_only,
            )
            model = adapter.load_lora(
                base.path,
                resolved.path,
                parameter_dtype=parameter_dtype,
                compute_dtype=compute_dtype,
                model_id=resolved.model_id,
                revision=resolved.revision,
                pooling="last",
            )
        else:
            model = adapter.load(
                resolved.path,
                parameter_dtype=parameter_dtype,
                compute_dtype=compute_dtype,
                model_id=resolved.model_id,
                revision=resolved.revision,
                pooling="last",
            )
    processor = make_qwen2_vl_processor(
        resolved.path,
        model.config,
        mode=(
            _embedding_mode(resolved.model_id, resolved.path)
            if processor_mode is None
            else processor_mode
        ),
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
    )
    return model, processor


def load_qwen2_vl_reranker(
    model_name_or_path: str | Path = JINA_RERANKER_M0_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    patch_count_buckets: Sequence[int] = (256, 1024, 4096, 8192),
    **adapter_options: Any,
) -> tuple[Qwen2VLReranker, Processor]:
    """Resolve Jina reranker m0 and its exact pair formatter."""

    resolved = _resolve(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    with jax.default_device(jax.devices("cpu")[0]):
        model = Qwen2VLCheckpointAdapter(**adapter_options).load_reranker(
            resolved.path,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            model_id=resolved.model_id,
            revision=resolved.revision,
        )
    processor = make_qwen2_vl_processor(
        resolved.path,
        model.model.config,
        mode="reranking",
        sequence_length_buckets=sequence_length_buckets,
        patch_count_buckets=patch_count_buckets,
    )
    return model, processor


__all__ = ["load_qwen2_vl_embedding", "load_qwen2_vl_reranker"]
