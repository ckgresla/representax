"""One-shot loading for ordinary and Sentence Transformers CLIP artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .checkpoint import CLIPCheckpointAdapter, clip_checkpoint_directory
from .config import BGE_VL_BASE_MODEL_ID, BGE_VL_BASE_REVISION
from .model import CLIPEncoder
from .processing import make_clip_processor


def _normalizes_output(checkpoint: Path) -> bool:
    modules_path = checkpoint / "modules.json"
    if not modules_path.is_file():
        return False
    modules = json.loads(modules_path.read_text())
    if not isinstance(modules, list):
        raise ValueError("Sentence Transformers modules.json must contain a list")
    return any(
        isinstance(module, dict)
        and str(module.get("type", "")).rsplit(".", 1)[-1] == "Normalize"
        for module in modules
    )


def load_clip(
    model_name_or_path: str | Path = BGE_VL_BASE_MODEL_ID,
    *,
    revision: str = BGE_VL_BASE_REVISION,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.float32,
    compute_dtype: jnp.dtype = jnp.float32,
    **adapter_options,
) -> tuple[CLIPEncoder, Processor]:
    """Resolve one checkpoint and construct its native model and processor."""

    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    source = clip_checkpoint_directory(resolved.path)
    model = CLIPCheckpointAdapter(
        normalize_output=_normalizes_output(resolved.path),
        **adapter_options,
    ).load(
        source,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    return model, make_clip_processor(
        source,
        model.config,
        normalize_output=model.normalize_output,
    )


__all__ = ["load_clip"]
