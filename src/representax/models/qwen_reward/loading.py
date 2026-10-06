"""One-shot Hub/local loading for the canonical Qwen3 reward model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from representax.integrations.huggingface import resolve_hf_checkpoint
from representax.models.processing import Processor

from .artifacts import QwenRewardCheckpointAdapter
from .config import QWEN3_REWARD_0_6B_MODEL_ID, QWEN3_REWARD_0_6B_REVISION
from .model import QwenRewardModel
from .processing import make_qwen_reward_processor


def load_qwen_reward_model(
    model_name_or_path: str | Path = QWEN3_REWARD_0_6B_MODEL_ID,
    *,
    revision: str | None = None,
    cache_directory: str | Path | None = None,
    local_files_only: bool = False,
    parameter_dtype: jnp.dtype = jnp.bfloat16,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    head_seed: int = 0,
    sequence_length_buckets: Sequence[int] = (512, 2048, 8192),
    **adapter_options,
) -> tuple[QwenRewardModel, Processor]:
    if model_name_or_path == QWEN3_REWARD_0_6B_MODEL_ID and revision is None:
        revision = QWEN3_REWARD_0_6B_REVISION
    resolved = resolve_hf_checkpoint(
        model_name_or_path,
        revision=revision,
        cache_directory=cache_directory,
        local_files_only=local_files_only,
    )
    model: QwenRewardModel = QwenRewardCheckpointAdapter(**adapter_options).load(
        resolved.path,
        head_key=jax.random.key(head_seed),
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        model_id=resolved.model_id,
        revision=resolved.revision,
    )
    processor = make_qwen_reward_processor(
        resolved.path,
        model.config.backbone.pad_token_id,
        sequence_length_buckets=sequence_length_buckets,
    )
    return model, processor


__all__ = ["load_qwen_reward_model"]
