"""Physical acceptance probes for the pinned production Qwen2.5-Omni model."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models import apply_quantized_lora, lora_parameter_filter
from representax.models.qwen2_5_omni import Qwen2_5OmniEncoder
from representax.precision import PrecisionPolicy
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_QWEN2_5_OMNI_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_QWEN2_5_OMNI_CHECKPOINT for GPU acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def test_real_checkpoint_runs_three_quantized_adapter_updates() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Qwen2.5-Omni training acceptance requires a GPU")
    model, processor = Qwen2_5OmniEncoder.load_from_hf(
        _checkpoint(),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(64,),
        patch_count_buckets=(64,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(64,),
    )
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(71),
        target_pattern="text",
    )
    trainable = lora_parameter_filter(model)
    left = processor(
        (
            "A quiet harbor at sunrise.",
            "A microscope image of living cells.",
        )
    )
    right = processor(
        (
            "Morning light over calm water.",
            "A city street during a rainstorm.",
        )
    )
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=np.asarray((0.8, 0.1), dtype=np.float32),
    )
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.0)
    precision = PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.float32),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )
    state = init_train_state(
        model,
        optimizer,
        precision=precision,
        trainable_filter=trainable,
    )
    step = build_train_step(
        CosineRegressionTask(),
        optimizer,
        max_grad_norm=1.0,
        precision=precision,
        trainable_filter=trainable,
    )
    losses = []
    update_norms = []
    for _ in range(3):
        result = step(state, batch, None)
        jax.block_until_ready(result.metrics.loss)
        state = result.state
        losses.append(float(result.metrics.loss))
        update_norms.append(float(result.metrics.update_global_norm))
        assert bool(result.metrics.numeric_finite)
        assert not bool(result.metrics.skipped_update)
    assert int(state.step) == 3
    assert all(np.isfinite(loss) for loss in losses)
    assert all(norm > 0 for norm in update_norms)
    print({"losses": losses, "update_norms": update_norms})
