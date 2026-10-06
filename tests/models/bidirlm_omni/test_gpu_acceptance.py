"""Physical multimodal adapter-training acceptance for BidirLM Omni."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from PIL import Image

from representax.models import apply_quantized_lora, lora_parameter_filter
from representax.models.bidirlm_omni import load_bidirlm_omni
from representax.precision import PrecisionPolicy
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_BIDIRLM_OMNI_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_BIDIRLM_OMNI_CHECKPOINT for GPU acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _image() -> Image.Image:
    pixels = np.arange(64 * 96 * 3, dtype=np.uint8).reshape((64, 96, 3))
    return Image.fromarray(pixels)


def _audio(frequency: float) -> dict[str, object]:
    sample_rate = 16_000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    return {
        "array": np.sin(2 * np.pi * frequency * time).astype(np.float32),
        "sampling_rate": sample_rate,
    }


def test_real_checkpoint_runs_three_multimodal_adapter_updates() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real BidirLM Omni training acceptance requires a GPU")
    model, processor = load_bidirlm_omni(
        _checkpoint(),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(128,),
        patch_count_buckets=(512,),
        audio_chunk_buckets=(2,),
        rematerialization="none",
    )
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(137),
        target_pattern="text.layers",
    )
    trainable = lora_parameter_filter(model)
    batch = pairwise_batch(
        left=processor(
            (
                {"image": _image(), "text": "A precise red square."},
                {"audio": _audio(440.0), "text": "A clear sustained tone."},
            )
        ),
        right=processor(
            (
                "A deterministic geometric image.",
                {"audio": _audio(523.25), "text": "A different musical tone."},
            )
        ),
        labels=np.asarray((0.8, 0.2), dtype=np.float32),
    )
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.0)
    precision = PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.bfloat16),
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
