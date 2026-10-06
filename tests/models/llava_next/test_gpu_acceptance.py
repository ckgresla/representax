"""Physical adapter-training acceptance for real LLaVA-NeXT checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.models import apply_quantized_lora, lora_parameter_filter
from representax.models.llava_next import load_llava_next
from representax.precision import PrecisionPolicy
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance

_ENVIRONMENTS = {
    "bge-s1": "REPRESENTAX_BGE_VL_MLLM_S1_CHECKPOINT",
    "bge-s2": "REPRESENTAX_BGE_VL_MLLM_S2_CHECKPOINT",
    "bge-v15-zs": "REPRESENTAX_BGE_VL_V15_ZS_CHECKPOINT",
    "bge-v15-mmeb": "REPRESENTAX_BGE_VL_V15_MMEB_CHECKPOINT",
    "e5-v": "REPRESENTAX_E5_V_CHECKPOINT",
}


def _checkpoint(variant: str) -> Path:
    value = os.environ.get(_ENVIRONMENTS[variant])
    if value is None:
        pytest.skip(f"set {_ENVIRONMENTS[variant]} for LLaVA-NeXT GPU acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _image():
    from PIL import Image

    pixels = (
        np.arange(64 * 512 * 3, dtype=np.uint32).reshape((64, 512, 3)) % 251
    ).astype(np.uint8)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("variant", tuple(_ENVIRONMENTS))
def test_real_checkpoint_runs_three_image_text_updates(variant):
    if jax.default_backend() != "gpu":
        pytest.skip("real LLaVA-NeXT training acceptance requires a GPU")
    model, processor = load_llava_next(
        _checkpoint(variant),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(1024,),
        image_count_buckets=(1,),
        tile_count_buckets=(3,),
    )
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(83),
        target_pattern="text.layers",
    )
    trainable = lora_parameter_filter(model)
    instruction = (
        "Retrieve the visual document that answers the question."
        if variant != "e5-v"
        else None
    )
    left = processor(
        ("Which item contains a chart?",),
        route=Route.QUERY,
        instruction=instruction,
    )
    right = processor(
        ({"text": "A deterministic chart.", "image": _image()},),
        route=Route.DOCUMENT,
    )
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=np.asarray((0.8,), dtype=np.float32),
    )
    precision = PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.bfloat16),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.0)
    state = init_train_state(
        model,
        optimizer,
        precision=precision,
        trainable_filter=trainable,
    )
    step = build_train_step(
        CosineRegressionTask(
            left_route=Route.QUERY,
            right_route=Route.DOCUMENT,
        ),
        optimizer,
        max_grad_norm=1.0,
        precision=precision,
        trainable_filter=trainable,
    )
    losses = []
    for _ in range(3):
        result = step(state, batch, None)
        jax.block_until_ready(result.metrics.loss)
        state = result.state
        losses.append(float(result.metrics.loss))
        assert bool(result.metrics.numeric_finite)
        assert not bool(result.metrics.skipped_update)
        assert float(result.metrics.update_global_norm) > 0
    assert int(state.step) == 3
    print({"variant": variant, "losses": losses})
