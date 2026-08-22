"""Physical few-step training acceptance for the production BGE-VL family."""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route, encode
from representax.models.clip import CLIPCheckpointAdapter, CLIPEncoder, load_clip
from representax.precision import PrecisionPolicy, precision_context
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_BGE_VL_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_BGE_VL_CHECKPOINT for GPU acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _images():
    from PIL import Image

    pixels = (
        np.arange(2 * 224 * 224 * 3, dtype=np.uint32).reshape(2, 224, 224, 3) % 251
    ).astype(np.uint8)
    return tuple(Image.fromarray(value) for value in pixels)


def test_real_bge_vl_runs_three_updates_and_round_trips(tmp_path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real BGE-VL training acceptance requires a GPU")
    model, processor = load_clip(
        _checkpoint(),
        local_files_only=True,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
    )
    left = processor(("A patterned image.", "A different patterned image."))
    right = processor(_images())
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=np.asarray((0.8, 0.2), dtype=np.float32),
    )
    optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.01)
    precision = PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.float32),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )
    state = init_train_state(model, optimizer, precision=precision)
    step = build_train_step(
        CosineRegressionTask(),
        optimizer,
        max_grad_norm=1.0,
        precision=precision,
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

    trained_model = state.model
    if not isinstance(trained_model, CLIPEncoder):
        raise TypeError("the CLIP training step returned a different model family")
    adapter = CLIPCheckpointAdapter(normalize_output=True)
    export = adapter.save(trained_model, tmp_path / "hf-export")
    reloaded = adapter.load(
        export,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
    )
    with precision_context(precision):
        expected = encode(trained_model, left, route=Route.GENERIC)
        actual = encode(reloaded, left, route=Route.GENERIC)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    print({"losses": losses, "update_norms": update_norms})
