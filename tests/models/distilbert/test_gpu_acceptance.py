"""Physical few-step training acceptance for multilingual CLIP DistilBERT."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.core import Route
from representax.integrations import load_sentence_transformer
from representax.models import (
    apply_quantized_lora,
    lora_parameter_filter,
    merge_quantized_lora,
)
from representax.models.sentence import SentenceEncoder
from representax.precision import PrecisionPolicy, precision_context
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint() -> Path:
    value = os.environ.get("REPRESENTAX_CLIP_MULTILINGUAL_CHECKPOINT")
    if value is None:
        pytest.skip("set REPRESENTAX_CLIP_MULTILINGUAL_CHECKPOINT for acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _precision() -> PrecisionPolicy:
    return PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.bfloat16),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )


def test_real_checkpoint_runs_three_adapter_updates_and_reloads(tmp_path) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real multilingual DistilBERT acceptance requires a GPU")
    checkpoint = _checkpoint()
    loaded = load_sentence_transformer(
        checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(32,),
        rematerialization="full",
    )
    model = apply_quantized_lora(
        loaded.model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(167),
        target_pattern="backbone.tower.layers",
    )
    trainable = lora_parameter_filter(model)
    batch = pairwise_batch(
        left=loaded.processor(
            (
                "A red geometric shape.",
                "A quiet harbor at sunrise.",
            )
        ),
        right=loaded.processor(
            (
                "Ein rotes geometrisches Objekt.",
                "Eine laute Straße in der Nacht.",
            )
        ),
        labels=np.asarray((0.95, 0.05), dtype=np.float32),
    )
    optimizer = optax.adamw(learning_rate=1e-4, weight_decay=0.0)
    precision = _precision()
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

    trained = cast(SentenceEncoder, merge_quantized_lora(state.model))
    export = trained.save_to_hf(
        tmp_path / "trained-sentence-transformer",
        source_checkpoint=checkpoint,
    )
    reloaded = load_sentence_transformer(
        export,
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(32,),
    )
    with precision_context(precision), jax.default_matmul_precision("highest"):
        expected = jax.device_get(trained.encode(batch.left, route=Route.GENERIC))
        actual = jax.device_get(reloaded.model.encode(batch.left, route=Route.GENERIC))
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
    print({"losses": losses, "update_norms": update_norms})
