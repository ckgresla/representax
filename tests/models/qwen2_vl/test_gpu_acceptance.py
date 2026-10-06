"""Physical few-step training acceptance for native Qwen2-VL checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import LossOutput, Route
from representax.models import (
    apply_quantized_lora,
    lora_parameter_filter,
    quantize_lora_base,
)
from representax.models.qwen2_vl import (
    Qwen2VLEncoder,
    Qwen2VLReranker,
    load_qwen2_vl_embedding,
    load_qwen2_vl_reranker,
)
from representax.precision import PrecisionPolicy
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"set {variable} for Qwen2-VL GPU acceptance")
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _precision() -> PrecisionPolicy:
    return PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.float32),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )


def _images() -> tuple[Any, Any]:
    from PIL import Image

    values = (
        np.arange(2 * 56 * 56 * 3, dtype=np.uint32).reshape((2, 56, 56, 3)) % 251
    ).astype(np.uint8)
    return Image.fromarray(values[0]), Image.fromarray(values[1])


@pytest.mark.parametrize(
    ("variant", "variable"),
    (
        ("bge", "REPRESENTAX_BGE_VL_SCREENSHOT_CHECKPOINT"),
        ("nomic3", "REPRESENTAX_NOMIC_MULTIMODAL_3B_CHECKPOINT"),
        ("nomic7", "REPRESENTAX_NOMIC_MULTIMODAL_7B_CHECKPOINT"),
    ),
)
def test_real_embedding_checkpoint_runs_three_adapter_updates(
    variant: str,
    variable: str,
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Qwen2-VL training acceptance requires a GPU")
    model, processor = load_qwen2_vl_embedding(
        _checkpoint(variable),
        local_files_only=True,
        cache_directory=os.environ.get("HF_HUB_CACHE"),
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(128,),
        patch_count_buckets=(512,),
    )
    if variant == "bge":
        model = apply_quantized_lora(
            model,
            rank=4,
            alpha=8.0,
            key=jax.random.key(81),
            target_pattern="text",
        )
    else:
        model = quantize_lora_base(model)
    trainable = lora_parameter_filter(model)
    left = processor(("A striped scientific chart.",), route=Route.QUERY)
    right = processor((_images()[0],), route=Route.DOCUMENT)
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=np.asarray((0.8,), dtype=np.float32),
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
    trained = cast(Qwen2VLEncoder, state.model)
    encoded = trained.encode(left, route=Route.QUERY)
    assert bool(jnp.all(jnp.isfinite(encoded)))
    assert int(state.step) == 3
    print({"variant": variant, "losses": losses})


class _ScoreBatch(eqx.Module):
    inputs: Any
    labels: Float[Array, " pair"]
    valid: Bool[Array, " pair"]


class _ScoreRegressionTask(eqx.Module):
    """Acceptance-only task proving the scorer uses the generic trainer."""

    def loss(
        self,
        model: Qwen2VLReranker,
        batch: _ScoreBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        scores = model.score(batch.inputs, key=key)
        errors = jnp.square(scores.astype(jnp.float32) - batch.labels)
        count = jnp.maximum(jnp.sum(batch.valid), 1)
        loss = jnp.sum(jnp.where(batch.valid, errors, 0.0)) / count
        return LossOutput(loss=loss, metrics={"score_mean": jnp.mean(scores)})


def test_real_jina_reranker_runs_three_adapter_updates() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Jina reranker training acceptance requires a GPU")
    model, processor = load_qwen2_vl_reranker(
        _checkpoint("REPRESENTAX_JINA_RERANKER_M0_CHECKPOINT"),
        local_files_only=True,
        cache_directory=os.environ.get("HF_HUB_CACHE"),
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(128,),
        patch_count_buckets=(512,),
    )
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(91),
        target_pattern="model.text",
    )
    trainable = lora_parameter_filter(model)
    inputs = processor(
        (
            ("Which item is a chart?", "A chart with three colored bars."),
            ("Which item is a chart?", _images()[0]),
        )
    )
    batch = _ScoreBatch(
        inputs=inputs,
        labels=jnp.asarray((0.9, 0.2), dtype=jnp.float32),
        valid=jnp.ones((2,), dtype=bool),
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
        _ScoreRegressionTask(),
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
    trained = cast(Qwen2VLReranker, state.model)
    scores = trained.score(inputs)
    assert scores.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(scores)))
    assert int(state.step) == 3
    print({"variant": "jina", "losses": losses})
