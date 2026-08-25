"""Physical adapter-training acceptance for real Llama Nemotron VL checkpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from PIL import Image

from representax.core import Route
from representax.models import apply_quantized_lora, lora_parameter_filter
from representax.models.nemotron_vl import (
    LlamaNemotronVLReranker,
    load_nemotron_vl,
)
from representax.precision import PrecisionPolicy
from representax.tasks.cross_encoder import PointwiseBatch, PointwiseScoringTask
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


def _checkpoint(environment: str) -> Path:
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for Llama Nemotron VL GPU acceptance")
    assert value is not None
    checkpoint = Path(value)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _image() -> Image.Image:
    pixels = (
        np.arange(64 * 512 * 3, dtype=np.uint32).reshape((64, 512, 3)) % 251
    ).astype(np.uint8)
    return Image.fromarray(pixels)


def _precision() -> PrecisionPolicy:
    return PrecisionPolicy(
        parameter_dtype=jnp.dtype(jnp.bfloat16),
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
        accumulation_dtype=jnp.dtype(jnp.float32),
        loss_dtype=jnp.dtype(jnp.float32),
    )


def test_real_embedder_runs_three_multimodal_adapter_updates() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Llama Nemotron VL training acceptance requires a GPU")
    model, processor = load_nemotron_vl(
        _checkpoint("REPRESENTAX_NEMOTRON_VL_EMBED_CHECKPOINT"),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(2048,),
        tile_count_buckets=(7,),
    )
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(107),
        target_pattern="model.text.layers",
    )
    trainable = lora_parameter_filter(model)
    batch = pairwise_batch(
        left=processor(("Which item contains a chart?",), route=Route.QUERY),
        right=processor(
            ({"image": _image(), "text": "A deterministic chart."},),
            route=Route.DOCUMENT,
        ),
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
    assert int(state.step) == 3
    print({"mode": "embedding", "losses": losses})


def test_real_reranker_runs_three_multimodal_adapter_updates() -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Llama Nemotron VL training acceptance requires a GPU")
    loaded, processor = load_nemotron_vl(
        _checkpoint("REPRESENTAX_NEMOTRON_VL_RERANK_CHECKPOINT"),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(2048,),
        tile_count_buckets=(7,),
    )
    assert isinstance(loaded, LlamaNemotronVLReranker)
    model = apply_quantized_lora(
        loaded,
        rank=4,
        alpha=8.0,
        key=jax.random.key(109),
        target_pattern="model.text.layers",
    )
    trainable = lora_parameter_filter(model)
    inputs = processor(
        (
            {
                "query": "Which item contains a chart?",
                "image": _image(),
                "text": "A deterministic chart.",
            },
        ),
        route=Route.GENERIC,
    )
    batch = PointwiseBatch(
        inputs=inputs,
        labels=jnp.asarray((0.8,), dtype=jnp.float32),
        valid=jnp.ones((1,), dtype=bool),
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
        PointwiseScoringTask(objective="mse"),
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
    trained = cast(LlamaNemotronVLReranker, state.model)
    scores = trained.encode(inputs, route=Route.GENERIC)
    assert scores.shape == (1, 1)
    assert bool(jnp.all(jnp.isfinite(scores)))
    assert int(state.step) == 3
    print({"mode": "reranking", "losses": losses})
