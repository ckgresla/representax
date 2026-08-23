"""Physical few-step training acceptance for native Qwen text rerankers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import LossOutput
from representax.models import apply_quantized_lora, lora_parameter_filter
from representax.models.qwen_reranker import QwenReranker, load_qwen_reranker
from representax.precision import PrecisionPolicy
from representax.train import build_train_step, init_train_state

pytestmark = pytest.mark.performance


class _RerankingBatch(eqx.Module):
    inputs: Any
    labels: Float[Array, " batch"]
    valid: Bool[Array, " batch"]


class _BinaryRerankingTask(eqx.Module):
    """Exercise raw relevance logits through the generic trainer."""

    def loss(
        self,
        model: QwenReranker,
        batch: _RerankingBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> LossOutput:
        logits = model.logits(batch.inputs, key=key)
        losses = optax.sigmoid_binary_cross_entropy(logits, batch.labels)
        count = jnp.maximum(jnp.sum(batch.valid), 1)
        loss = jnp.sum(jnp.where(batch.valid, losses, 0.0)) / count
        return LossOutput(loss=loss, metrics={"logit_mean": jnp.mean(logits)})


def _checkpoint(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"set {variable} for Qwen reranker GPU acceptance")
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


@pytest.mark.parametrize(
    ("generation", "variable"),
    (
        ("qwen3", "REPRESENTAX_QWEN3_RERANKER_CHECKPOINT"),
        ("qwen3", "REPRESENTAX_CONTEXTUAL_RERANKER_CHECKPOINT"),
        ("qwen2", "REPRESENTAX_MXBAI_RERANKER_CHECKPOINT"),
    ),
)
def test_real_checkpoint_runs_three_adapter_updates(
    generation: str,
    variable: str,
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("real Qwen reranker training acceptance requires a GPU")
    model, processor = load_qwen_reranker(
        _checkpoint(variable),
        local_files_only=True,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        sequence_length_buckets=(128,),
        rematerialization="none",
    )
    assert model.config.generation == generation
    model = apply_quantized_lora(
        model,
        rank=4,
        alpha=8.0,
        key=jax.random.key(149),
        target_pattern="text.layers",
    )
    trainable = lora_parameter_filter(model)
    inputs = processor(
        (
            (
                "Which planet is known as the Red Planet?",
                "Mars is called the Red Planet because of iron oxide.",
            ),
            (
                "Which planet is known as the Red Planet?",
                "Venus has a dense carbon-dioxide atmosphere.",
            ),
        )
    )
    batch = _RerankingBatch(
        inputs=inputs,
        labels=jnp.asarray((1.0, 0.0), dtype=jnp.float32),
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
        _BinaryRerankingTask(),
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
    print(
        {
            "generation": generation,
            "losses": losses,
            "update_norms": update_norms,
        }
    )
