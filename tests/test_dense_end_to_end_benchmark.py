"""Regression coverage for the dense end-to-end paper runner."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
from benchmarks.dense_end_to_end import MODEL_SPECS, _native_probe_embeddings

from representax.models.mpnet import MPNetBatch, MPNetConfig, MPNetEncoder


def test_mpnet_probe_applies_the_training_precision_policy():
    config = MPNetConfig(
        vocab_size=31,
        hidden_size=12,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=3,
        max_position_embeddings=16,
        relative_attention_num_buckets=32,
    )
    model = MPNetEncoder.init(
        config,
        key=jax.random.key(0),
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
    )
    inputs = MPNetBatch(
        input_ids=jnp.asarray([[0, 4, 5, 2], [0, 6, 2, 1]]),
        attention_mask=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 0]]),
    )
    batch = SimpleNamespace(left=inputs, right=inputs)

    left, right = _native_probe_embeddings(MODEL_SPECS["mpnet"], model, batch)

    assert left.shape == right.shape == (2, config.hidden_size)
    assert jnp.all(jnp.isfinite(left))
    assert jnp.all(jnp.isfinite(right))
