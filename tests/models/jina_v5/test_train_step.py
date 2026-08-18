"""Compiled task integration for the native Jina v5 text tower."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from representax.models.jina_v5 import (
    JinaV5TextBatch,
    JinaV5TextCheckpointAdapter,
)
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import build_train_step, init_train_state
from tests.models.jina_v5.test_model import _synthetic_state, tiny_config


@pytest.mark.runtime
def test_compiled_jina_v5_update_is_finite_and_preserves_bfloat16_parameters():
    config = tiny_config()
    state_dict = {
        name: value.astype(jnp.bfloat16)
        for name, value in _synthetic_state(config).items()
    }
    model = JinaV5TextCheckpointAdapter(rematerialization="full").from_state_dict(
        config,
        state_dict,
        parameter_dtype=jnp.bfloat16,
        compute_dtype=jnp.bfloat16,
        model_id="test/jina-v5",
        revision="test",
    )
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    left = JinaV5TextBatch(
        input_ids=jnp.asarray([[1, 2, 3, 0], [4, 5, 0, 0]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    right = JinaV5TextBatch(
        input_ids=jnp.asarray([[1, 2, 6, 0], [4, 7, 0, 0]]),
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    batch = pairwise_batch(
        left=left,
        right=right,
        labels=jnp.asarray([0.9, 0.7], dtype=jnp.float32),
    )

    result = build_train_step(
        CosineRegressionTask(),
        optimizer,
        max_grad_norm=1.0,
    )(state, batch, jax.random.key(9))

    assert bool(result.metrics.numeric_finite)
    assert not bool(result.metrics.skipped_update)
    assert int(result.state.step) == 1
    assert {
        value.dtype
        for value in jax.tree.leaves(result.state.model)
        if eqx.is_inexact_array(value)
    } == {jnp.dtype(jnp.bfloat16)}
