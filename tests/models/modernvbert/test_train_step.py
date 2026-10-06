"""Compiled ModernVBERT training integration tests."""

import jax
import jax.numpy as jnp
import optax
import pytest

from representax.models.modernvbert import (
    ModernVBERTBatch,
    ModernVBERTConfig,
    ModernVBERTEncoder,
    ModernVBERTTextBatch,
    ModernVBERTTextConfig,
    ModernVBERTTextEncoder,
    ModernVBERTVisionConfig,
)
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, make_train_state


@pytest.mark.runtime
def test_modernvbert_runs_one_compiled_grad_cache_retrieval_update():
    config = ModernVBERTTextConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        layer_types=("full_attention", "sliding_attention"),
        local_attention=4,
        full_attention_rope_theta=10_000.0,
        sliding_attention_rope_theta=1_000.0,
        norm_epsilon=1e-5,
        max_position_embeddings=16,
    )
    model = ModernVBERTTextEncoder.init(
        config,
        key=jax.random.key(0),
    )
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    state = make_train_state(model, optimizer)
    step = build_train_step(
        MNRTask(scale=5.0, symmetric=True),
        optimizer,
        execution=GradCache(query_chunk_size=1, document_chunk_size=1),
    )
    batch = retrieval_batch(
        query=ModernVBERTTextBatch(
            input_ids=jnp.asarray([[1, 2, 3, 0], [4, 5, 6, 0]]),
            attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 0]]),
        ),
        document=ModernVBERTTextBatch(
            input_ids=jnp.asarray([[7, 8, 9, 0], [10, 11, 12, 0]]),
            attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 0]]),
        ),
        positive_mask=jnp.eye(2, dtype=jnp.bool_),
    )

    result = step(state, batch, jax.random.key(1))

    assert int(result.state.step) == 1
    assert bool(result.metrics.numeric_finite)
    assert float(result.metrics.update_global_norm) > 0.0


@pytest.mark.runtime
def test_multimodal_modernvbert_updates_vision_and_connector():
    text = ModernVBERTTextConfig(
        vocab_size=20,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        layer_types=("full_attention",),
        local_attention=4,
        full_attention_rope_theta=10_000.0,
        sliding_attention_rope_theta=1_000.0,
        norm_epsilon=1e-5,
        max_position_embeddings=16,
    )
    config = ModernVBERTConfig(
        text=text,
        vision=ModernVBERTVisionConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=8,
            patch_size=2,
            norm_epsilon=1e-6,
        ),
        image_token_id=19,
        pixel_shuffle_factor=2,
    )
    model = ModernVBERTEncoder.init(config, key=jax.random.key(2))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    state = make_train_state(model, optimizer)
    step = build_train_step(MNRTask(scale=5.0, symmetric=True), optimizer)
    input_ids = jnp.asarray([[1, 19, 19, 19, 19, 2], [3, 19, 19, 19, 19, 4]])
    query = ModernVBERTBatch(
        input_ids=input_ids,
        attention_mask=jnp.ones_like(input_ids),
        pixel_values=jax.random.normal(jax.random.key(3), (2, 1, 3, 8, 8)),
    )
    document_ids = jnp.asarray([[5, 6, 7, 0], [8, 9, 10, 0]])
    document = ModernVBERTBatch(
        input_ids=document_ids,
        attention_mask=jnp.asarray([[1, 1, 1, 0], [1, 1, 1, 0]]),
    )
    batch = retrieval_batch(
        query=query,
        document=document,
        positive_mask=jnp.eye(2, dtype=jnp.bool_),
    )

    result = step(state, batch, jax.random.key(4))

    assert int(result.state.step) == 1
    assert bool(result.metrics.numeric_finite)
    assert not jnp.array_equal(
        result.state.model.connector.weight,
        model.connector.weight,
    )
