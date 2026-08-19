"""Compiled ModernVBERT training integration tests."""

from typing import Literal

import equinox as eqx
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
from representax.train import (
    GradCache,
    ShardingPlan,
    build_sharded_train_step,
    build_train_step,
    init_train_state,
)


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
    state = init_train_state(model, optimizer)
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


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("materialization_boundary", ["model", "layer"])
def test_modernvbert_fsdp_matches_ten_one_device_grad_cache_updates(
    world_size: int,
    materialization_boundary: Literal["model", "layer"],
):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")
    config = ModernVBERTTextConfig(
        vocab_size=20,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        layer_types=("full_attention", "sliding_attention"),
        local_attention=4,
        full_attention_rope_theta=10_000.0,
        sliding_attention_rope_theta=1_000.0,
        norm_epsilon=1e-5,
        max_position_embeddings=16,
    )
    model = ModernVBERTTextEncoder.init(config, key=jax.random.key(0))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer)
    task = MNRTask(scale=5.0, symmetric=True)
    execution = GradCache(query_chunk_size=1, document_chunk_size=1)
    reference_step = build_train_step(task, optimizer, execution=execution)
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        devices=devices[:world_size],
    )
    plan = ShardingPlan.fsdp(
        state,
        optimizer,
        mesh,
        parameter_axis_name="data",
        data_axis_name="data",
        minimum_parameter_elements=1,
        materialization_boundary=materialization_boundary,
    )
    distributed_step = build_sharded_train_step(
        task,
        optimizer,
        plan,
        execution=execution,
    )
    input_ids = jnp.asarray(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 0],
            [7, 8, 9, 0],
            [10, 11, 12, 0],
        ]
    )
    attention_mask = jnp.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ]
    )
    batch = retrieval_batch(
        query=ModernVBERTTextBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
        document=ModernVBERTTextBatch(
            input_ids=jnp.roll(input_ids, 1, axis=1),
            attention_mask=attention_mask,
        ),
        positive_mask=jnp.eye(4, dtype=jnp.bool_),
    )
    reference_state = state
    distributed_state = plan.place_state(state)
    distributed_batch = plan.place_batch(batch)

    with jax.default_matmul_precision("highest"):
        for update_index in range(10):
            key = jax.random.fold_in(jax.random.key(1), update_index)
            reference = reference_step(reference_state, batch, key)
            distributed = distributed_step(
                distributed_state,
                distributed_batch,
                jax.device_put(key, plan.replicated_sharding),
            )
            jax.block_until_ready((reference, distributed))
            assert bool(distributed.metrics.numeric_finite)
            assert jnp.allclose(
                distributed.metrics.loss,
                reference.metrics.loss,
                rtol=5e-5,
                atol=5e-6,
            )
            reference_state = reference.state
            distributed_state = distributed.state

    assert int(distributed_state.step) == 10
    for actual, expected in zip(
        (leaf for leaf in jax.tree.leaves(distributed_state) if eqx.is_array(leaf)),
        (leaf for leaf in jax.tree.leaves(reference_state) if eqx.is_array(leaf)),
        strict=True,
    ):
        assert jnp.allclose(actual, expected, rtol=5e-5, atol=5e-6)


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
    state = init_train_state(model, optimizer)
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
    assert isinstance(result.state.model, ModernVBERTEncoder)
    assert not jnp.array_equal(
        result.state.model.connector.weight,
        model.connector.weight,
    )
