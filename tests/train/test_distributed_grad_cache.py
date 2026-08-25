"""Global-negative GradCache acceptance over named data-parallel meshes."""

from __future__ import annotations

from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.config import (
    BatchConfig,
    ComponentConfig,
    CustomShardingConfig,
    DataConfig,
    FSDPConfig,
    GradCacheConfig,
    JobConfig,
    MeshConfig,
    PartitionRuleConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.core import (
    EncoderMetadata,
    LateInteractionRepresentation,
    Modality,
    Route,
    encode,
    encode_late_interaction,
)
from representax.models import DenseEncoder
from representax.precision import resolve_precision_policy
from representax.tasks.late_interaction import (
    LateInteractionTask,
    late_interaction_loss_terms,
)
from representax.tasks.modifiers import MatryoshkaTask
from representax.tasks.retrieval import MNRTask, mnr_loss_terms, retrieval_batch
from representax.train import (
    CheckpointManager,
    Direct,
    GradCache,
    ShardingPlan,
    build_job_runtime,
    build_train_step,
    init_train_state,
    training_checkpointables,
)
from tests.train.toy_retrieval import (
    TOY_BATCH_SIZE,
    identity,
    resolve_toy_retrieval,
    toy_job_config,
)


def _assert_array_trees_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 4e-5,
    atol: float = 4e-6,
) -> None:
    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


class _LateInteractionBatch(eqx.Module):
    values: jax.Array
    valid: jax.Array


class _LateInteractionEncoder(eqx.Module):
    projection: jax.Array
    metadata: EncoderMetadata

    def __init__(self, *, key: jax.Array) -> None:
        self.projection = jax.random.normal(key, (8, 8)) / jnp.sqrt(8.0)
        self.metadata = EncoderMetadata(
            model_id="representax/distributed-late-interaction",
            revision="1",
            output_dimension=8,
            routes=frozenset({Route.QUERY, Route.DOCUMENT}),
            modalities=frozenset({Modality.TEXT}),
        )

    def encode_late_interaction(
        self,
        inputs: _LateInteractionBatch,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> LateInteractionRepresentation:
        del route, key
        return LateInteractionRepresentation(
            values=jnp.einsum(
                "bth,hd->btd",
                inputs.values,
                self.projection,
                precision=jax.lax.Precision.HIGHEST,
            ),
            valid=inputs.valid,
        )


def _late_interaction_batch():
    queries = jax.random.normal(jax.random.key(101), (8, 4, 8))
    documents = queries + 0.15 * jax.random.normal(jax.random.key(102), queries.shape)
    query_valid = jnp.asarray(
        [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, False],
            [True, True, True, True],
            [True, True, False, False],
        ]
    )
    document_valid = jnp.asarray(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    return retrieval_batch(
        query=_LateInteractionBatch(queries, query_valid),
        document=_LateInteractionBatch(documents, document_valid),
        positive_mask=jnp.eye(8, dtype=jnp.bool_),
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_late_interaction_grad_cache_matches_global_ten_update_trajectory(
    world_size: int,
    strategy: str,
):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = _LateInteractionEncoder(key=jax.random.key(103))
    task = LateInteractionTask(
        temperature=0.2,
        symmetric=True,
        negative_scope="global",
    )
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    initial = init_train_state(model, optimizer)
    batch = _late_interaction_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        devices=devices[:world_size],
    )
    plan = (
        ShardingPlan.ddp(initial, optimizer, mesh, axis_name="data")
        if strategy == "ddp"
        else ShardingPlan.fsdp(
            initial,
            optimizer,
            mesh,
            parameter_axis_name="data",
            data_axis_name="data",
            minimum_parameter_elements=1,
        )
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
    reference = initial
    distributed = plan.place_state(initial)

    with jax.default_matmul_precision("highest"):
        for iteration in range(10):
            key = jax.random.fold_in(jax.random.key(104), iteration)
            reference_result = reference_step(reference, batch, key)
            distributed_result = distributed_step(
                distributed,
                plan.place_batch(batch),
                jax.device_put(key, plan.replicated_sharding),
            )
            jax.block_until_ready((reference_result, distributed_result))
            _assert_array_trees_close(
                distributed_result.metrics,
                reference_result.metrics,
            )
            reference = reference_result.state
            distributed = distributed_result.state

    _assert_array_trees_close(distributed, reference)
    assert int(distributed.step) == 10

    queries = encode_late_interaction(model, batch.query, route=Route.QUERY)
    documents = encode_late_interaction(model, batch.document, route=Route.DOCUMENT)
    global_terms = late_interaction_loss_terms(
        queries,
        documents,
        batch.positive_mask,
        temperature=task.temperature,
    )
    local_count = 8 // world_size
    local_terms = late_interaction_loss_terms(
        jax.tree.map(lambda value: value[:local_count], queries),
        jax.tree.map(lambda value: value[:local_count], documents),
        batch.positive_mask[:local_count, :local_count],
        temperature=task.temperature,
    )
    assert float(global_terms.loss) > float(local_terms.loss)


@pytest.mark.distributed
def test_late_interaction_global_negative_lowering_contains_candidate_all_gather():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices")

    model = _LateInteractionEncoder(key=jax.random.key(105))
    task = LateInteractionTask(temperature=0.2, negative_scope="global")
    optimizer = optax.adamw(learning_rate=2e-3)
    state = init_train_state(model, optimizer)
    mesh = jax.make_mesh((2,), ("data",), devices=devices[:2])
    plan = ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
    step = build_train_step(
        task,
        optimizer,
        plan=plan,
        execution=GradCache(
            query_chunk_size=2,
            document_chunk_size=2,
            loss_row_chunk_size=2,
        ),
    )
    compiled = (
        cast(Any, step)
        .lower(
            plan.place_state(state),
            plan.place_batch(_late_interaction_batch()),
            jax.device_put(jax.random.key(106), plan.replicated_sharding),
        )
        .compile()
    )
    partitioned_hlo = "\n".join(
        module.to_string() for module in compiled.runtime_executable().hlo_modules()
    )

    assert "all-gather" in partitioned_hlo
    assert "all-reduce" in partitioned_hlo


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_mixed_precision_sharding_matches_global_update(
    world_size: int,
    strategy: str,
):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = DenseEncoder(4, 4, key=jax.random.key(51), normalize=False)
    task = MNRTask(scale=7.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.0)
    state = init_train_state(model, optimizer, precision=precision)
    batch = _global_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        execution=execution,
        precision=precision,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        devices=devices[:world_size],
    )
    plan = (
        ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
        if strategy == "ddp"
        else ShardingPlan.fsdp(
            state,
            optimizer,
            mesh,
            parameter_axis_name="data",
            data_axis_name="data",
            minimum_parameter_elements=1,
        )
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        execution=execution,
        precision=precision,
    )

    reference = reference_step(state, batch, jax.random.key(53))
    distributed = distributed_step(
        plan.place_state(state),
        plan.place_batch(batch),
        jax.device_put(jax.random.key(53), plan.replicated_sharding),
    )
    jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(
        distributed,
        reference,
        rtol=2e-2,
        atol=2e-3,
    )
    assert {
        leaf.dtype
        for leaf in jax.tree.leaves(distributed.state)
        if eqx.is_inexact_array(leaf)
    } == {jnp.dtype(jnp.float32)}


def _global_batch():
    query = jnp.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    document = jnp.asarray(
        [
            [0.8, 0.2, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
            [0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.1, 0.9],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.8, 0.0, 0.0, 0.2],
        ],
        dtype=jnp.float32,
    )
    positive_mask = jnp.eye(8, dtype=jnp.bool_)
    positive_weights = jnp.eye(8, dtype=jnp.float32).at[2, 2].set(2.0)
    return retrieval_batch(
        query=query,
        document=document,
        positive_mask=positive_mask,
        positive_weights=positive_weights,
        query_valid=jnp.asarray([True, True, True, True, True, True, True, False]),
        document_valid=jnp.asarray([True, True, True, True, True, True, False, True]),
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
def test_ddp_grad_cache_matches_one_device_global_update(world_size: int):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = DenseEncoder(4, 3, key=jax.random.key(5), normalize=False)
    base_task = MNRTask(
        scale=9.0,
        symmetric=True,
    )
    task = MatryoshkaTask(base_task, (2, 3), weights=(1.0, 2.0))
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _global_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        devices=devices[:world_size],
    )
    plan = ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )

    with jax.default_matmul_precision("highest"):
        reference = reference_step(state, batch, jax.random.key(17))
        distributed = distributed_step(
            plan.place_state(state),
            plan.place_batch(batch),
            jax.device_put(jax.random.key(17), plan.replicated_sharding),
        )
        jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)

    local_count = batch.query.shape[0] // world_size
    queries = encode(model, batch.query, route=Route.QUERY)
    documents = encode(model, batch.document, route=Route.DOCUMENT)
    global_terms = mnr_loss_terms(
        queries,
        documents,
        batch.positive_mask,
        positive_weights=batch.positive_weights,
        query_valid=batch.query_valid,
        document_valid=batch.document_valid,
        scale=base_task.scale,
    )
    local_terms = mnr_loss_terms(
        queries[:1],
        documents[:local_count],
        batch.positive_mask[:1, :local_count],
        positive_weights=batch.positive_weights[:1, :local_count],
        query_valid=batch.query_valid[:1],
        document_valid=batch.document_valid[:local_count],
        scale=base_task.scale,
    )
    assert float(global_terms.row_losses[0]) > float(local_terms.row_losses[0])


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
def test_fsdp_grad_cache_matches_replicated_global_update(world_size: int):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = DenseEncoder(4, 4, key=jax.random.key(5), normalize=False)
    task = MNRTask(scale=9.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _global_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )
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
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
        donate_state=False,
    )

    with jax.default_matmul_precision("highest"):
        reference = reference_step(state, batch, jax.random.key(17))
        distributed = distributed_step(
            plan.place_state(state),
            plan.place_batch(batch),
            jax.device_put(jax.random.key(17), plan.replicated_sharding),
        )
        jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)
    weight = next(
        leaf
        for leaf in jax.tree.leaves(distributed.state.model)
        if leaf.shape == (4, 4)
    )
    weight_spec = next(
        spec
        for spec in jax.tree.leaves(
            plan.parameter_specs,
            is_leaf=lambda value: isinstance(value, jax.sharding.PartitionSpec),
        )
        if tuple(spec) == ("data", None)
    )
    assert weight.sharding.spec == weight_spec
    assert {shard.data.shape for shard in weight.addressable_shards} == {
        (4 // world_size, 4)
    }
    optimizer_weight_moments = [
        leaf
        for leaf in jax.tree.leaves(distributed.state.optimizer_state)
        if leaf.shape == (4, 4)
    ]
    assert len(optimizer_weight_moments) == 2
    assert all(
        moment.sharding.spec == weight.sharding.spec
        for moment in optimizer_weight_moments
    )


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
def test_pure_fsdp_direct_matches_replicated_update(world_size: int):
    devices = jax.devices()
    if len(devices) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")

    model = DenseEncoder(4, 4, key=jax.random.key(5), normalize=False)
    task = MNRTask(scale=9.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _global_batch()
    execution = Direct()
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("model",),
        devices=devices[:world_size],
    )
    plan = ShardingPlan.fsdp(
        state,
        optimizer,
        mesh,
        parameter_axis_name="model",
        data_axis_name=None,
        minimum_parameter_elements=1,
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
    )

    with jax.default_matmul_precision("highest"):
        reference = reference_step(state, batch, jax.random.key(17))
        distributed = distributed_step(
            plan.place_state(state),
            plan.place_batch(batch),
            jax.device_put(jax.random.key(17), plan.replicated_sharding),
        )
        jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)


@pytest.mark.distributed
def test_hybrid_data_and_fsdp_axes_match_replicated_update():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("requires at least four JAX devices")

    model = DenseEncoder(4, 4, key=jax.random.key(5), normalize=False)
    task = MNRTask(scale=9.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    batch = _global_batch()
    execution = GradCache(
        query_chunk_size=2,
        document_chunk_size=2,
        loss_row_chunk_size=2,
    )
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=0.7,
        execution=execution,
    )
    mesh = jax.make_mesh(
        (2, 2),
        ("data", "model"),
        devices=devices[:4],
    )
    plan = ShardingPlan.fsdp(
        state,
        optimizer,
        mesh,
        parameter_axis_name="model",
        data_axis_name="data",
        minimum_parameter_elements=1,
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=execution,
    )

    with jax.default_matmul_precision("highest"):
        reference = reference_step(state, batch, jax.random.key(17))
        distributed = distributed_step(
            plan.place_state(state),
            plan.place_batch(batch),
            jax.device_put(jax.random.key(17), plan.replicated_sharding),
        )
        jax.block_until_ready((reference, distributed))

    _assert_array_trees_close(distributed.metrics, reference.metrics)
    _assert_array_trees_close(distributed.state, reference.state)


@pytest.mark.distributed
def test_fsdp_lowering_contains_materialization_and_gradient_collectives():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices")

    model = DenseEncoder(4, 4, key=jax.random.key(5), normalize=False)
    task = MNRTask(scale=9.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    state = init_train_state(model, optimizer)
    mesh = jax.make_mesh((2,), ("data",), devices=devices[:2])
    plan = ShardingPlan.fsdp(
        state,
        optimizer,
        mesh,
        parameter_axis_name="data",
        data_axis_name="data",
        minimum_parameter_elements=1,
    )
    step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=GradCache(
            query_chunk_size=2,
            document_chunk_size=2,
            loss_row_chunk_size=2,
        ),
    )

    compiled = (
        cast(Any, step)
        .lower(
            plan.place_state(state),
            plan.place_batch(_global_batch()),
            jax.device_put(jax.random.key(17), plan.replicated_sharding),
        )
        .compile()
    )
    partitioned_hlo = "\n".join(
        module.to_string() for module in compiled.runtime_executable().hlo_modules()
    )

    assert "all-gather(" in partitioned_hlo
    assert "reduce-scatter(" in partitioned_hlo or "all-reduce(" in partitioned_hlo


@pytest.mark.distributed
def test_fsdp_checkpoint_restore_preserves_state_and_shardings(tmp_path):
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip("requires at least two JAX devices")

    model = DenseEncoder(4, 4, key=jax.random.key(5), normalize=False)
    task = MNRTask(scale=9.0, symmetric=True)
    optimizer = optax.adamw(learning_rate=2e-3, weight_decay=1e-2)
    initial = init_train_state(model, optimizer)
    mesh = jax.make_mesh((2,), ("data",), devices=devices[:2])
    plan = ShardingPlan.fsdp(
        initial,
        optimizer,
        mesh,
        parameter_axis_name="data",
        data_axis_name="data",
        minimum_parameter_elements=1,
    )
    step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=0.7,
        execution=GradCache(
            query_chunk_size=2,
            document_chunk_size=2,
            loss_row_chunk_size=2,
        ),
    )
    initial = plan.place_state(initial)
    result = step(
        initial,
        plan.place_batch(_global_batch()),
        jax.device_put(jax.random.key(17), plan.replicated_sharding),
    )
    jax.block_until_ready(result)
    checkpointables = training_checkpointables(
        state=result.state,
        iteration=1,
        rng=jax.random.key(19),
        data_state={"next_index": 8},
        logging_cursor={
            "events_bytes": 0,
            "metrics_bytes": 0,
            "optimizer_step": 1,
            "sequence": 0,
        },
    )
    manager = CheckpointManager(
        tmp_path / "run",
        scientific_fingerprint="sha256:fsdp-test",
        data_fingerprint="sha256:data-test",
        asynchronous=True,
    )
    manager.save(1, checkpointables)
    manager.close()

    resumed = CheckpointManager(
        tmp_path / "run",
        scientific_fingerprint="sha256:fsdp-test",
        data_fingerprint="sha256:data-test",
    )
    restored = resumed.restore_training_state(initial)
    resumed.close()

    assert restored.iteration == 1
    assert restored.data_state == {"next_index": 8}
    _assert_array_trees_close(restored.state, result.state, rtol=0.0, atol=0.0)
    for actual, expected in zip(
        (leaf for leaf in jax.tree.leaves(restored.state) if eqx.is_array(leaf)),
        (leaf for leaf in jax.tree.leaves(result.state) if eqx.is_array(leaf)),
        strict=True,
    ):
        assert actual.sharding == expected.sharding


@pytest.mark.distributed
@pytest.mark.parametrize("sharding_kind", ["fsdp", "custom"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_job_config_builds_and_runs_sharding_plan(
    world_size: int,
    sharding_kind: str,
):
    if len(jax.devices()) < world_size:
        pytest.skip(f"requires at least {world_size} JAX devices")
    job = toy_job_config(global_batch_size=TOY_BATCH_SIZE, max_steps=1)
    data = DataConfig.model_validate(
        {
            **job.data.model_dump(),
            "collate": ComponentConfig(
                target="tests.train.toy_retrieval.collate_retrieval"
            ),
            "num_threads": 0,
            "prefetch_buffer_size": 0,
        }
    )
    sharding = (
        FSDPConfig(
            data_axis="data",
            minimum_parameter_elements=1,
        )
        if sharding_kind == "fsdp"
        else CustomShardingConfig(
            data_axis="data",
            parameter_axes=("data",),
            parameter_rules=(
                PartitionRuleConfig(
                    pattern=r"\.projection\.weight$",
                    axes=("data", None),
                ),
                PartitionRuleConfig(
                    pattern=r"\.projection\.bias$",
                    axes=("data",),
                ),
            ),
        )
    )
    training = TrainingConfig.model_validate(
        {
            **job.training.model_dump(),
            "mesh": MeshConfig(axis_shapes=(world_size,), axis_names=("data",)),
            "sharding": sharding,
            "batch": BatchConfig(micro_batch_size=TOY_BATCH_SIZE // world_size),
            "grad_cache": GradCacheConfig(micro_batch_size=2),
        }
    )
    job = JobConfig.model_validate(
        {**job.model_dump(), "data": data, "training": training}
    )
    runtime = build_job_runtime(
        job,
        resolvers={"memory": resolve_toy_retrieval},
        mappers={job.data.distribution.sources[0].mapper: identity},
    )

    batch = runtime.place_batch(next(iter(runtime.batches)))
    result = runtime.step(runtime.state, batch, jax.random.key(9))
    jax.block_until_ready(result)

    assert bool(result.metrics.numeric_finite)
    assert int(result.state.step) == 1
    parameter = next(
        leaf for leaf in jax.tree.leaves(result.state.model) if leaf.ndim == 2
    )
    assert "data" in tuple(parameter.sharding.spec)
