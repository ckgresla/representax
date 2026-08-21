"""Packed low-rank adapter contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType

from representax.core import EncoderMetadata, Modality, Route
from representax.models import (
    Linear,
    QuantizedLoRALinear,
    apply_quantized_lora,
    lora_parameter_filter,
    merge_quantized_lora,
)
from representax.tasks.pairwise import CosineRegressionTask, pairwise_batch
from representax.train import ShardingPlan, build_train_step, init_train_state


class _AdapterEncoder(eqx.Module):
    projection: Any
    metadata: EncoderMetadata

    def __init__(self, *, key: jax.Array) -> None:
        self.projection = Linear.init(
            8,
            6,
            key=key,
            scale=8**-0.5,
            dtype=jnp.float32,
            bias=True,
        )
        self.metadata = EncoderMetadata(
            model_id="representax/test-adapter",
            revision="1",
            output_dimension=6,
            routes=frozenset(Route),
            modalities=frozenset({Modality.TEXT}),
        )

    def encode(
        self,
        inputs: jax.Array,
        *,
        route: Route,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del route, key
        return self.projection(inputs)


def _adapter(model: _AdapterEncoder) -> _AdapterEncoder:
    return apply_quantized_lora(
        model,
        rank=2,
        alpha=4.0,
        key=jax.random.key(2),
    )


def _array_bytes(tree: Any) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf)
    )


def test_quantized_lora_packs_base_and_zero_initializes_adapter_output():
    original = _AdapterEncoder(key=jax.random.key(1))
    adapted = _adapter(original)

    assert isinstance(adapted.projection, QuantizedLoRALinear)
    assert adapted.projection.packed_weight.dtype == jnp.dtype(jnp.uint8)
    assert adapted.projection.packed_weight.size == 6 * 4
    assert adapted.projection.scale_bits.dtype == jnp.dtype(jnp.uint16)
    assert np.count_nonzero(adapted.projection.lora_b) == 0

    values = jax.random.normal(jax.random.key(3), (4, 8))
    merged = merge_quantized_lora(adapted)
    np.testing.assert_allclose(
        adapted.encode(values, route=Route.GENERIC),
        merged.encode(values, route=Route.GENERIC),
        rtol=2e-3,
        atol=2e-3,
    )


def test_lora_filter_allocates_optimizer_state_only_for_adapter_arrays():
    original = _AdapterEncoder(key=jax.random.key(4))
    adapted = _adapter(original)
    optimizer = optax.adamw(1e-3)
    selected = lora_parameter_filter(adapted)
    full_state = init_train_state(original, optimizer)
    adapter_state = init_train_state(
        adapted,
        optimizer,
        trainable_filter=selected,
    )

    selected_arrays = [
        leaf
        for leaf in jax.tree.leaves(eqx.filter(adapted, selected))
        if eqx.is_array(leaf)
    ]
    assert {leaf.shape for leaf in selected_arrays} == {(2, 8), (6, 2)}
    assert _array_bytes(adapter_state.optimizer_state) < _array_bytes(
        full_state.optimizer_state
    )


def test_adapter_step_changes_only_lora_parameters():
    model = _adapter(_AdapterEncoder(key=jax.random.key(5)))
    selected = lora_parameter_filter(model)
    optimizer = optax.adamw(1e-2, weight_decay=0.0)
    state = init_train_state(model, optimizer, trainable_filter=selected)
    values = jax.random.normal(jax.random.key(6), (4, 8))
    batch = pairwise_batch(
        left=values,
        right=values.at[:, 0].add(0.1),
        labels=jnp.asarray([0.7, 0.8, 0.9, 0.6], dtype=jnp.float32),
    )
    step = build_train_step(
        CosineRegressionTask(),
        optimizer,
        trainable_filter=selected,
    )

    result = step(state, batch, None)
    state_model = cast(_AdapterEncoder, state.model)
    result_model = cast(_AdapterEncoder, result.state.model)
    _, frozen_before = eqx.partition(state.model, selected)
    _, frozen_after = eqx.partition(result.state.model, selected)

    for before, after in zip(
        (leaf for leaf in jax.tree.leaves(frozen_before) if eqx.is_array(leaf)),
        (leaf for leaf in jax.tree.leaves(frozen_after) if eqx.is_array(leaf)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    assert not np.array_equal(
        state_model.projection.lora_b,
        result_model.projection.lora_b,
    )
    assert bool(result.metrics.numeric_finite)


def test_quantized_adapter_tree_round_trips_without_source_weights(tmp_path: Path):
    model = _adapter(_AdapterEncoder(key=jax.random.key(7)))
    template = _adapter(_AdapterEncoder(key=jax.random.key(8)))
    path = tmp_path / "adapter.eqx"

    eqx.tree_serialise_leaves(path, model)
    restored = eqx.tree_deserialise_leaves(path, template)

    for expected, actual in zip(
        (leaf for leaf in jax.tree.leaves(model) if eqx.is_array(leaf)),
        (leaf for leaf in jax.tree.leaves(restored) if eqx.is_array(leaf)),
        strict=True,
    ):
        np.testing.assert_array_equal(expected, actual)


@pytest.mark.distributed
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("strategy", ["ddp", "fsdp"])
def test_quantized_adapter_matches_ten_unsharded_updates(
    world_size: int,
    strategy: str,
):
    if len(jax.devices()) < world_size:
        pytest.skip(f"test requires {world_size} devices")
    model = _adapter(_AdapterEncoder(key=jax.random.key(21)))
    selected = lora_parameter_filter(model)
    optimizer = optax.adamw(1e-2, weight_decay=0.0)
    reference_state = init_train_state(
        model,
        optimizer,
        trainable_filter=selected,
    )
    distributed_state = init_train_state(
        model,
        optimizer,
        trainable_filter=selected,
    )
    mesh = jax.make_mesh(
        (world_size,),
        ("data",),
        axis_types=(AxisType.Auto,),
        devices=jax.devices()[:world_size],
    )
    if strategy == "ddp":
        plan = ShardingPlan.ddp(
            distributed_state,
            optimizer,
            mesh,
            axis_name="data",
            trainable_filter=selected,
        )
    else:
        plan = ShardingPlan.fsdp(
            distributed_state,
            optimizer,
            mesh,
            parameter_axis_name="data",
            data_axis_name="data",
            minimum_parameter_elements=1,
            trainable_filter=selected,
        )
    values = jax.random.normal(jax.random.key(22), (4 * world_size, 8))
    batch = pairwise_batch(
        left=values,
        right=values.at[:, 0].add(0.1),
        labels=jnp.linspace(0.55, 0.95, values.shape[0], dtype=jnp.float32),
    )
    task = CosineRegressionTask()
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        trainable_filter=selected,
    )
    distributed_step = build_train_step(
        task,
        optimizer,
        plan=plan,
        max_grad_norm=None,
        trainable_filter=selected,
    )
    distributed_state = plan.place_state(distributed_state)
    distributed_batch = plan.place_batch(batch)

    for _ in range(10):
        reference = reference_step(reference_state, batch, None)
        distributed = distributed_step(distributed_state, distributed_batch, None)
        reference_state = reference.state
        distributed_state = distributed.state

    reference_trainable = eqx.filter(reference_state.model, selected)
    distributed_trainable = eqx.filter(distributed_state.model, selected)
    for expected, actual in zip(
        (leaf for leaf in jax.tree.leaves(reference_trainable) if eqx.is_array(leaf)),
        (leaf for leaf in jax.tree.leaves(distributed_trainable) if eqx.is_array(leaf)),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=4e-5)
    np.testing.assert_allclose(
        distributed.metrics.loss,
        reference.metrics.loss,
        rtol=5e-4,
        atol=4e-5,
    )
