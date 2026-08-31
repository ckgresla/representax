from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from representax.models.vjepa2_1 import VJEPA2_1Config, VJEPA2_1Model
from representax.tasks.jepa import VJEPA2_1Batch, VJEPA2_1Task
from representax.train import ShardingPlan, build_train_step, init_train_state


def model_and_batch() -> tuple[VJEPA2_1Model, VJEPA2_1Batch]:
    config = VJEPA2_1Config(
        image_size=8,
        patch_size=4,
        video_frames=4,
        tubelet_size=2,
        hidden_size=12,
        depth=2,
        heads=2,
        predictor_hidden_size=12,
        predictor_depth=2,
        predictor_heads=2,
        supervision_layers=(0, 1),
    )
    model = VJEPA2_1Model.init(
        config,
        key=jax.random.key(0),
        rematerialization="full",
    )
    context = jnp.asarray([[[0, 1]], [[0, 2]], [[1, 2]], [[0, 3]]], dtype=jnp.int32)
    target = jnp.asarray([[[2, 3]], [[1, 3]], [[0, 3]], [[1, 2]]], dtype=jnp.int32)
    return model, VJEPA2_1Batch(
        pixels=jax.random.normal(jax.random.key(1), (4, 3, 8, 8)),
        context_ids=context,
        target_ids=target,
        context_valid=jnp.ones_like(context, dtype=jnp.bool_),
        target_valid=jnp.ones_like(target, dtype=jnp.bool_),
    )


def run_steps(state, step, batch, count=2):
    for index in range(count):
        state = step(state, batch, jax.random.fold_in(jax.random.key(7), index)).state
    return state


def assert_models_close(actual, expected) -> None:
    actual = cast(VJEPA2_1Model, actual)
    expected = cast(VJEPA2_1Model, expected)
    for left, right in zip(
        jax.tree.leaves(eqx.filter(actual, eqx.is_array)),
        jax.tree.leaves(eqx.filter(expected, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6)


@pytest.mark.distributed
@pytest.mark.parametrize("device_count", (2, 4))
@pytest.mark.parametrize("strategy", ("ddp", "fsdp"))
def test_vjepa_trajectory_matches_unsharded_reference(
    device_count: int,
    strategy: str,
) -> None:
    if len(jax.devices()) < device_count:
        pytest.skip(f"requires {device_count} visible devices")
    model, batch = model_and_batch()
    task = VJEPA2_1Task(ema_start=0.9, ema_end=0.95, ema_steps=4)
    optimizer = optax.sgd(1e-3)
    trainable_filter = model.training_filter()
    reference_state = init_train_state(
        model,
        optimizer,
        trainable_filter=trainable_filter,
    )
    reference = run_steps(
        reference_state,
        build_train_step(
            task,
            optimizer,
            max_grad_norm=None,
            trainable_filter=trainable_filter,
        ),
        batch,
    )

    mesh = jax.make_mesh(
        (device_count,),
        ("data" if strategy == "ddp" else "fsdp",),
        devices=jax.devices()[:device_count],
        axis_types=(jax.sharding.AxisType.Auto,),
    )
    state = init_train_state(
        model,
        optimizer,
        trainable_filter=trainable_filter,
    )
    if strategy == "ddp":
        plan = ShardingPlan.ddp(
            state,
            optimizer,
            mesh,
            axis_name="data",
            trainable_filter=trainable_filter,
        )
    else:
        plan = ShardingPlan.fsdp(
            state,
            optimizer,
            mesh,
            parameter_axis_name="fsdp",
            minimum_parameter_elements=1,
            trainable_filter=trainable_filter,
        )
    distributed = run_steps(
        plan.place_state(state),
        build_train_step(
            task,
            optimizer,
            plan=plan,
            max_grad_norm=None,
            trainable_filter=trainable_filter,
        ),
        plan.place_batch(batch),
    )
    assert int(distributed.step) == int(reference.step) == 2
    assert_models_close(distributed.model, reference.model)
    if strategy == "fsdp":
        assert any(
            tuple(spec)
            for spec in jax.tree.leaves(
                plan.parameter_specs,
                is_leaf=lambda value: isinstance(value, jax.sharding.PartitionSpec),
            )
        )
