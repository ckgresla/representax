"""Scientific specification and execution-plan tests."""

import pytest

from representax.planning import ExecutionPlan, ScientificSpec


def test_execution_plan_preserves_scientific_batch():
    science = ScientificSpec(
        task="retrieval/mnr",
        global_batch_size=64,
        max_steps=100,
        seed=7,
    )
    plan = ExecutionPlan(
        device_count=4,
        data_axis_size=4,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
    )

    plan.validate_science(science)
    assert plan.effective_batch_size == 64


@pytest.mark.parametrize("policy", ["none", "selective", "full"])
def test_execution_plan_accepts_supported_rematerialization_policies(policy):
    plan = ExecutionPlan(
        device_count=1,
        data_axis_size=1,
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        rematerialization=policy,
    )

    assert plan.rematerialization == policy


def test_execution_plan_rejects_unknown_rematerialization_policy():
    with pytest.raises(ValueError, match="rematerialization must be"):
        ExecutionPlan(
            device_count=1,
            data_axis_size=1,
            per_device_batch_size=8,
            gradient_accumulation_steps=1,
            rematerialization="automatic",  # type: ignore[arg-type]
        )


def test_execution_plan_rejects_scientific_drift():
    science = ScientificSpec(
        task="retrieval/mnr",
        global_batch_size=64,
        max_steps=100,
        seed=7,
    )
    plan = ExecutionPlan(
        device_count=2,
        data_axis_size=2,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="changes the scientific"):
        plan.validate_science(science)
