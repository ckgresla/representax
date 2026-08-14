"""Scientific and execution configuration tests."""

import pytest
from pydantic import ValidationError

from representax.config import ExecutionConfig, ScientificConfig, TrainingConfig


def _scientific(**overrides):
    values = {
        "task": "retrieval/mnr",
        "global_batch_size": 64,
        "max_steps": 100,
        "seed": 7,
    }
    values.update(overrides)
    return ScientificConfig(**values)


def test_execution_config_preserves_scientific_batch():
    scientific = _scientific()
    execution = ExecutionConfig(
        device_count=4,
        data_axis_size=4,
        per_device_batch_size=4,
        gradient_accumulation_steps=4,
    )

    training = TrainingConfig(scientific=scientific, execution=execution)

    assert training.scientific is scientific
    assert execution.effective_batch_size == 64


@pytest.mark.parametrize("policy", ["none", "selective", "full"])
def test_execution_config_accepts_supported_rematerialization_policies(policy):
    execution = ExecutionConfig(
        device_count=1,
        data_axis_size=1,
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        rematerialization=policy,
    )

    assert execution.rematerialization == policy


def test_execution_config_rejects_unknown_rematerialization_policy():
    with pytest.raises(
        ValueError,
        match="Input should be 'none', 'selective' or 'full'",
    ):
        ExecutionConfig(
            device_count=1,
            data_axis_size=1,
            per_device_batch_size=8,
            gradient_accumulation_steps=1,
            rematerialization="automatic",  # type: ignore[arg-type]
        )


def test_training_config_rejects_scientific_drift():
    scientific = _scientific()
    execution = ExecutionConfig(
        device_count=2,
        data_axis_size=2,
        per_device_batch_size=8,
        gradient_accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="changes the scientific"):
        TrainingConfig(scientific=scientific, execution=execution)


def test_configuration_is_frozen_validated_and_serializable():
    training = TrainingConfig(scientific=_scientific(global_batch_size=16, max_steps=8))

    assert TrainingConfig.model_validate_json(training.model_dump_json()) == training
    with pytest.raises(ValidationError, match="frozen"):
        training.scientific.seed = 9
    with pytest.raises(ValidationError, match="valid integer"):
        ScientificConfig(
            task="retrieval/mnr",
            global_batch_size="sixteen",  # type: ignore[arg-type]
            max_steps=8,
            seed=7,
        )
