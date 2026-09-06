from __future__ import annotations

from types import SimpleNamespace

import pytest
from experiments.preflights.accelerator import data_parallel_job


@pytest.mark.parametrize(
    ("global_batch", "preferred_micro", "devices", "micro", "accumulation"),
    (
        (256, 32, 16, 16, 1),
        (128, 32, 16, 8, 1),
        (64, 8, 16, 4, 1),
        (256, 6, 16, 4, 4),
    ),
)
def test_data_parallel_job_preserves_global_batch(
    global_batch: int,
    preferred_micro: int,
    devices: int,
    micro: int,
    accumulation: int,
) -> None:
    from representax.config import BatchConfig, TrainingConfig

    job = SimpleNamespace(
        training=TrainingConfig(
            global_batch_size=global_batch,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=preferred_micro),
        )
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    result = data_parallel_job(job, device_count=devices)

    assert result.training.global_batch_size == global_batch
    assert result.training.mesh.axis_shapes == (devices,)
    assert result.training.batch.micro_batch_size == micro
    assert result.training.batch.gradient_accumulation_steps == accumulation


def test_data_parallel_job_rejects_fractional_local_batches() -> None:
    from representax.config import BatchConfig, TrainingConfig

    job = SimpleNamespace(
        training=TrainingConfig(
            global_batch_size=10,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=2),
        )
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    with pytest.raises(ValueError, match="does not divide"):
        data_parallel_job(job, device_count=4)
