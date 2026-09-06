from __future__ import annotations

from types import SimpleNamespace

import pytest
from experiments.preflights.accelerator import data_parallel_job, use_fixed_text_padding


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
        ),
        logging=SimpleNamespace(
            accelerator=True,
            model_copy=lambda *, update: SimpleNamespace(**update),
        ),
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    result = data_parallel_job(
        job,
        device_count=devices,
        platform="tpu",
    )

    assert result.training.global_batch_size == global_batch
    assert result.training.mesh.axis_shapes == (devices,)
    assert result.training.batch.micro_batch_size == micro
    assert result.training.batch.gradient_accumulation_steps == accumulation
    assert not result.logging.accelerator


def test_data_parallel_job_rejects_fractional_local_batches() -> None:
    from representax.config import BatchConfig, TrainingConfig

    job = SimpleNamespace(
        training=TrainingConfig(
            global_batch_size=10,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=2),
        ),
        logging=SimpleNamespace(
            accelerator=False,
            model_copy=lambda *, update: SimpleNamespace(**update),
        ),
    )
    job.model_copy = lambda *, update: SimpleNamespace(**update)

    with pytest.raises(ValueError, match="does not divide"):
        data_parallel_job(job, device_count=4)


def test_fixed_text_padding_preserves_other_processing_settings() -> None:
    module = SimpleNamespace(
        processing_kwargs={
            "text": {"padding_side": "right"},
            "audio": {"sampling_rate": 16_000},
        }
    )

    use_fixed_text_padding([module], 256)

    assert module.processing_kwargs == {
        "text": {
            "padding_side": "right",
            "padding": "max_length",
            "truncation": True,
            "max_length": 256,
        },
        "audio": {"sampling_rate": 16_000},
    }
