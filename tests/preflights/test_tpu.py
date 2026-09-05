from __future__ import annotations

import jax
import pytest
from experiments.preflights.tpu import TOY_BATCH_SIZE, _job, run, variants


def test_tpu_acceptance_matrix_covers_core_training_paths() -> None:
    rows = {variant.name: variant for variant in variants()}

    assert set(rows) == {
        "pairwise-direct",
        "pairwise-bf16",
        "pairwise-gradient-accumulation",
        "pairwise-ddp",
        "pairwise-fsdp",
        "retrieval-direct",
        "retrieval-grad-cache-rematerialized",
        "retrieval-grad-cache-custom-vjp",
    }
    assert rows["pairwise-direct"].telemetry is True


def test_tpu_acceptance_jobs_preserve_global_batch() -> None:
    for variant in variants():
        job = _job(variant, device_count=4, steps=6)
        data_replicas = 4 if variant.sharding == "ddp" else 1
        assert (
            job.training.batch.micro_batch_size
            * job.training.batch.gradient_accumulation_steps
            * data_replicas
            == TOY_BATCH_SIZE
        )
        assert job.checkpointing is not None
        assert job.evaluation is not None
        assert job.export.enabled is True


@pytest.mark.runtime
@pytest.mark.distributed
def test_tpu_acceptance_runner_completes_on_multiple_devices(tmp_path) -> None:
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two JAX devices")

    result = run(tmp_path / "acceptance", steps=4, device_count=2)

    assert result["accepted"] is True
    assert len(result["variants"]) == 8
    assert all(row["accepted"] for row in result["parity"].values())
