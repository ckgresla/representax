from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest


def _experiment():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "07-bert-parameter-scaling"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("bert_parameter_scaling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_original_bert_schedule_uses_128_then_512_for_the_final_ten_percent():
    experiment = _experiment()

    assert experiment.sequence_length_for_step(1, long_sequence_start=901) == 128
    assert experiment.sequence_length_for_step(900, long_sequence_start=901) == 128
    assert experiment.sequence_length_for_step(901, long_sequence_start=901) == 512
    assert experiment.sequence_length_for_step(1_000, long_sequence_start=901) == 512


def test_phase_summary_excludes_the_first_compiled_step_of_each_shape():
    experiment = _experiment()
    rows = tuple((step, float(step)) for step in range(1, 7))

    summary = experiment._phase_summary(
        rows,
        batch_size=8,
        long_sequence_start=5,
    )

    assert summary["sequence-128"]["warmed_steps"] == 3
    assert summary["sequence-128"]["median_step_seconds"] == 3.0
    assert summary["sequence-512"]["warmed_steps"] == 1
    assert summary["sequence-512"]["median_step_seconds"] == 6.0


def test_accelerator_summary_reports_mean_and_maximum():
    experiment = _experiment()

    assert experiment._accelerator_summary(
        [
            {
                "accelerator/0/utilization_percent": 50,
                "accelerator/0/power_watts": 200.0,
            },
            {
                "accelerator/0/utilization_percent": 100,
                "accelerator/0/power_watts": 300.0,
            },
        ]
    ) == {
        "accelerator/0/power_watts": {"mean": 250.0, "maximum": 300.0},
        "accelerator/0/utilization_percent": {"mean": 75.0, "maximum": 100.0},
    }


def test_native_lifecycle_uses_fresh_process_for_resume(monkeypatch):
    experiment = _experiment()
    monkeypatch.setattr(sys, "argv", ["run.py", "worker", "--lifecycle"])

    initial, resume = experiment._native_lifecycle_commands()

    assert initial[-2:] == ["--native-stage", "initial"]
    assert resume[-2:] == ["--native-stage", "resume"]
    assert initial[:2] == [sys.executable, str(Path(experiment.__file__).resolve())]


def test_distributed_output_path_includes_world_size(tmp_path):
    experiment = _experiment()

    def output(gpus):
        return experiment._pair_output(
            Namespace(
                artifact_root=tmp_path,
                lifecycle=False,
                size="bert-4b",
                topology="fsdp",
                sequential=True,
                gpus=gpus,
                steps=6,
                batch_size=60,
                representax_chunk_size=1,
                reference_chunk_size=1,
                rematerialization="full",
            )
        )

    assert output([0, 1, 2, 3, 4]) != output([0, 1, 2, 3, 4, 5])
    assert "fsdp-5gpu-sequential" in str(output([0, 1, 2, 3, 4]))


def test_4b_native_job_shards_scanned_linear_contracted_axes(tmp_path):
    experiment = _experiment()

    job = experiment._native_job(
        size="bert-4b",
        checkpoint=tmp_path / "checkpoint",
        training_data=tmp_path / "training.jsonl",
        steps=1_000,
        long_sequence_start=901,
        batch_size=64,
        chunk_size=1,
        topology="fsdp",
        world_size=4,
        rematerialization="full",
        lifecycle=True,
    )

    sharding = job.training.sharding
    assert sharding.kind == "custom"
    assert sharding.data_axis == "data"
    assert sharding.parameter_axes == ("data",)
    assert tuple(sharding.parameter_rules[1].axes) == (None, "data", None)


def test_sub_4b_native_job_retains_generic_fsdp(tmp_path):
    experiment = _experiment()

    job = experiment._native_job(
        size="bert-1b",
        checkpoint=tmp_path / "checkpoint",
        training_data=tmp_path / "training.jsonl",
        steps=1_000,
        long_sequence_start=901,
        batch_size=64,
        chunk_size=4,
        topology="fsdp",
        world_size=4,
        rematerialization="selective",
        lifecycle=True,
    )

    assert job.training.sharding.kind == "fsdp"


@pytest.mark.parametrize(
    ("steps", "long_start"),
    ((2, 2), (10, 1), (10, 11)),
)
def test_invalid_length_schedules_are_rejected(steps, long_start):
    experiment = _experiment()

    with pytest.raises(ValueError):
        experiment._validate_schedule(
            steps=steps,
            long_sequence_start=long_start,
        )
