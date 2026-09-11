"""Bounded cloud orchestration and matched scientific work."""

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def cloud(monkeypatch):
    directory = (
        Path(__file__).resolve().parents[2] / "experiments/15-modernbert-mlm-scaling"
    )
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("mlm_cloud", directory / "cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_work_divides_all_eight_gpu_cases(cloud):
    for seed in cloud.SEEDS:
        for microbatch in cloud.MICROBATCHES:
            config = cloud.configuration(seed, microbatch)
            assert config["tokens_per_update"] == 131072
            assert config["lengths"] == (512,)
            assert config["steps_per_length"] == 21
            assert config["seed"] == seed
            for count in cloud.DEVICES:
                assert 131072 % (512 * microbatch * count) == 0
                assert 131072 // (512 * microbatch * count) >= 1
            tuned = cloud.configuration(seed, microbatch, tuning=True)
            assert tuned.pop("steps_per_length") == 5
            config.pop("steps_per_length")
            assert tuned == config


def test_nvlink_legend_is_not_topology_evidence(cloud):
    assert not cloud.has_nvlink("GPU0 X PHB\nGPU1 PHB X\nNV# = NVLink connections")
    assert cloud.has_nvlink("GPU0 X NV12\nGPU1 NV12 X\nNV# = NVLink connections")


def sample_report():
    observation = dict(
        seconds=2.0,
        finite=True,
        skipped=False,
        loss=3.0,
        tokens_sha256="a",
        corrupted_sha256="b",
        labels_sha256="c",
        positions_sha256="d",
        supervised_tokens=100,
    )
    return dict(
        status="passed",
        configuration=dict(seed=7, tokens_per_update=131072),
        phases=[
            dict(
                global_batch=256,
                compile_seconds=12,
                compiled_required_bytes_per_device=100,
                observations=[dict(observation) for _ in range(21)],
            )
        ],
    )


def test_complete_intervals_and_hash_gate(cloud, tmp_path):
    report = sample_report()
    row = cloud.row_from_report(report, 8, tmp_path)
    assert row["tokens_per_second"] == 65536
    assert row["examples_per_second"] == 128
    cloud.verify_batches(report, report)
    altered = copy.deepcopy(report)
    altered["phases"][0]["observations"][3]["labels_sha256"] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        cloud.verify_batches(report, altered)
    report["phases"][0]["observations"].pop()
    with pytest.raises(ValueError, match="20 finite"):
        cloud.row_from_report(report, 8, tmp_path)


def test_timeout_stops_worker_and_preserves_log(cloud, tmp_path):
    import os

    log = tmp_path / "worker.log"
    with pytest.raises(subprocess.TimeoutExpired):
        cloud.checked_run(
            [
                sys.executable,
                "-u",
                "-c",
                "import time; print('started'); time.sleep(30)",
            ],
            env=os.environ.copy(),
            log=log,
            timeout=0.5,
        )
    assert "started" in log.read_text()
