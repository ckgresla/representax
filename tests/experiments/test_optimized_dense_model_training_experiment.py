"""Public launcher contract for the execution-tuned dense experiment."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "01.1-optimized-dense-model-training"
    / "run.py"
)


def _experiment() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_launcher_executes_directly() -> None:
    result = subprocess.run(
        (sys.executable, SCRIPT, "--help"),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "{tune,run}" in result.stdout


def test_candidate_commands_keep_the_scientific_contract_fixed(tmp_path: Path) -> None:
    experiment = _experiment()
    candidate = experiment["Candidate"]("representax", 64, 8, 16)
    command = experiment["_candidate_command"](
        candidate,
        checkpoint=Path("/model"),
        data=Path("/data"),
        seed=7,
        steps=16,
        directory=tmp_path,
    )

    assert command[command.index("--batch-size") + 1] == "2048"
    assert command[command.index("--maximum-length") + 1] == "256"
    assert command[command.index("--cache-chunk-size") + 1] == "64"
    assert command[command.index("--data-threads") + 1] == "8"
    assert command[command.index("--seed") + 1] == "7"
    assert "--mixed-precision" in command
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--sequence-length-bucket"
    ] == ["16", "32", "64", "128", "256"]


def test_projected_rate_preserves_cold_overhead() -> None:
    experiment = _experiment()
    report = {
        "batch_size": 2048,
        "steps": 16,
        "steady_state_seconds": 8.0,
        "steady_state_step_count": 8,
        "training_compute_seconds": 20.0,
    }

    assert experiment["_projected_rate"](report, target_steps=256) == pytest.approx(
        2016.4923076923078
    )


def test_final_command_uses_independently_selected_winners(tmp_path: Path) -> None:
    experiment = _experiment()
    tuning = tmp_path / "tuning"
    tuning.mkdir()
    (tuning / "summary.json").write_text(
        json.dumps(
            {
                "winners": {
                    "representax": {
                        "framework": "representax",
                        "cache_chunk_size": 64,
                        "data_threads": 8,
                        "prefetch_buffer_size": 16,
                        "persistent_workers": False,
                        "torch_compile": False,
                    },
                    "sentence-transformers": {
                        "framework": "sentence-transformers",
                        "cache_chunk_size": 128,
                        "data_threads": 4,
                        "prefetch_buffer_size": 8,
                        "persistent_workers": True,
                        "torch_compile": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    arguments = experiment["argparse"].Namespace(
        artifact_root=tmp_path,
        checkpoint=Path("/model"),
        data=Path("/data"),
        seed=7,
        gpus=[0, 1],
    )

    command = experiment["_final_command"](arguments)

    assert command[command.index("--representax-cache-chunk-size") + 1] == "64"
    assert (
        command[command.index("--sentence-transformers-cache-chunk-size") + 1] == "128"
    )
    assert "--sentence-transformers-persistent-workers" in command
    assert "--sentence-transformers-torch-compile" in command
    assert command[command.index("--steps") + 1] == "256"
    assert command[command.index("--evaluation-every-steps") + 1] == "64"
    assert "--export" in command
