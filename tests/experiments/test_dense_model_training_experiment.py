"""Public launcher contract for paper experiment 01."""

from __future__ import annotations

import runpy
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "experiments" / "01-dense-model-training" / "run.py"
)


def _experiment() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_pair_command_pins_the_accepted_scientific_contract(tmp_path: Path) -> None:
    experiment = _experiment()
    command = experiment["_pair_command"](
        42,
        (0, 1),
        tmp_path,
        Path("/model"),
        Path("/data"),
    )

    assert command[:4] == [
        experiment["sys"].executable,
        "-m",
        "benchmarks.dense_retrieval",
        "pair",
    ]
    assert command[command.index("--batch-size") + 1] == "2048"
    assert command[command.index("--steps") + 1] == "256"
    assert command[command.index("--cache-chunk-size") + 1] == "32"
    assert command[command.index("--seed") + 1] == "42"
    assert command[-4:] == [
        "--representax-gpu",
        "0",
        "--sentence-transformers-gpu",
        "1",
    ]
    assert [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--sequence-length-bucket"
    ] == ["16", "32", "64", "128", "256"]


def test_aggregate_command_uses_all_three_quality_seeds(tmp_path: Path) -> None:
    command = _experiment()["_aggregate_command"](tmp_path)

    summaries = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--summary"
    ]
    assert summaries == [
        str(tmp_path / "runs" / f"seed-{seed}" / "summary.json")
        for seed in (7, 42, 773)
    ]
    assert command[-2:] == [
        "--output",
        str(tmp_path / "three-seed-summary.json"),
    ]


def test_reference_training_and_shared_evaluation_are_public_commands(
    tmp_path: Path,
) -> None:
    experiment = _experiment()
    worker = experiment["_worker_command"](
        "sentence-transformers",
        773,
        tmp_path,
        Path("/model"),
        Path("/data"),
    )
    evaluation = experiment["_evaluation_command"](
        "sentence-transformers",
        Path("/export"),
        tmp_path / "evaluation.json",
        Path("/data"),
    )

    assert worker[worker.index("--framework") + 1] == "sentence-transformers"
    assert worker[worker.index("--seed") + 1] == "773"
    assert evaluation[3] == "offline-evaluate"
    assert evaluation[evaluation.index("--artifact-kind") + 1] == (
        "sentence-transformers"
    )
    assert evaluation[evaluation.index("--artifact") + 1] == "/export"
