"""Public launcher contract for paper experiment 01."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "experiments" / "01-dense-model-training" / "run.py"
)


def _experiment() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_run_command_pins_the_accepted_scientific_contract(tmp_path: Path) -> None:
    experiment = _experiment()
    command = experiment["_run_command"](
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


def test_public_cli_only_exposes_paired_experiment_commands() -> None:
    experiment = _experiment()
    parser = experiment["_parser"]()
    help_text = parser.format_help()
    arguments = parser.parse_args(["run", "--seed", "773", "--gpus", "2", "3"])

    assert "{run,aggregate,campaign}" in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    assert arguments.command == "run"
    assert arguments.seed == 773
    assert arguments.gpus == [2, 3]
