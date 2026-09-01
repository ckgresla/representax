"""Public launcher contract for paper experiment 01."""

from __future__ import annotations

import runpy
import subprocess
import tomllib
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2] / "experiments" / "01-dense-model-training" / "run.py"
)
SHELL_RUNNER = SCRIPT.with_name("run.sh")
SETUP = SCRIPT.with_name("setup.sh")
ENVIRONMENT = SCRIPT.with_name("pyproject.toml")
LOCK = SCRIPT.with_name("uv.lock")


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

    assert "{run,aggregate}" in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["campaign"])
    assert arguments.command == "run"
    assert arguments.seed == 773
    assert arguments.gpus == [2, 3]


def test_shell_files_are_syntactically_valid() -> None:
    for script in (SHELL_RUNNER, SETUP):
        subprocess.run(("bash", "-n", script), check=True)


def test_locked_environment_uses_one_cuda_stack() -> None:
    environment = tomllib.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    dependencies = set(environment["project"]["dependencies"])
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}

    assert "representax[config,hf,performance,wandb,cuda12]" in dependencies
    assert "torch==2.11.0" in dependencies
    assert packages["torch"]["version"] == "2.11.0+cu128"
    assert "jax-cuda12-plugin" in packages
    assert all("cu13" not in name for name in packages)


@pytest.mark.parametrize(
    ("gpus", "expected"),
    [
        (
            ("0", "1"),
            {
                "run --seed 7 --gpus 0 1",
                "run --seed 42 --gpus 0 1",
                "run --seed 773 --gpus 0 1",
            },
        ),
        (
            ("0", "1", "2", "3"),
            {
                "run --seed 7 --gpus 0 1",
                "run --seed 42 --gpus 2 3",
                "run --seed 773 --gpus 0 1",
            },
        ),
        (
            ("0", "1", "2", "3", "4", "5"),
            {
                "run --seed 7 --gpus 0 1",
                "run --seed 42 --gpus 2 3",
                "run --seed 773 --gpus 4 5",
            },
        ),
    ],
)
def test_shell_runner_schedules_seed_pairs_in_waves(
    tmp_path: Path,
    gpus: tuple[str, ...],
    expected: set[str],
) -> None:
    calls = tmp_path / "calls"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${CALLS}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    subprocess.run(
        (SHELL_RUNNER, "--gpus", ",".join(gpus)),
        check=True,
        env={
            "CALLS": str(calls),
            "PATH": "/usr/bin:/bin",
            "REPRESENTAX_EXPERIMENT_PYTHON": str(fake_python),
        },
    )

    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 4
    assert invocations[3].endswith("run.py aggregate")
    assert {
        row.partition("run.py ")[2]
        for row in invocations[:3]
    } == expected
