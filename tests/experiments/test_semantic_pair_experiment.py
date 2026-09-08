"""Public launcher contract for paper experiment 02."""

from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "02-semantic-similarity-pair-classification"
    / "run.py"
)
SHELL_RUNNER = SCRIPT.with_name("run.sh")


def _experiment() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_training_steps_are_four_complete_epochs(tmp_path: Path) -> None:
    experiment = _experiment()
    (tmp_path / "semantic-train.jsonl").write_text("{}\n" * 5749)
    (tmp_path / "pair-train.jsonl").write_text("{}\n" * 2048)

    assert experiment["training_steps"]("semantic-similarity", tmp_path) == 88
    assert experiment["training_steps"]("pair-classification", tmp_path) == 32


def test_worker_command_pins_model_task_seed_and_budget(tmp_path: Path) -> None:
    experiment = _experiment()
    command = experiment["_worker_command"](
        workload="semantic-similarity",
        model="bert-base",
        framework="representax",
        checkpoint=Path("/model"),
        data=Path("/data"),
        output=tmp_path,
        steps=88,
        seed=773,
    )

    assert command[command.index("--model") + 1] == "bert-base"
    assert "--serious" in command
    assert command[command.index("--workload") + 1] == "semantic-similarity"
    assert command[command.index("--framework") + 1] == "representax"
    assert command[command.index("--steps") + 1] == "88"
    assert command[command.index("--seed") + 1] == "773"


def test_aggregate_requires_all_paired_cells_and_timing_reports(tmp_path: Path) -> None:
    experiment = _experiment()
    summaries = experiment["_expected_summaries"](tmp_path)
    timing = experiment["_expected_timing_reports"](tmp_path)

    assert len(summaries) == 12
    assert len(timing) == 12
    assert {path.parts[-4:-1] for path in summaries} == {
        (workload, model, f"seed-{seed}")
        for workload in ("semantic-similarity", "pair-classification")
        for model in ("mpnet-base", "bert-base")
        for seed in (7, 42, 773)
    }


def test_shell_runner_schedules_six_single_gpu_workers(tmp_path: Path) -> None:
    subprocess.run(("bash", "-n", SHELL_RUNNER), check=True)
    help_result = subprocess.run(
        (SHELL_RUNNER, "-h"),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "-g GPU_IDS" in help_result.stdout

    calls = tmp_path / "calls"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${CALLS}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    subprocess.run(
        (SHELL_RUNNER, "-g", "0,1,2,3,4,5"),
        check=True,
        env={
            "CALLS": str(calls),
            "PATH": "/usr/bin:/bin",
            "REPRESENTAX_EXPERIMENT_PYTHON": str(fake_python),
            "REPRESENTAX_PAPER_ROOT": str(tmp_path / "paper"),
        },
    )

    invocations = calls.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 26
    assert sum(" run " in invocation for invocation in invocations) == 12
    assert sum(" reference-timing " in invocation for invocation in invocations) == 12
    assert invocations[-1].endswith(" aggregate")
