from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPOSITORY_ROOT / "experiments/08-five-seed-framework-comparison/run.py"
)


def _launcher() -> ModuleType:
    name = "representax_five_seed_framework_comparison"
    spec = importlib.util.spec_from_file_location(name, LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_every_recipe_routes_its_assigned_gpus(tmp_path: Path) -> None:
    launcher = _launcher()
    for recipe in launcher.RECIPES:
        gpus = (4, 5) if recipe.gpu_count == 2 else (4,)
        command = recipe.command(7, gpus, tmp_path / recipe.name)
        if recipe.name == "dense-retrieval":
            assert _option(command, "--representax-gpu") == "4"
            assert _option(command, "--sentence-transformers-gpu") == "5"
        elif recipe.name.startswith(("semantic-similarity-", "pair-classification-")):
            assert _option(command, "--representax-gpu") == "4"
            assert _option(command, "--reference-gpu") == "5"
        elif recipe.name == "audio-text":
            assert _option(command, "--representax-gpus") == "4"
            assert _option(command, "--reference-gpu") == "4"
        else:
            assert _option(command, "--gpu") == "4"

    audio = next(recipe for recipe in launcher.RECIPES if recipe.name == "audio-text")
    assert audio.serial


def test_loss_comparison_records_direction_and_distance() -> None:
    launcher = _launcher()
    comparison = launcher._loss_comparison([3.0, 2.0, 1.0], [3.0, 2.2, 1.1])
    assert comparison["paired_steps"] == 3
    assert comparison["mean_absolute_difference"] == pytest.approx(0.1)
    assert comparison["final_difference"] == pytest.approx(-0.1)
    assert comparison["direction_agrees"] is True


def test_reference_throughput_excludes_warmup_and_midpoint_checkpoint(
    tmp_path: Path,
) -> None:
    launcher = _launcher()
    timings = [
        {"step": step, "seconds": 101.0 if step == 11 else 1.0}
        for step in range(1, launcher.STEPS + 1)
    ]

    rate, durations = launcher._reference_throughput(
        {},
        {"global_batch_size": 10, "step_timings": timings},
        tmp_path,
    )

    assert len(durations) == launcher.STEPS - 2
    assert rate == pytest.approx(10.0)
