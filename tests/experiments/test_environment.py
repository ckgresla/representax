"""Shared environment contract for numbered paper experiments."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

EXPERIMENTS = Path(__file__).parents[2] / "experiments"
SETUP = EXPERIMENTS / "setup.sh"
ENVIRONMENT = EXPERIMENTS / "pyproject.toml"
LOCK = EXPERIMENTS / "uv.lock"


def test_setup_script_is_syntactically_valid() -> None:
    subprocess.run(("bash", "-n", SETUP), check=True)


def test_locked_environment_uses_local_representax_and_one_cuda_stack() -> None:
    environment = tomllib.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    groups = environment["dependency-groups"]
    sources = environment["tool"]["uv"]["sources"]
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}

    assert sources["representax"] == {"path": "..", "editable": True}
    assert "representax[config,hf,performance,wandb,cuda12]" in groups["native"]
    assert "torch==2.11.0" in groups["vjepa"]
    assert packages["representax"]["source"] == {"editable": "../"}
    assert packages["torch"]["version"] == "2.11.0+cu128"
    assert "jax-cuda12-plugin" in packages
    assert all("cu13" not in name for name in packages)


def test_named_groups_cover_every_reference_runtime() -> None:
    environment = tomllib.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    groups = environment["dependency-groups"]

    assert set(groups) == {
        "main",
        "late-interaction",
        "lejepa",
        "native",
        "reward",
        "sentence-transformers",
        "vjepa",
    }
    assert environment["tool"]["uv"]["default-groups"] == ["main"]
    assert {entry["include-group"] for entry in groups["main"]} == {
        "native",
        "sentence-transformers",
        "reward",
        "lejepa",
        "vjepa",
    }


def test_reference_versions_and_sources_match_the_frozen_manifest() -> None:
    environment = tomllib.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    groups = environment["dependency-groups"]
    manifest = json.loads(
        (EXPERIMENTS.parent / "benchmarks/configs/paper-references-v1.json").read_text(
            encoding="utf-8"
        )
    )["references"]

    assert (
        f"sentence-transformers=={manifest['sentence-transformers']['release']}"
        in groups["sentence-transformers"]
    )
    assert f"pylate=={manifest['pylate']['release']}" in groups["late-interaction"]
    assert f"trl=={manifest['trl']['release']}" in groups["reward"]
    sources = environment["tool"]["uv"]["sources"]
    assert sources["lejepa"]["rev"] == manifest["lejepa-paper"]["commit"]
    assert (
        sources["stable-pretraining"]["rev"] == manifest["stable-pretraining"]["commit"]
    )


def test_pylate_is_an_explicitly_conflicting_environment() -> None:
    environment = tomllib.loads(ENVIRONMENT.read_text(encoding="utf-8"))
    pairs = {
        frozenset(item["group"] for item in conflict)
        for conflict in environment["tool"]["uv"]["conflicts"]
    }

    assert pairs == {
        frozenset(("native", "late-interaction")),
        frozenset(("sentence-transformers", "late-interaction")),
    }
