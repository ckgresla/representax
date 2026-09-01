"""Shared environment contract for numbered paper experiments."""

from __future__ import annotations

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
    dependencies = set(environment["project"]["dependencies"])
    sources = environment["tool"]["uv"]["sources"]
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    packages = {package["name"]: package for package in lock["package"]}

    assert sources["representax"] == {"path": "..", "editable": True}
    assert "representax[config,hf,performance,wandb,cuda12]" in dependencies
    assert "torch==2.11.0" in dependencies
    assert packages["representax"]["source"] == {"editable": "../"}
    assert packages["torch"]["version"] == "2.11.0+cu128"
    assert "jax-cuda12-plugin" in packages
    assert all("cu13" not in name for name in packages)
