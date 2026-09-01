"""Backward compatibility for import targets stored in accepted artifacts."""

from __future__ import annotations

import importlib
from pathlib import Path


def test_former_paper_module_prefix_resolves_without_an_old_directory() -> None:
    module = importlib.import_module("experiments.paper.dense_retrieval")

    assert Path(module.__file__).resolve() == (
        Path(__file__).parents[2] / "experiments/preflights/dense_retrieval.py"
    ).resolve()
    assert not (Path(__file__).parents[2] / "experiments/paper").exists()
