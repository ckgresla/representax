"""General acceptance gates required of checkpoint-backed model families."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .acceptance import MODEL_IMPLEMENTATIONS, compare_model_performance


def test_every_checkpoint_backed_model_has_parity_tests():
    implementation_root = Path("src/representax/models")
    checkpoint_backed = {
        path.parent.name for path in implementation_root.glob("*/checkpoint.py")
    }
    for package in checkpoint_backed:
        tests = Path("tests/models") / package
        assert (tests / "test_model.py").is_file()
        assert (tests / "test_transformers_parity.py").is_file()


def test_every_performance_case_has_a_probe():
    for package in {case.package for case in MODEL_IMPLEMENTATIONS}:
        assert (Path("tests/models") / package / "performance_probe.py").is_file()


@pytest.mark.performance
@pytest.mark.parametrize("case", MODEL_IMPLEMENTATIONS, ids=lambda case: case.name)
def test_model_implementation_beats_upstream(case, tmp_path):
    checkpoint_value = os.environ.get(case.checkpoint_environment)
    if checkpoint_value is None:
        pytest.skip(f"set {case.checkpoint_environment} for the performance gate")
    result = compare_model_performance(case, Path(checkpoint_value), tmp_path)
    print(json.dumps(result, indent=2, sort_keys=True))
