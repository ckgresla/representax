"""General acceptance gates required of checkpoint-backed model families."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from representax.integrations import MODEL_FAMILIES, AcceptanceGate

from .acceptance import MODEL_IMPLEMENTATIONS, compare_model_performance


def test_every_checkpoint_backed_model_has_an_acceptance_registration():
    implementation_root = Path("src/representax/models")
    checkpoint_backed = {
        path.parent.name for path in implementation_root.glob("*/checkpoint.py")
    }
    registered = {case.package for case in MODEL_IMPLEMENTATIONS}
    assert registered == checkpoint_backed
    performance_claims = {
        family.name
        for family in MODEL_FAMILIES.values()
        if AcceptanceGate.PERFORMANCE in family.acceptance_gates
    }
    assert registered == performance_claims
    for case in MODEL_IMPLEMENTATIONS:
        tests = Path("tests/models") / case.package
        assert (tests / "test_model.py").is_file()
        assert (tests / "test_transformers_parity.py").is_file()
        assert (tests / "performance_probe.py").is_file()


@pytest.mark.performance
@pytest.mark.parametrize("case", MODEL_IMPLEMENTATIONS, ids=lambda case: case.name)
def test_model_implementation_beats_upstream(case, tmp_path):
    checkpoint_value = os.environ.get(case.checkpoint_environment)
    if checkpoint_value is None:
        pytest.skip(f"set {case.checkpoint_environment} for the performance gate")
    result = compare_model_performance(case, Path(checkpoint_value), tmp_path)
    print(json.dumps(result, indent=2, sort_keys=True))
