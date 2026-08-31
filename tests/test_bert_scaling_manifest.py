from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.validate_bert_scaling import (
    DEFAULT_MANIFEST,
    load_and_validate,
    validate_manifest,
)


def test_frozen_bert_scaling_manifest() -> None:
    records = load_and_validate()

    assert [record["parameters"] for record in records] == [
        29_811_072,
        109_482_240,
        486_296_576,
        985_885_440,
        4_177_190_400,
    ]


def test_bert_scaling_manifest_rejects_count_drift(tmp_path: Path) -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    document["sizes"][2]["expected_parameters"] += 1

    with pytest.raises(ValueError, match="has 486296576 parameters"):
        validate_manifest(document)
