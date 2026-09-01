"""Reference-source provenance contracts for paper jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from experiments.paper import provenance
from experiments.paper.provenance import REFERENCE_MANIFEST, reference_source


def test_every_paper_reference_has_one_immutable_source_record() -> None:
    document = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    assert set(document["references"]) == {
        "lejepa-paper",
        "pylate",
        "sentence-transformers",
        "stable-pretraining",
        "trl",
        "vjepa2",
    }
    for name in document["references"]:
        source = reference_source(name)
        assert source.repository.startswith("https://github.com/")
        assert len(source.commit) == 40


def test_unknown_paper_reference_fails_explicitly() -> None:
    with pytest.raises(KeyError, match="unknown paper reference"):
        reference_source("missing")


def test_reference_results_use_content_addressed_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {
        "schema_version": "representax-reference-provenance-v1",
        "source": {"commit": "a" * 40},
    }
    monkeypatch.setattr(
        provenance,
        "reference_runtime_provenance",
        lambda name, checkout=None: observed,
    )
    report_path = tmp_path / "reference.json"
    report = provenance.write_reference_result(
        report_path,
        {"loss": 1.0},
        reference="sentence-transformers",
    )
    fingerprint = report["reference"]["provenance"]
    sidecar = tmp_path / "provenance" / f"{fingerprint.removeprefix('sha256:')}.json"
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        **observed,
        "fingerprint": fingerprint,
    }
