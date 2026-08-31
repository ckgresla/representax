from __future__ import annotations

import json

import pytest
from scripts.validate_paper_panel import (
    DEFAULT_MANIFEST,
    load_and_validate,
    validate_manifest,
)


def test_frozen_text_and_reward_panel() -> None:
    assert load_and_validate() == (
        "dense-realistic",
        "dense-natural-questions",
        "dense-multilingual",
        "semantic-similarity",
        "pair-classification",
        "cross-encoder-reranking",
        "late-interaction",
        "pairwise-outcome-reward",
        "process-reward",
    )


def test_natural_questions_and_miracl_are_pinned() -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    workloads = {row["name"]: row for row in document["workloads"]}

    assert workloads["dense-natural-questions"]["train"] == [
        "natural-questions-train-pairs"
    ]
    multilingual = workloads["dense-multilingual"]
    assert multilingual["languages"] == ["ar", "en", "hi", "ja"]
    assert multilingual["report_each_language"] is True


def test_panel_rejects_floating_revision() -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    document["models"]["qwen3-reward"]["revision"] = "main"

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        validate_manifest(document)


def test_panel_rejects_unknown_artifact() -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    document["workloads"][0]["train"] = ["missing"]

    with pytest.raises(ValueError, match="unknown datasets"):
        validate_manifest(document)
