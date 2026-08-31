from __future__ import annotations

import json

import pytest
from scripts.validate_multimodal_jepa_panel import (
    DEFAULT_MANIFEST,
    load_and_validate,
    validate_manifest,
)


def test_frozen_multimodal_and_jepa_panel() -> None:
    assert load_and_validate() == (
        "image-text-retrieval",
        "audio-text-retrieval",
        "video-text-retrieval",
        "vjepa2-1-video-representation",
    )
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    workloads = {row["name"]: row for row in document["workloads"]}
    audio_model = workloads["audio-text-retrieval"]["model"]
    assert audio_model == workloads["video-text-retrieval"]["model"]
    assert (
        document["models"][audio_model]["repo_id"]
        == "LCO-Embedding/LCO-Embedding-Omni-3B-2605"
    )


def test_panel_rejects_unpinned_dataset() -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    document["datasets"]["audiocaps-train"]["revision"] = "main"

    with pytest.raises(ValueError, match="full lowercase commit SHA"):
        validate_manifest(document)


def test_panel_preserves_random_init_systems_comparison() -> None:
    document = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    document["models"]["vjepa2-1-vit-b16"]["initialization"] = "pretrained"

    with pytest.raises(ValueError, match="must start from random init"):
        validate_manifest(document)
