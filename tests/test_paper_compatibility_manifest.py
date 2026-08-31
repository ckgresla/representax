from __future__ import annotations

import json
from pathlib import Path

from experiments.paper.compatibility import _read_manifest, _recipe


def test_paper_compatibility_panel_has_25_pinned_checkpoints() -> None:
    root = Path(__file__).parents[1]
    path = root / "benchmarks/configs/paper-compatibility-v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == "representax-paper-compatibility-v1"
    models = document["models"]
    assert len(models) == 25
    assert len({model["name"] for model in models}) == 25
    assert len({model["repo_id"] for model in models}) == 25
    for model in models:
        revision = model["revision"]
        assert len(revision) == 40
        assert set(revision) <= set("0123456789abcdef")
        assert model["family"]
        assert model["modalities"]
        assert set(model) == {
            "name",
            "repo_id",
            "revision",
            "family",
            "modalities",
        }
    modalities = {model["name"]: model["modalities"] for model in models}
    assert modalities["lco-embedding-omni-3b"] == [
        "text",
        "image",
        "audio",
        "video",
    ]
    assert modalities["omni-embed-nemotron-3b"] == ["text", "image"]
    assert modalities["bidirlm-omni-2.5b-embedding"] == [
        "text",
        "image",
        "audio",
    ]


def test_every_pinned_checkpoint_has_one_executable_lifecycle_recipe() -> None:
    entries = _read_manifest()

    recipes = tuple(_recipe(entry) for entry in entries)

    assert len(recipes) == 25
    assert all(recipe.loader.startswith("representax.") for recipe in recipes)
    imported = {
        entry.name: recipe.adapter_target
        for entry, recipe in zip(entries, recipes, strict=True)
        if entry.name.startswith("nomic-embed-multimodal-")
    }
    assert imported == {
        "nomic-embed-multimodal-3b": None,
        "nomic-embed-multimodal-7b": None,
    }
    assert (
        next(
            recipe.devices
            for entry, recipe in zip(entries, recipes, strict=True)
            if entry.name == "nomic-embed-multimodal-7b"
        )
        == 4
    )
    assert all(
        recipe.adapter_target
        for entry, recipe in zip(entries, recipes, strict=True)
        if entry.name not in imported
    )
