#!/usr/bin/env python3
"""Validate the frozen multimodal retrieval and V-JEPA paper panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "configs"
    / "paper-multimodal-jepa-v1.json"
)
EXPECTED_WORKLOADS = (
    "image-text-retrieval",
    "audio-text-retrieval",
    "video-text-retrieval",
    "vjepa2-1-video-representation",
)


def _objects(value: Any, name: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(record, Mapping) for record in value.values()):
        raise TypeError(f"every {name} record must be an object")
    return value


def _sha(value: Any, name: str) -> None:
    revision = str(value)
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(f"{name} must use a full lowercase commit SHA")


def validate_manifest(document: Mapping[str, Any]) -> tuple[str, ...]:
    if document.get("schema_version") != "representax-paper-multimodal-jepa-v1":
        raise ValueError("unsupported multimodal/V-JEPA schema")
    models = _objects(document.get("models"), "models")
    datasets = _objects(document.get("datasets"), "datasets")
    references = _objects(
        {
            name: {"version": version}
            for name, version in _objects_or_scalars(
                document.get("references"), "references"
            ).items()
        },
        "references",
    )
    for name, artifact in (*models.items(), *datasets.items()):
        _sha(artifact.get("revision"), f"{name} revision")
        if not artifact.get("license"):
            raise ValueError(f"{name} requires a license disclosure")
    workloads = document.get("workloads")
    if not isinstance(workloads, Sequence):
        raise TypeError("workloads must be an ordered array")
    names = tuple(str(workload.get("name")) for workload in workloads)
    if names != EXPECTED_WORKLOADS:
        raise ValueError(f"workloads must be ordered as {EXPECTED_WORKLOADS!r}")
    for workload in workloads:
        if workload.get("model") not in models:
            raise ValueError(f"{workload['name']} references an unknown model")
        for field in ("train", "evaluate", "secondary_evaluate"):
            dataset = workload.get(field)
            if dataset is not None and dataset not in datasets:
                raise ValueError(f"{workload['name']} references unknown {field}")
        if workload.get("reference") not in references:
            raise ValueError(f"{workload['name']} references an unknown framework")
    if models["vjepa2-1-vit-b16"].get("initialization") != "random":
        raise ValueError("V-JEPA systems comparison must start from random init")
    return names


def _objects_or_scalars(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def load_and_validate(path: Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    print(json.dumps(load_and_validate(arguments.manifest), indent=2))


if __name__ == "__main__":
    main()
