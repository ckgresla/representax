#!/usr/bin/env python3
"""Validate immutable identities and references in paper workload manifests."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = (
    Path(__file__).parents[1] / "benchmarks" / "configs" / "paper-text-reward-v1.json"
)
REFERENCE_MANIFEST = DEFAULT_MANIFEST.with_name("paper-references-v1.json")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    revision = str(value)
    invalid_character = any(
        character not in "0123456789abcdef" for character in revision
    )
    if len(revision) != 40 or invalid_character:
        raise ValueError(f"{name} must use a full lowercase commit SHA")
    return revision


def validate_manifest(document: Mapping[str, Any]) -> tuple[str, ...]:
    if document.get("schema_version") != "representax-paper-text-reward-v1":
        raise ValueError("unsupported paper text/reward schema")
    models = _mapping(document.get("models"), "models")
    datasets = _mapping(document.get("datasets"), "datasets")
    references = _mapping(
        json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8")).get("references"),
        "references",
    )
    workloads = document.get("workloads")
    if not isinstance(workloads, Sequence):
        raise TypeError("workloads must be an ordered array")

    for name, value in models.items():
        artifact = _mapping(value, f"model {name}")
        if not artifact.get("repo_id"):
            raise ValueError(f"model {name} requires repo_id")
        if not artifact.get("license"):
            raise ValueError(f"model {name} requires a license disclosure")
        _sha(artifact.get("revision"), f"model {name} revision")
    for name, value in datasets.items():
        artifact = _mapping(value, f"dataset {name}")
        if not artifact.get("license"):
            raise ValueError(f"dataset {name} requires a license disclosure")
        if "repo_id" in artifact:
            _sha(artifact.get("revision"), f"dataset {name} revision")
        elif not all(
            artifact.get(field) for field in ("source", "source_version", "dataset_id")
        ):
            raise ValueError(f"dataset {name} requires a pinned external source")

    names = []
    for value in workloads:
        workload = _mapping(value, "workload")
        name = str(workload.get("name", ""))
        if not name or name in names:
            raise ValueError("workload names must be non-empty and unique")
        names.append(name)
        for model_field in ("model", "architecture_control"):
            model = workload.get(model_field)
            if model is not None and model not in models:
                raise ValueError(f"{name} references unknown model {model!r}")
        for split in ("train", "evaluate"):
            keys = workload.get(split)
            if not isinstance(keys, Sequence) or isinstance(keys, str) or not keys:
                raise ValueError(f"{name} requires at least one {split} dataset")
            unknown = set(keys).difference(datasets)
            if unknown:
                raise ValueError(
                    f"{name} references unknown datasets {sorted(unknown)}"
                )
        reference = workload.get("reference")
        if not isinstance(reference, str):
            raise ValueError(f"{name} requires a reference framework")
        if reference not in references:
            raise ValueError(f"{name} references an unknown framework")
    return tuple(names)


def load_and_validate(path: Path = DEFAULT_MANIFEST) -> tuple[str, ...]:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    print(json.dumps(load_and_validate(arguments.manifest), indent=2))


if __name__ == "__main__":
    main()
