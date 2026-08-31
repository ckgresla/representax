#!/usr/bin/env python3
"""Validate the paper-frozen controlled BERT scaling family."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from representax.models.bert import BertConfig

SCHEMA_VERSION = "representax-bert-scaling-v1"
EXPECTED_NAMES = ("bert-30m", "bert-110m", "bert-500m", "bert-1b", "bert-4b")
DEFAULT_MANIFEST = (
    Path(__file__).parents[1] / "benchmarks" / "configs" / "bert-scaling-v1.json"
)


def bert_parameter_count(config: BertConfig) -> int:
    """Return the exact number of parameters allocated by ``BertEncoder``."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    embeddings = (
        config.vocab_size + config.max_position_embeddings + config.type_vocab_size
    ) * hidden + 2 * hidden
    pooler = hidden * hidden + hidden
    layer = 4 * hidden * hidden + 2 * hidden * intermediate + intermediate + 9 * hidden
    return embeddings + pooler + config.num_hidden_layers * layer


def _config(shared: Mapping[str, Any], size: Mapping[str, Any]) -> BertConfig:
    values = {
        key: value
        for key, value in shared.items()
        if key not in {"head_dimension", "intermediate_multiplier"}
    }
    values.update(
        {
            key: size[key]
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
            )
        }
    )
    return BertConfig(**values)


def validate_manifest(document: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Validate invariants and return normalized records for reporting."""

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}")
    if document.get("family") != "bert":
        raise ValueError("controlled scaling family must be BERT")
    shared = document.get("shared")
    sizes = document.get("sizes")
    if not isinstance(shared, Mapping) or not isinstance(sizes, Sequence):
        raise TypeError("manifest requires shared configuration and ordered sizes")
    names = tuple(size.get("name") for size in sizes if isinstance(size, Mapping))
    if names != EXPECTED_NAMES:
        raise ValueError(f"size order must be {EXPECTED_NAMES!r}; received {names!r}")

    head_dimension = int(shared["head_dimension"])
    intermediate_multiplier = int(shared["intermediate_multiplier"])
    records = []
    previous_count = 0
    for size in sizes:
        if not isinstance(size, Mapping):
            raise TypeError("each size must be an object")
        config = _config(shared, size)
        if config.head_dimension != head_dimension:
            raise ValueError(f"{size['name']} does not preserve head dimension")
        if config.intermediate_size != intermediate_multiplier * config.hidden_size:
            raise ValueError(f"{size['name']} does not preserve MLP expansion")
        count = bert_parameter_count(config)
        if count != int(size["expected_parameters"]):
            raise ValueError(
                f"{size['name']} has {count} parameters, not "
                f"{size['expected_parameters']}"
            )
        if count <= previous_count:
            raise ValueError("parameter counts must increase monotonically")
        target = int(size["target_parameters"])
        if abs(count - target) / target > 0.05:
            raise ValueError(f"{size['name']} is more than 5% from its target")
        if size["name"] == "bert-4b" and not size.get("admission_gate"):
            raise ValueError("bert-4b requires an explicit admission gate")
        previous_count = count
        records.append(
            {
                "name": size["name"],
                "parameters": count,
                "hidden_size": config.hidden_size,
                "layers": config.num_hidden_layers,
                "attention_heads": config.num_attention_heads,
            }
        )
    return tuple(records)


def load_and_validate(path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], ...]:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()
    print(json.dumps(load_and_validate(arguments.manifest), indent=2))


if __name__ == "__main__":
    main()
