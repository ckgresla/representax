"""Generate the Torch-free semantic Hugging Face family registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "src/representax/integrations/_transformers_5_3_0.py"
MANIFEST = ROOT / "scripts/model_family_manifest.py"
TARGET = ROOT / "src/representax/integrations/_model_families.py"

MODALITIES = frozenset({"text", "image", "audio", "video"})
SUPPORT_LEVELS = frozenset({"catalogued", "native", "verified"})
ACCEPTANCE_GATES = frozenset(
    {
        "config_mapping",
        "checkpoint_roundtrip",
        "forward",
        "input_gradient",
        "parameter_gradient",
        "optimizer_update",
        "export_reload",
        "performance",
    }
)
VERIFIED_GATES = frozenset(
    {
        "config_mapping",
        "checkpoint_roundtrip",
        "forward",
        "input_gradient",
        "performance",
    }
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nonempty_strings(value: Any, *, field: str, family: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError(f"{family}.{field} must be a non-empty sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{family}.{field} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{family}.{field} contains duplicates")
    return result


def _strings(value: Any, *, field: str, family: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{family}.{field} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{family}.{field} must contain non-empty strings")
    if len(result) != len(set(result)):
        raise ValueError(f"{family}.{field} contains duplicates")
    return result


def validate_manifest(
    families: Sequence[Mapping[str, Any]],
    *,
    catalog_rows: Sequence[tuple[str, str, str, tuple[str, ...]]],
    catalog_sha256: str,
    reference_catalog_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Validate and normalize reviewed family definitions."""

    if reference_catalog_sha256 != catalog_sha256:
        raise ValueError(
            "model-family manifest targets a stale architecture catalogue: "
            f"{reference_catalog_sha256} != {catalog_sha256}"
        )
    catalog = {row[0]: row for row in catalog_rows}
    owners: dict[str, str] = {}
    names: set[str] = set()
    normalized = []
    for raw in families:
        family = raw.get("name")
        if not isinstance(family, str) or not family:
            raise ValueError("each model family requires a non-empty name")
        if family in names:
            raise ValueError(f"duplicate model family {family!r}")
        names.add(family)

        support = raw.get("support")
        if support not in SUPPORT_LEVELS:
            raise ValueError(
                f"{family}.support must be one of {sorted(SUPPORT_LEVELS)}"
            )

        model_types = _nonempty_strings(
            raw.get("model_types"), field="model_types", family=family
        )
        for model_type in model_types:
            if model_type not in catalog:
                raise ValueError(f"{family} owns unknown model type {model_type!r}")
            if support == "catalogued" and not catalog[model_type][3]:
                raise ValueError(
                    f"{family} owns {model_type!r}, which has no task-neutral AutoModel"
                )
            if model_type in owners:
                raise ValueError(
                    f"model type {model_type!r} is owned by both "
                    f"{owners[model_type]!r} and {family!r}"
                )
            owners[model_type] = family

        modalities = _nonempty_strings(
            raw.get("modalities"), field="modalities", family=family
        )
        unknown_modalities = set(modalities).difference(MODALITIES)
        if unknown_modalities:
            raise ValueError(
                f"{family} has unknown modalities {sorted(unknown_modalities)}"
            )
        components = _nonempty_strings(
            raw.get("components"), field="components", family=family
        )
        constraints = _strings(
            raw.get("configuration_constraints", ()),
            field="configuration_constraints",
            family=family,
        )
        outputs = _nonempty_strings(
            raw.get("output_contracts"),
            field="output_contracts",
            family=family,
        )
        gates = _nonempty_strings(
            raw.get("acceptance_gates"), field="acceptance_gates", family=family
        )
        unknown_gates = set(gates).difference(ACCEPTANCE_GATES)
        if unknown_gates:
            raise ValueError(f"{family} has unknown gates {sorted(unknown_gates)}")

        config_adapter = raw.get("config_adapter")
        checkpoint_adapter = raw.get("checkpoint_adapter")
        implementation_module = raw.get("implementation_module")
        native_symbols = (config_adapter, checkpoint_adapter, implementation_module)
        if support == "catalogued":
            if any(value is not None for value in native_symbols) or gates:
                raise ValueError(
                    f"catalogued family {family!r} cannot claim native symbols or gates"
                )
        elif any(not isinstance(value, str) or not value for value in native_symbols):
            raise ValueError(f"native family {family!r} requires all native symbols")
        if support == "verified":
            missing = VERIFIED_GATES.difference(gates)
            if missing:
                raise ValueError(
                    f"verified family {family!r} is missing gates {sorted(missing)}"
                )

        contracts = raw.get("input_contracts")
        if not isinstance(contracts, (tuple, list)) or not contracts:
            raise ValueError(f"{family}.input_contracts must be a non-empty sequence")
        normalized_contracts = []
        contract_names: set[str] = set()
        for contract in contracts:
            if not isinstance(contract, (tuple, list)) or len(contract) != 2:
                raise ValueError(f"{family} has an invalid input contract")
            name, fields = contract
            if not isinstance(name, str) or not name or name in contract_names:
                raise ValueError(f"{family} has an invalid input-contract name")
            contract_names.add(name)
            normalized_contracts.append(
                (
                    name,
                    _nonempty_strings(
                        fields,
                        field=f"input_contracts.{name}",
                        family=family,
                    ),
                )
            )

        checkpoint_layout = raw.get("checkpoint_layout")
        if checkpoint_layout != "huggingface_safetensors":
            raise ValueError(
                f"{family} has unsupported checkpoint layout {checkpoint_layout!r}"
            )
        normalized.append(
            {
                "name": family,
                "model_types": model_types,
                "modalities": modalities,
                "components": components,
                "configuration_constraints": constraints,
                "config_adapter": config_adapter,
                "input_contracts": tuple(normalized_contracts),
                "output_contracts": outputs,
                "checkpoint_layout": checkpoint_layout,
                "checkpoint_adapter": checkpoint_adapter,
                "implementation_module": implementation_module,
                "acceptance_gates": gates,
                "support": support,
            }
        )
    return tuple(sorted(normalized, key=lambda item: item["name"]))


def _render(families: tuple[dict[str, Any], ...], catalog_sha256: str) -> str:
    payload = json.dumps(families, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    chunks = "\n".join(
        f"    {payload[offset : offset + 72]!r}"
        for offset in range(0, len(payload), 72)
    )
    return f'''"""Generated semantic model-family registry.

Generated by ``scripts/generate_model_families.py`` from the reviewed manifest.
This file contains identifiers and contracts, not model implementation source.
Do not edit it by hand.
"""

import json

ARCHITECTURE_CATALOG_SHA256 = (
    "{catalog_sha256}"
)
FAMILY_MANIFEST_SHA256 = (
    "{digest}"
)

# The fixed-width JSON chunks are generated data, not Python string style.
# fmt: off
MODEL_FAMILY_ROWS = json.loads(
{chunks}
)
# fmt: on
'''


def generate() -> str:
    catalog = _load(CATALOG, "_representax_transformers_catalog")
    manifest = _load(MANIFEST, "_representax_model_family_manifest")
    families = validate_manifest(
        manifest.MODEL_FAMILIES,
        catalog_rows=catalog.ARCHITECTURE_ROWS,
        catalog_sha256=catalog.CATALOG_SHA256,
        reference_catalog_sha256=manifest.REFERENCE_CATALOG_SHA256,
    )
    return _render(families, catalog.CATALOG_SHA256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = generate()
    if arguments.check:
        if not TARGET.is_file() or TARGET.read_text() != rendered:
            raise SystemExit(f"family registry is stale: run {Path(__file__).name}")
        return
    TARGET.write_text(rendered)


if __name__ == "__main__":
    main()
