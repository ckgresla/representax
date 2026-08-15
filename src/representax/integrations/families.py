"""Semantic implementation families for pinned Hugging Face architectures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypedDict, cast

from representax.core import Modality

from ._model_families import (
    ARCHITECTURE_CATALOG_SHA256,
    FAMILY_MANIFEST_SHA256,
    MODEL_FAMILY_ROWS,
)


class FamilySupport(StrEnum):
    """The strongest implementation claim made by one semantic family."""

    CATALOGUED = "catalogued"
    NATIVE = "native"
    VERIFIED = "verified"


class AcceptanceGate(StrEnum):
    """An independently reviewable model-family acceptance result."""

    CONFIG_MAPPING = "config_mapping"
    CHECKPOINT_ROUNDTRIP = "checkpoint_roundtrip"
    FORWARD = "forward"
    INPUT_GRADIENT = "input_gradient"
    PARAMETER_GRADIENT = "parameter_gradient"
    OPTIMIZER_UPDATE = "optimizer_update"
    EXPORT_RELOAD = "export_reload"
    PERFORMANCE = "performance"


class CheckpointLayout(StrEnum):
    """Checkpoint transport understood by the shared integration layer."""

    HUGGING_FACE_SAFETENSORS = "huggingface_safetensors"


@dataclass(frozen=True, slots=True)
class ModelInputContract:
    """One named set of tensors accepted by a model family."""

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelFamily:
    """One reusable native implementation and its upstream ownership."""

    name: str
    model_types: tuple[str, ...]
    modalities: tuple[Modality, ...]
    components: tuple[str, ...]
    configuration_constraints: tuple[str, ...]
    config_adapter: str | None
    input_contracts: tuple[ModelInputContract, ...]
    output_contracts: tuple[str, ...]
    checkpoint_layout: CheckpointLayout
    checkpoint_adapter: str | None
    implementation_module: str | None
    acceptance_gates: frozenset[AcceptanceGate]
    support: FamilySupport


class _ModelFamilyRow(TypedDict):
    """Typed view of one generator-validated manifest row."""

    name: str
    model_types: list[str]
    modalities: list[str]
    components: list[str]
    configuration_constraints: list[str]
    config_adapter: str | None
    input_contracts: list[tuple[str, list[str]]]
    output_contracts: list[str]
    checkpoint_layout: str
    checkpoint_adapter: str | None
    implementation_module: str | None
    acceptance_gates: list[str]
    support: str


def _family(row: _ModelFamilyRow) -> ModelFamily:
    return ModelFamily(
        name=str(row["name"]),
        model_types=tuple(str(value) for value in row["model_types"]),
        modalities=tuple(Modality(value) for value in row["modalities"]),
        components=tuple(str(value) for value in row["components"]),
        configuration_constraints=tuple(
            str(value) for value in row["configuration_constraints"]
        ),
        config_adapter=(
            None if row["config_adapter"] is None else str(row["config_adapter"])
        ),
        input_contracts=tuple(
            ModelInputContract(
                name=str(name), fields=tuple(str(field) for field in fields)
            )
            for name, fields in row["input_contracts"]
        ),
        output_contracts=tuple(str(value) for value in row["output_contracts"]),
        checkpoint_layout=CheckpointLayout(row["checkpoint_layout"]),
        checkpoint_adapter=(
            None
            if row["checkpoint_adapter"] is None
            else str(row["checkpoint_adapter"])
        ),
        implementation_module=(
            None
            if row["implementation_module"] is None
            else str(row["implementation_module"])
        ),
        acceptance_gates=frozenset(
            AcceptanceGate(value) for value in row["acceptance_gates"]
        ),
        support=FamilySupport(row["support"]),
    )


_TYPED_MODEL_FAMILY_ROWS = cast(list[_ModelFamilyRow], MODEL_FAMILY_ROWS)

MODEL_FAMILIES: Mapping[str, ModelFamily] = MappingProxyType(
    {
        family.name: family
        for row in _TYPED_MODEL_FAMILY_ROWS
        for family in (_family(row),)
    }
)
MODEL_TYPE_TO_FAMILY: Mapping[str, ModelFamily] = MappingProxyType(
    {
        model_type: family
        for family in MODEL_FAMILIES.values()
        for model_type in family.model_types
    }
)


def get_model_family(name: str) -> ModelFamily:
    """Return one semantic family by its Representax family name."""

    try:
        return MODEL_FAMILIES[name]
    except KeyError as error:
        raise KeyError(f"unknown Representax model family {name!r}") from error


def get_model_type_family(model_type: str) -> ModelFamily | None:
    """Return the native family owning an upstream model type, if any."""

    return MODEL_TYPE_TO_FAMILY.get(model_type)


__all__ = [
    "ARCHITECTURE_CATALOG_SHA256",
    "AcceptanceGate",
    "CheckpointLayout",
    "FAMILY_MANIFEST_SHA256",
    "FamilySupport",
    "MODEL_FAMILIES",
    "MODEL_TYPE_TO_FAMILY",
    "ModelFamily",
    "ModelInputContract",
    "get_model_family",
    "get_model_type_family",
]
