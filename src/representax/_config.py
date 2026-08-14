"""Shared Pydantic boundary for declarative Representax configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import Annotated, Any, TypeVar, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeAliasType

NonEmptyString = Annotated[str, Field(min_length=1)]
FinitePositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ParameterRole(StrEnum):
    """Intrinsic effect of changing a configuration value."""

    SCIENTIFIC = "scientific"
    EXECUTION = "execution"


@dataclass(frozen=True, slots=True)
class ParameterRoleMetadata:
    """Typing marker attached to one field or complete config subtree."""

    role: ParameterRole


_T = TypeVar("_T")
Scientific = TypeAliasType(
    "Scientific",
    Annotated[_T, ParameterRoleMetadata(ParameterRole.SCIENTIFIC)],
    type_params=(_T,),
)
Execution = TypeAliasType(
    "Execution",
    Annotated[_T, ParameterRoleMetadata(ParameterRole.EXECUTION)],
    type_params=(_T,),
)


class FrozenConfig(BaseModel):
    """Immutable, closed configuration accepted from Python or Hydra-Zen."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
    )


@cache
def _field_roles(model: type[BaseModel]) -> dict[str, ParameterRole]:
    roles = {}
    for name, annotation in get_type_hints(model, include_extras=True).items():
        origin = get_origin(annotation)
        if origin is Scientific:
            roles[name] = ParameterRole.SCIENTIFIC
        elif origin is Execution:
            roles[name] = ParameterRole.EXECUTION
    return roles


def _field_role(model: type[BaseModel], name: str) -> ParameterRole | None:
    markers = [
        item
        for item in model.model_fields[name].metadata
        if isinstance(item, ParameterRoleMetadata)
    ]
    if len(markers) > 1:
        raise TypeError(f"configuration field {model.__name__}.{name} has many roles")
    metadata_role = None if not markers else markers[0].role
    alias_role = _field_roles(model).get(name)
    if metadata_role is not None and alias_role is not None:
        raise TypeError(f"configuration field {model.__name__}.{name} has many roles")
    return metadata_role or alias_role


def _project_model(
    model: BaseModel,
    dumped: dict[str, Any],
    role: ParameterRole,
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for name in type(model).model_fields:
        field_role = _field_role(type(model), name)
        value = getattr(model, name)
        serialized = dumped[name]
        if field_role is role:
            projected[name] = serialized
        elif field_role is None and isinstance(value, BaseModel):
            nested = _project_model(value, serialized, role)
            if nested:
                projected[name] = nested
    return projected


def project_parameters(
    config: FrozenConfig,
    role: ParameterRole,
) -> dict[str, Any]:
    """Project one config into fields carrying an intrinsic parameter role."""

    dumped = config.model_dump(mode="json")
    return _project_model(config, dumped, role)


__all__ = [
    "Execution",
    "FinitePositiveFloat",
    "FrozenConfig",
    "NonEmptyString",
    "ParameterRole",
    "Scientific",
    "project_parameters",
]
