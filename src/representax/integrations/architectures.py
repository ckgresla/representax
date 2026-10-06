"""Versioned Hugging Face architecture identities and support state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ._transformers_5_3_0 import (
    ARCHITECTURE_ROWS,
    CATALOG_SHA256,
    TRANSFORMERS_VERSION,
)


class ArchitectureSupport(StrEnum):
    """The strongest Representax claim made for an upstream architecture."""

    CATALOGUED = "catalogued"
    NATIVE = "native"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class HuggingFaceArchitecture:
    """One architecture identity in the pinned Transformers reference."""

    model_type: str
    module: str
    config_class: str
    auto_model_classes: tuple[str, ...]
    support: ArchitectureSupport

    @property
    def has_auto_model(self) -> bool:
        """Whether Transformers registers a task-neutral ``AutoModel``."""

        return bool(self.auto_model_classes)


_NATIVE_SUPPORT = {
    "modernvbert": ArchitectureSupport.VERIFIED,
}

HUGGING_FACE_ARCHITECTURES: Mapping[str, HuggingFaceArchitecture] = MappingProxyType(
    {
        model_type: HuggingFaceArchitecture(
            model_type=model_type,
            module=module,
            config_class=config_class,
            auto_model_classes=auto_model_classes,
            support=_NATIVE_SUPPORT.get(model_type, ArchitectureSupport.CATALOGUED),
        )
        for model_type, module, config_class, auto_model_classes in ARCHITECTURE_ROWS
    }
)


def get_hugging_face_architecture(model_type: str) -> HuggingFaceArchitecture:
    """Return one pinned architecture definition by ``config.model_type``."""

    try:
        return HUGGING_FACE_ARCHITECTURES[model_type]
    except KeyError as error:
        raise KeyError(
            f"{model_type!r} is not registered by Transformers {TRANSFORMERS_VERSION}"
        ) from error


__all__ = [
    "ArchitectureSupport",
    "CATALOG_SHA256",
    "HUGGING_FACE_ARCHITECTURES",
    "HuggingFaceArchitecture",
    "TRANSFORMERS_VERSION",
    "get_hugging_face_architecture",
]
