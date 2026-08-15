"""Optional interoperability with external model and data ecosystems."""

from .architectures import (
    CATALOG_SHA256,
    HUGGING_FACE_ARCHITECTURES,
    TRANSFORMERS_VERSION,
    ArchitectureSupport,
    HuggingFaceArchitecture,
    get_hugging_face_architecture,
)
from .families import (
    FAMILY_MANIFEST_SHA256,
    MODEL_FAMILIES,
    MODEL_TYPE_TO_FAMILY,
    AcceptanceGate,
    CheckpointLayout,
    FamilySupport,
    ModelFamily,
    ModelInputContract,
    get_model_family,
    get_model_type_family,
)
from .huggingface import (
    HuggingFaceCheckpointAdapter,
    load_hf_config,
    load_safetensor_subset,
)

__all__ = [
    "ArchitectureSupport",
    "AcceptanceGate",
    "CATALOG_SHA256",
    "CheckpointLayout",
    "FAMILY_MANIFEST_SHA256",
    "FamilySupport",
    "HUGGING_FACE_ARCHITECTURES",
    "HuggingFaceCheckpointAdapter",
    "HuggingFaceArchitecture",
    "MODEL_FAMILIES",
    "MODEL_TYPE_TO_FAMILY",
    "ModelFamily",
    "ModelInputContract",
    "TRANSFORMERS_VERSION",
    "get_hugging_face_architecture",
    "get_model_family",
    "get_model_type_family",
    "load_hf_config",
    "load_safetensor_subset",
]
