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
    ResolvedHuggingFaceCheckpoint,
    load_hf_config,
    load_safetensor_subset,
    resolve_hf_checkpoint,
)
from .jina import load_jina_v5_small_text_encoder
from .sentence_transformers import (
    SENTENCE_TRANSFORMERS_ORACLE_VERSION,
    LoadedSentenceTransformer,
    SentencePairCollator,
    SentenceTransformerModuleSpec,
    load_sentence_transformer,
    load_sentence_transformer_artifact,
    load_sentence_transformer_encoder,
    load_sentence_transformer_modules,
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
    "LoadedSentenceTransformer",
    "MODEL_FAMILIES",
    "MODEL_TYPE_TO_FAMILY",
    "ModelFamily",
    "ModelInputContract",
    "ResolvedHuggingFaceCheckpoint",
    "SENTENCE_TRANSFORMERS_ORACLE_VERSION",
    "SentencePairCollator",
    "SentenceTransformerModuleSpec",
    "TRANSFORMERS_VERSION",
    "get_hugging_face_architecture",
    "get_model_family",
    "get_model_type_family",
    "load_hf_config",
    "load_jina_v5_small_text_encoder",
    "load_safetensor_subset",
    "load_sentence_transformer",
    "load_sentence_transformer_artifact",
    "load_sentence_transformer_encoder",
    "load_sentence_transformer_modules",
    "resolve_hf_checkpoint",
]
