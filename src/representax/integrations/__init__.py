"""Optional interoperability with external model and data ecosystems."""

from .huggingface import (
    HuggingFaceCheckpointAdapter,
    ResolvedHuggingFaceCheckpoint,
    load_hf_config,
    load_safetensor_subset,
    resolve_hf_checkpoint,
)
from .jina import load_jina_v5_small_text_encoder, load_jina_v5_text_encoder
from .late_interaction import (
    GTE_MODERN_COLBERT_MODEL_ID,
    GTE_MODERN_COLBERT_REVISION,
    LateInteractionCheckpointAdapter,
    load_late_interaction_text_model,
    save_late_interaction_text_model,
)
from .modernvbert import load_modernvbert_encoder, load_modernvbert_text_encoder
from .sentence_transformers import (
    SENTENCE_TRANSFORMERS_ORACLE_VERSION,
    LoadedSentenceTransformer,
    SentenceTransformerGraphKind,
    SentenceTransformerGraphSpec,
    SentenceTransformerInputSpec,
    SentenceTransformerModuleRole,
    SentenceTransformerModuleSpec,
    SentenceTransformerRouteMapping,
    SentenceTransformerRouteSpec,
    load_sentence_transformer,
    load_sentence_transformer_artifact,
    load_sentence_transformer_graph,
    load_sentence_transformer_modules,
    save_sentence_transformer_artifact,
)

__all__ = [
    "HuggingFaceCheckpointAdapter",
    "LoadedSentenceTransformer",
    "GTE_MODERN_COLBERT_MODEL_ID",
    "GTE_MODERN_COLBERT_REVISION",
    "LateInteractionCheckpointAdapter",
    "ResolvedHuggingFaceCheckpoint",
    "SENTENCE_TRANSFORMERS_ORACLE_VERSION",
    "SentenceTransformerGraphKind",
    "SentenceTransformerGraphSpec",
    "SentenceTransformerInputSpec",
    "SentenceTransformerModuleSpec",
    "SentenceTransformerModuleRole",
    "SentenceTransformerRouteMapping",
    "SentenceTransformerRouteSpec",
    "load_hf_config",
    "load_jina_v5_small_text_encoder",
    "load_jina_v5_text_encoder",
    "load_late_interaction_text_model",
    "load_modernvbert_encoder",
    "load_modernvbert_text_encoder",
    "load_safetensor_subset",
    "load_sentence_transformer",
    "load_sentence_transformer_artifact",
    "load_sentence_transformer_graph",
    "load_sentence_transformer_modules",
    "save_sentence_transformer_artifact",
    "save_late_interaction_text_model",
    "resolve_hf_checkpoint",
]
