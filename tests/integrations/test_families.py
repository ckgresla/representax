"""Semantic model-family manifest and code-generation contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from scripts.generate_model_families import TARGET, generate, validate_manifest
from scripts.model_family_manifest import (
    MODEL_FAMILIES as REVIEWED_MODEL_FAMILIES,
)
from scripts.model_family_manifest import (
    REFERENCE_CATALOG_SHA256,
)

from representax.core import Modality
from representax.integrations import (
    CATALOG_SHA256,
    FAMILY_MANIFEST_SHA256,
    MODEL_FAMILIES,
    MODEL_TYPE_TO_FAMILY,
    AcceptanceGate,
    CheckpointLayout,
    FamilySupport,
    get_model_family,
    get_model_type_family,
)
from representax.integrations._transformers_5_3_0 import ARCHITECTURE_ROWS


def test_modernvbert_family_is_semantic_and_verified():
    family = get_model_family("modernvbert")
    assert family is get_model_type_family("modernvbert")
    assert MODEL_TYPE_TO_FAMILY["modernvbert"] is family
    assert family.support is FamilySupport.VERIFIED
    assert family.modalities == (Modality.TEXT, Modality.IMAGE)
    assert "rotary_attention" in family.components
    assert "siglip_vision" in family.components
    assert family.checkpoint_layout is CheckpointLayout.HUGGING_FACE_SAFETENSORS
    assert AcceptanceGate.FORWARD in family.acceptance_gates
    assert AcceptanceGate.INPUT_GRADIENT in family.acceptance_gates
    assert AcceptanceGate.PERFORMANCE in family.acceptance_gates
    assert get_model_type_family("albert") is None


def test_bert_family_is_native_with_explicit_configuration_constraints():
    family = get_model_family("bert")
    assert family is get_model_type_family("bert")
    assert family.support is FamilySupport.NATIVE
    assert family.configuration_constraints == (
        "is_decoder=false",
        "add_cross_attention=false",
    )
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
        AcceptanceGate.PERFORMANCE,
    }


def test_mpnet_family_is_verified_with_explicit_position_semantics():
    family = get_model_family("mpnet")
    assert family is get_model_type_family("mpnet")
    assert family.support is FamilySupport.VERIFIED
    assert family.configuration_constraints == (
        "pad_token_id=1",
        "relative_attention_num_buckets=32",
    )
    assert "padding_aware_absolute_position_embedding" in family.components
    assert "bucketed_relative_attention_bias" in family.components
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
        AcceptanceGate.PERFORMANCE,
    }


def test_jina_v5_family_is_checkpoint_specific_and_native():
    family = get_model_family("jina_v5")
    assert family is get_model_type_family("qwen3_vl_text")
    assert family.support is FamilySupport.NATIVE
    assert family.configuration_constraints == (
        "checkpoint=jina-embeddings-v5-omni-small-retrieval",
        "revision=12949877f0092093f366c6450340011320152a05",
        "text_path_only=true",
    )
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
    }


def test_qwen3_vl_family_has_multimodal_embedding_and_reranking_gates():
    family = get_model_family("qwen3_vl")
    assert family is get_model_type_family("qwen3_vl")
    assert family.support is FamilySupport.NATIVE
    assert family.modalities == (Modality.TEXT, Modality.IMAGE, Modality.VIDEO)
    assert "deepstack_vision_fusion" in family.components
    assert "tied_token_binary_reranking" in family.components
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
    }


def test_qwen2_5_omni_family_has_four_modality_training_and_export_gates():
    family = get_model_family("qwen2_5_omni")
    assert family is get_model_type_family("qwen2_5_omni")
    assert family.support is FamilySupport.NATIVE
    assert family.modalities == (
        Modality.TEXT,
        Modality.IMAGE,
        Modality.AUDIO,
        Modality.VIDEO,
    )
    assert "chunked_audio_transformer" in family.components
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
    }


def test_qwen2_vl_family_owns_both_generations_and_dense_reranking_contracts():
    family = get_model_family("qwen2_vl")
    assert family is get_model_type_family("qwen2_vl")
    assert family is get_model_type_family("qwen2_5_vl")
    assert family.modalities == (Modality.TEXT, Modality.IMAGE, Modality.VIDEO)
    assert "peft_lora_import" in family.components
    assert "mlp_relevance_scoring" in family.components
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
    }


def test_clip_family_has_text_image_training_and_export_gates():
    family = get_model_family("clip")
    assert family is get_model_type_family("clip")
    assert family.support is FamilySupport.VERIFIED
    assert family.modalities == (Modality.TEXT, Modality.IMAGE)
    assert "additive_late_fusion" in family.components
    assert family.acceptance_gates == {
        AcceptanceGate.CONFIG_MAPPING,
        AcceptanceGate.CHECKPOINT_ROUNDTRIP,
        AcceptanceGate.FORWARD,
        AcceptanceGate.INPUT_GRADIENT,
        AcceptanceGate.PARAMETER_GRADIENT,
        AcceptanceGate.OPTIMIZER_UPDATE,
        AcceptanceGate.EXPORT_RELOAD,
        AcceptanceGate.PERFORMANCE,
    }


def test_generated_family_registry_is_current_and_torch_free():
    assert REFERENCE_CATALOG_SHA256 == CATALOG_SHA256
    assert FAMILY_MANIFEST_SHA256
    assert generate() == TARGET.read_text()


def _validate(families, *, reference=REFERENCE_CATALOG_SHA256):
    return validate_manifest(
        families,
        catalog_rows=ARCHITECTURE_ROWS,
        catalog_sha256=CATALOG_SHA256,
        reference_catalog_sha256=reference,
    )


def test_manifest_rejects_stale_unknown_and_overlapping_ownership():
    with pytest.raises(ValueError, match="stale architecture catalogue"):
        _validate(REVIEWED_MODEL_FAMILIES, reference="stale")

    unknown = deepcopy(REVIEWED_MODEL_FAMILIES[0])
    unknown["model_types"] = ("not-a-real-model",)
    with pytest.raises(ValueError, match="unknown model type"):
        _validate((unknown,))

    duplicate = deepcopy(REVIEWED_MODEL_FAMILIES[0])
    duplicate["name"] = "duplicate"
    with pytest.raises(ValueError, match="owned by both"):
        _validate((REVIEWED_MODEL_FAMILIES[0], duplicate))


def test_manifest_rejects_unearned_verified_support():
    unverified: dict[str, Any] = deepcopy(
        next(
            family
            for family in REVIEWED_MODEL_FAMILIES
            if family["support"] == "verified"
        )
    )
    unverified["acceptance_gates"] = ("config_mapping", "forward")
    with pytest.raises(ValueError, match="missing gates"):
        _validate((unverified,))


def test_family_names_and_model_types_are_unique():
    assert len(MODEL_FAMILIES) == len(set(MODEL_FAMILIES))
    owned = [
        model_type
        for family in MODEL_FAMILIES.values()
        for model_type in family.model_types
    ]
    assert len(owned) == len(set(owned))
