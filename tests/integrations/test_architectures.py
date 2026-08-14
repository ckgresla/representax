"""Pinned Hugging Face architecture-catalog acceptance."""

from __future__ import annotations

import pytest

from representax.integrations import (
    HUGGING_FACE_ARCHITECTURES,
    TRANSFORMERS_VERSION,
    ArchitectureSupport,
    get_hugging_face_architecture,
)


def test_transformers_5_3_catalog_is_complete_and_support_is_explicit():
    assert TRANSFORMERS_VERSION == "5.3.0"
    assert len(HUGGING_FACE_ARCHITECTURES) == 475

    modernvbert = get_hugging_face_architecture("modernvbert")
    assert modernvbert.config_class == "ModernVBertConfig"
    assert modernvbert.auto_model_classes == ("ModernVBertModel",)
    assert modernvbert.support is ArchitectureSupport.VERIFIED

    bert = get_hugging_face_architecture("bert")
    assert bert.support is ArchitectureSupport.CATALOGUED
    assert bert.has_auto_model


def test_unknown_architecture_names_the_pinned_reference():
    with pytest.raises(KeyError, match="Transformers 5.3.0"):
        get_hugging_face_architecture("not-a-real-model")


@pytest.mark.parity
def test_generated_catalog_matches_pinned_transformers():
    transformers = pytest.importorskip("transformers")
    from transformers.models.auto.configuration_auto import (
        CONFIG_MAPPING_NAMES,
        model_type_to_module_name,
    )
    from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES

    assert transformers.__version__ == TRANSFORMERS_VERSION
    assert set(HUGGING_FACE_ARCHITECTURES) == set(CONFIG_MAPPING_NAMES)
    for model_type, config_class in CONFIG_MAPPING_NAMES.items():
        model_classes = MODEL_MAPPING_NAMES.get(model_type, ())
        if isinstance(model_classes, str):
            model_classes = (model_classes,)
        architecture = HUGGING_FACE_ARCHITECTURES[model_type]
        assert architecture.module == model_type_to_module_name(model_type)
        assert architecture.config_class == config_class
        assert architecture.auto_model_classes == tuple(model_classes)
