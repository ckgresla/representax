"""Reviewed semantic ownership for native Hugging Face model families.

This file is the source of truth for model-family code generation.  It is
ordinary Python data so changes remain easy to review, diff, and compose.  The
generator validates every upstream name against the pinned Transformers
catalogue; it never imports, copies, or transpiles an upstream model forward.
"""

from __future__ import annotations

REFERENCE_CATALOG_SHA256 = (
    "2af65737162af25c40803b987355fb09d4547d922842edeeb6c0f80838f4117c"
)

MODEL_FAMILIES = (
    {
        "name": "bert",
        "model_types": ("bert",),
        "modalities": ("text",),
        "components": (
            "token_embedding",
            "absolute_position_embedding",
            "token_type_embedding",
            "multi_head_attention",
            "post_norm_residual",
            "dense_mlp",
            "pooler",
            "mean_pooling",
            "l2_normalization",
        ),
        "configuration_constraints": (
            "is_decoder=false",
            "add_cross_attention=false",
        ),
        "config_adapter": "representax.models.bert.BertConfig.from_hf_config",
        "input_contracts": (
            ("text", ("input_ids", "attention_mask")),
            ("embedded_text", ("inputs_embeds", "attention_mask")),
        ),
        "output_contracts": (
            "last_hidden_state",
            "pooler_output",
            "mean_pooled_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": "representax.models.bert.BertCheckpointAdapter",
        "implementation_module": "representax.models.bert",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
            "performance",
        ),
        "support": "native",
    },
    {
        "name": "jina_v5",
        "model_types": ("qwen3_vl_text",),
        "modalities": ("text",),
        "components": (
            "token_embedding",
            "causal_grouped_query_attention",
            "rotary_position_embedding",
            "query_key_rms_normalization",
            "swiglu_mlp",
            "last_token_pooling",
            "matryoshka_truncation",
            "l2_normalization",
        ),
        "configuration_constraints": (
            "checkpoint=jina-embeddings-v5-omni-small-retrieval",
            "revision=12949877f0092093f366c6450340011320152a05",
            "text_path_only=true",
        ),
        "config_adapter": (
            "representax.models.jina_v5.JinaV5TextConfig.from_hf_config"
        ),
        "input_contracts": (("text", ("input_ids", "attention_mask")),),
        "output_contracts": (
            "last_hidden_state",
            "last_token_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": (
            "representax.models.jina_v5.JinaV5TextCheckpointAdapter"
        ),
        "implementation_module": "representax.models.jina_v5",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
        ),
        "support": "native",
    },
    {
        "name": "modernvbert",
        "model_types": ("modernvbert",),
        "modalities": ("text", "image", "fused"),
        "components": (
            "token_embedding",
            "rotary_attention",
            "sliding_attention",
            "gated_mlp",
            "siglip_vision",
            "pixel_shuffle_connector",
            "mean_pooling",
            "l2_normalization",
        ),
        "configuration_constraints": (),
        "config_adapter": (
            "representax.models.modernvbert.ModernVBERTConfig.from_hf_config"
        ),
        "input_contracts": (
            ("text", ("input_ids", "attention_mask")),
            (
                "fused",
                (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_valid",
                ),
            ),
        ),
        "output_contracts": (
            "last_hidden_state",
            "mean_pooled_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": (
            "representax.models.modernvbert.ModernVBERTCheckpointAdapter"
        ),
        "implementation_module": "representax.models.modernvbert",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "performance",
        ),
        "support": "verified",
    },
    {
        "name": "mpnet",
        "model_types": ("mpnet",),
        "modalities": ("text",),
        "components": (
            "token_embedding",
            "padding_aware_absolute_position_embedding",
            "bucketed_relative_attention_bias",
            "multi_head_attention",
            "post_norm_residual",
            "dense_mlp",
            "pooler",
            "mean_pooling",
            "l2_normalization",
        ),
        "configuration_constraints": (
            "pad_token_id=1",
            "relative_attention_num_buckets=32",
        ),
        "config_adapter": "representax.models.mpnet.MPNetConfig.from_hf_config",
        "input_contracts": (
            ("text", ("input_ids", "attention_mask")),
            ("embedded_text", ("inputs_embeds", "attention_mask")),
        ),
        "output_contracts": (
            "last_hidden_state",
            "pooler_output",
            "mean_pooled_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": "representax.models.mpnet.MPNetCheckpointAdapter",
        "implementation_module": "representax.models.mpnet",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
            "performance",
        ),
        "support": "verified",
    },
)
