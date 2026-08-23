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
        "name": "clip",
        "model_types": ("clip",),
        "modalities": ("text", "image"),
        "components": (
            "causal_text_transformer",
            "vision_patch_transformer",
            "text_projection",
            "vision_projection",
            "additive_late_fusion",
            "optional_l2_normalization",
        ),
        "configuration_constraints": (
            "BGE-VL-base@cc4c733ed997dbee4ac70ccffb911e70c9c24b93",
            "clip-ViT-B-32@327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
        ),
        "config_adapter": "representax.models.clip.CLIPConfig.from_hf_config",
        "input_contracts": (
            ("text", ("input_ids", "attention_mask")),
            ("image", ("pixel_values",)),
            (
                "text_image",
                ("input_ids", "attention_mask", "pixel_values"),
            ),
        ),
        "output_contracts": (
            "projected_representation",
            "optionally_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": "representax.models.clip.CLIPCheckpointAdapter",
        "implementation_module": "representax.models.clip",
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
        "modalities": ("text", "image"),
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
                "text_image",
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
        "name": "llava_next",
        "model_types": ("llava_next",),
        "modalities": ("text", "image"),
        "components": (
            "clip_vision_transformer",
            "any_resolution_image_packing",
            "multimodal_projector",
            "llama_or_mistral_rotary_decoder",
            "last_token_pooling",
            "l2_normalization",
        ),
        "configuration_constraints": (
            "BAAI/BGE-VL-MLLM-S1@455ac20c111813fbb263dd0f22d47d173a971582",
            "BAAI/BGE-VL-MLLM-S2@20137e245f277e7eca277bbb436ce7d632a16406",
            "BAAI/BGE-VL-v1.5-zs@a7ca46102a1a8be517e85cc1f03d1df39498e56c",
            "BAAI/BGE-VL-v1.5-mmeb@59f60b95765b32014df235059c4d8c60e8204be5",
            "royokong/e5-v@684c4c91ebabce3806d4fd8ac52c9c543043f962",
        ),
        "config_adapter": (
            "representax.models.llava_next.LlavaNextConfig.from_hf_config"
        ),
        "input_contracts": (
            ("text", ("input_ids", "attention_mask")),
            (
                "text_image",
                (
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "image_pack_indices",
                ),
            ),
        ),
        "output_contracts": (
            "last_hidden_state",
            "last_token_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": (
            "representax.models.llava_next.LlavaNextCheckpointAdapter"
        ),
        "implementation_module": "representax.models.llava_next",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
        ),
        "support": "native",
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
    {
        "name": "qwen2_5_omni",
        "model_types": ("qwen2_5_omni",),
        "modalities": ("text", "image", "audio", "video"),
        "components": (
            "causal_grouped_query_attention",
            "sectioned_multimodal_rotary_position_embedding",
            "windowed_vision_transformer",
            "chunked_audio_transformer",
            "last_token_pooling",
            "l2_normalized_embedding",
        ),
        "configuration_constraints": (
            "LCO-Embedding-Omni-3B-2605@5f6b5329da5141367da30e06a9826d1322d6c9b2",
            "LCO-Embedding-Omni-7B@108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c",
            "e5-omni-3B@c302bf66d9fb80112e867a1caf253c4e2a23b9e2",
            "e5-omni-7B@e7679b8ddcc20bf351811bedbca38e9cfee334d6",
            "nvidia/omni-embed-nemotron-3b@865db1bb57e369a85357cf114cbd6b3c5322d19d",
            "custom_model_type=nvomniembed",
            "nvidia_text_attention=causal|bidirectional",
        ),
        "config_adapter": (
            "representax.models.qwen2_5_omni.Qwen2_5OmniConfig.from_hf_config"
        ),
        "input_contracts": (
            ("text", ("input_ids", "attention_mask", "position_ids")),
            (
                "multimodal",
                (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "vision_layout",
                    "audio_layout",
                ),
            ),
        ),
        "output_contracts": (
            "last_hidden_state",
            "last_token_l2_normalized_representation",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": (
            "representax.models.qwen2_5_omni.Qwen2_5OmniCheckpointAdapter"
        ),
        "implementation_module": "representax.models.qwen2_5_omni",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
        ),
        "support": "native",
    },
    {
        "name": "qwen2_vl",
        "model_types": ("qwen2_vl", "qwen2_5_vl"),
        "modalities": ("text", "image", "video"),
        "components": (
            "causal_grouped_query_attention",
            "sectioned_multimodal_rotary_position_embedding",
            "generation_specific_vision_patch_transformer",
            "finite_static_shape_media_processing",
            "first_last_mean_pooling",
            "l2_normalized_embedding",
            "mlp_relevance_scoring",
            "peft_lora_import",
        ),
        "configuration_constraints": (
            "BAAI/BGE-VL-Screenshot@2b0f1cd3e4acf66be759d840954e0c9f1c9a42cf",
            "nomic-ai/nomic-embed-multimodal-3b@29259db79bc6ee5fcc9e6abc8a8e16d8491e5116",
            "nomic-ai/nomic-embed-multimodal-7b@234bc2738e2d5ae77beca8f94e1577a7a48fc609",
            "jinaai/jina-reranker-m0@94bfe0aeb2d4dd7978362699cddd5893d4e0adc8",
        ),
        "config_adapter": "representax.models.qwen2_vl.Qwen2VLConfig.from_hf_config",
        "input_contracts": (
            ("text", ("input_ids", "attention_mask", "position_ids")),
            (
                "text_image_video",
                (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "pixel_values",
                    "vision_layout",
                ),
            ),
        ),
        "output_contracts": (
            "last_hidden_state",
            "l2_normalized_representation",
            "sigmoid_relevance_score",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": "representax.models.qwen2_vl.Qwen2VLCheckpointAdapter",
        "implementation_module": "representax.models.qwen2_vl",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
        ),
        "support": "native",
    },
    {
        "name": "qwen3_vl",
        "model_types": ("qwen3_vl",),
        "modalities": ("text", "image", "video"),
        "components": (
            "causal_grouped_query_attention",
            "interleaved_multimodal_rotary_position_embedding",
            "vision_patch_transformer",
            "bilinear_learned_position_interpolation",
            "deepstack_vision_fusion",
            "last_token_pooling",
            "l2_normalized_embedding",
            "tied_token_binary_reranking",
        ),
        "configuration_constraints": (
            "checkpoint=Qwen3-VL-Embedding-2B|Qwen3-VL-Reranker-2B|eager-embed-v1",
            "eager-embed-v1@51dfdee0d1d1067afe00d816dca2cd72a02f6bec",
            "revisions_are_pinned=true",
        ),
        "config_adapter": "representax.models.qwen3_vl.Qwen3VLConfig.from_hf_config",
        "input_contracts": (
            ("text", ("input_ids", "attention_mask", "position_ids")),
            (
                "text_image_video",
                (
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "pixel_values",
                    "vision_layout",
                ),
            ),
        ),
        "output_contracts": (
            "last_hidden_state",
            "last_token_l2_normalized_representation",
            "binary_relevance_score",
        ),
        "checkpoint_layout": "huggingface_safetensors",
        "checkpoint_adapter": ("representax.models.qwen3_vl.Qwen3VLCheckpointAdapter"),
        "implementation_module": "representax.models.qwen3_vl",
        "acceptance_gates": (
            "config_mapping",
            "checkpoint_roundtrip",
            "forward",
            "input_gradient",
            "parameter_gradient",
            "optimizer_update",
            "export_reload",
        ),
        "support": "native",
    },
)
