"""Bidirectional Hugging Face weight mapping for native ModernVBERT paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from representax.core import EncoderMetadata, Modality, Route
from representax.integrations.huggingface import (
    load_hf_config,
    load_safetensor_subset,
)

from .config import (
    MODERNVBERT_MODEL_ID,
    MODERNVBERT_REVISION,
    ModernVBERTConfig,
    ModernVBERTTextConfig,
    ModernVBERTVisionConfig,
)
from .model import (
    AttentionImplementation,
    FusedSelfAttention,
    GatedMLP,
    LayerNorm,
    Linear,
    ModernVBERTTextBlock,
    ModernVBERTTextEncoder,
    ModernVBERTTextTower,
)
from .multimodal import ModernVBERTEncoder
from .vision import (
    PatchEmbedding,
    SigLIPVisionAttention,
    SigLIPVisionLayer,
    SigLIPVisionMLP,
    SigLIPVisionTower,
)


def modernvbert_text_weight_map(
    config: ModernVBERTTextConfig,
) -> dict[str, str]:
    """Map stable native parameter paths to Transformers tensor names."""

    prefix = "model.text_model."
    mapping = {
        "tower.token_embedding": prefix + "embeddings.tok_embeddings.weight",
        "tower.embedding_norm.weight": prefix + "embeddings.norm.weight",
        "tower.final_norm.weight": prefix + "final_norm.weight",
    }
    for index in range(config.num_hidden_layers):
        native = f"tower.layers.{index}."
        upstream = f"{prefix}layers.{index}."
        mapping.update(
            {
                native + "attention.qkv.weight": upstream + "attn.Wqkv.weight",
                native + "attention.output.weight": upstream + "attn.Wo.weight",
                native + "mlp_norm.weight": upstream + "mlp_norm.weight",
                native + "mlp.input.weight": upstream + "mlp.Wi.weight",
                native + "mlp.output.weight": upstream + "mlp.Wo.weight",
            }
        )
        if index > 0:
            mapping[native + "attention_norm.weight"] = upstream + "attn_norm.weight"
    return mapping


def _expected_text_shapes(
    config: ModernVBERTTextConfig,
) -> dict[str, tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    shapes = {
        "tower.token_embedding": (config.vocab_size, hidden),
        "tower.embedding_norm.weight": (hidden,),
        "tower.final_norm.weight": (hidden,),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"tower.layers.{index}."
        shapes.update(
            {
                prefix + "attention.qkv.weight": (3 * hidden, hidden),
                prefix + "attention.output.weight": (hidden, hidden),
                prefix + "mlp_norm.weight": (hidden,),
                prefix + "mlp.input.weight": (2 * intermediate, hidden),
                prefix + "mlp.output.weight": (hidden, intermediate),
            }
        )
        if index > 0:
            shapes[prefix + "attention_norm.weight"] = (hidden,)
    return shapes


def modernvbert_vision_weight_map(
    config: ModernVBERTVisionConfig,
) -> dict[str, str]:
    """Map native vision and connector paths to executed HF tensors."""

    prefix = "model.vision_model.vision_model."
    mapping = {
        "vision.patch_embedding.weight": prefix + "embeddings.patch_embedding.weight",
        "vision.patch_embedding.bias": prefix + "embeddings.patch_embedding.bias",
        "vision.position_embedding": prefix + "embeddings.position_embedding.weight",
        "vision.final_norm.weight": prefix + "post_layernorm.weight",
        "vision.final_norm.bias": prefix + "post_layernorm.bias",
        "connector.weight": "model.connector.modality_projection.weight",
    }
    for index in range(config.num_hidden_layers):
        native = f"vision.layers.{index}."
        upstream = f"{prefix}encoder.layers.{index}."
        mapping.update(
            {
                native + "attention_norm.weight": upstream + "layer_norm1.weight",
                native + "attention_norm.bias": upstream + "layer_norm1.bias",
                native + "attention.query.weight": upstream + "self_attn.q_proj.weight",
                native + "attention.query.bias": upstream + "self_attn.q_proj.bias",
                native + "attention.key.weight": upstream + "self_attn.k_proj.weight",
                native + "attention.key.bias": upstream + "self_attn.k_proj.bias",
                native + "attention.value.weight": upstream + "self_attn.v_proj.weight",
                native + "attention.value.bias": upstream + "self_attn.v_proj.bias",
                native + "attention.output.weight": upstream
                + "self_attn.out_proj.weight",
                native + "attention.output.bias": upstream + "self_attn.out_proj.bias",
                native + "mlp_norm.weight": upstream + "layer_norm2.weight",
                native + "mlp_norm.bias": upstream + "layer_norm2.bias",
                native + "mlp.input.weight": upstream + "mlp.fc1.weight",
                native + "mlp.input.bias": upstream + "mlp.fc1.bias",
                native + "mlp.output.weight": upstream + "mlp.fc2.weight",
                native + "mlp.output.bias": upstream + "mlp.fc2.bias",
            }
        )
    return mapping


def _expected_vision_shapes(
    config: ModernVBERTConfig,
) -> dict[str, tuple[int, ...]]:
    vision = config.vision
    hidden = vision.hidden_size
    intermediate = vision.intermediate_size
    shapes = {
        "vision.patch_embedding.weight": (
            hidden,
            vision.num_channels,
            vision.patch_size,
            vision.patch_size,
        ),
        "vision.patch_embedding.bias": (hidden,),
        "vision.position_embedding": (vision.patch_count, hidden),
        "vision.final_norm.weight": (hidden,),
        "vision.final_norm.bias": (hidden,),
        "connector.weight": (
            config.text.hidden_size,
            hidden * config.pixel_shuffle_factor**2,
        ),
    }
    for index in range(vision.num_hidden_layers):
        prefix = f"vision.layers.{index}."
        shapes.update(
            {
                prefix + "attention_norm.weight": (hidden,),
                prefix + "attention_norm.bias": (hidden,),
                prefix + "attention.query.weight": (hidden, hidden),
                prefix + "attention.query.bias": (hidden,),
                prefix + "attention.key.weight": (hidden, hidden),
                prefix + "attention.key.bias": (hidden,),
                prefix + "attention.value.weight": (hidden, hidden),
                prefix + "attention.value.bias": (hidden,),
                prefix + "attention.output.weight": (hidden, hidden),
                prefix + "attention.output.bias": (hidden,),
                prefix + "mlp_norm.weight": (hidden,),
                prefix + "mlp_norm.bias": (hidden,),
                prefix + "mlp.input.weight": (intermediate, hidden),
                prefix + "mlp.input.bias": (intermediate,),
                prefix + "mlp.output.weight": (hidden, intermediate),
                prefix + "mlp.output.bias": (hidden,),
            }
        )
    return shapes


def _array(
    values: Mapping[str, Any],
    name: str,
    dtype: jnp.dtype,
    expected_shape: tuple[int, ...],
) -> jax.Array:
    try:
        value = values[name]
    except KeyError as error:
        raise KeyError(f"ModernVBERT checkpoint is missing {name}") from error
    array = jnp.asarray(value, dtype=dtype)
    if array.shape != expected_shape:
        raise ValueError(
            f"ModernVBERT tensor {name} has shape {array.shape}; "
            f"expected {expected_shape}"
        )
    return array


def _require_biasless(model: ModernVBERTTextEncoder) -> None:
    biased = []
    if model.tower.embedding_norm.bias is not None:
        biased.append("tower.embedding_norm.bias")
    if model.tower.final_norm.bias is not None:
        biased.append("tower.final_norm.bias")
    for index, layer in enumerate(model.tower.layers):
        prefix = f"tower.layers.{index}."
        if layer.attention.qkv.bias is not None:
            biased.append(prefix + "attention.qkv.bias")
        if layer.attention.output.bias is not None:
            biased.append(prefix + "attention.output.bias")
        if layer.attention_norm is not None and layer.attention_norm.bias is not None:
            biased.append(prefix + "attention_norm.bias")
        if layer.mlp_norm.bias is not None:
            biased.append(prefix + "mlp_norm.bias")
        if layer.mlp.input.bias is not None:
            biased.append(prefix + "mlp.input.bias")
        if layer.mlp.output.bias is not None:
            biased.append(prefix + "mlp.output.bias")
    if biased:
        raise ValueError(
            "Transformers ModernVBERT text checkpoints are biasless; "
            f"cannot export {sorted(biased)}"
        )


@dataclass(frozen=True)
class ModernVBERTTextCheckpointAdapter:
    """Load and export the shared ModernVBERT text architecture.

    The adapter owns checkpoint identity and serialization facts.  The model
    graph itself contains only native arrays and architecture configuration.
    """

    model_id: str = MODERNVBERT_MODEL_ID
    revision: str = MODERNVBERT_REVISION

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("checkpoint model_id and revision must be non-empty")

    def from_state_dict(
        self,
        config: ModernVBERTTextConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
    ) -> ModernVBERTTextEncoder:
        mapping = modernvbert_text_weight_map(config)
        shapes = _expected_text_shapes(config)

        def get(native_name: str) -> jax.Array:
            return _array(
                state_dict,
                mapping[native_name],
                parameter_dtype,
                shapes[native_name],
            )

        layers = []
        for index, layer_type in enumerate(config.layer_types):
            prefix = f"tower.layers.{index}."
            attention_norm = (
                None
                if index == 0
                else LayerNorm(
                    weight=get(prefix + "attention_norm.weight"),
                    bias=None,
                    epsilon=config.norm_epsilon,
                )
            )
            layers.append(
                ModernVBERTTextBlock(
                    attention=FusedSelfAttention(
                        qkv=Linear(weight=get(prefix + "attention.qkv.weight")),
                        output=Linear(weight=get(prefix + "attention.output.weight")),
                        layer_type=layer_type,
                    ),
                    attention_norm=attention_norm,
                    mlp_norm=LayerNorm(
                        weight=get(prefix + "mlp_norm.weight"),
                        bias=None,
                        epsilon=config.norm_epsilon,
                    ),
                    mlp=GatedMLP(
                        input=Linear(weight=get(prefix + "mlp.input.weight")),
                        output=Linear(weight=get(prefix + "mlp.output.weight")),
                    ),
                )
            )
        tower = ModernVBERTTextTower(
            token_embedding=get("tower.token_embedding"),
            embedding_norm=LayerNorm(
                weight=get("tower.embedding_norm.weight"),
                bias=None,
                epsilon=config.norm_epsilon,
            ),
            layers=tuple(layers),
            final_norm=LayerNorm(
                weight=get("tower.final_norm.weight"),
                bias=None,
                epsilon=config.norm_epsilon,
            ),
            config=config,
        )
        return ModernVBERTTextEncoder(
            tower=tower,
            metadata=EncoderMetadata(
                model_id=self.model_id,
                revision=self.revision,
                output_dimension=config.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT}),
            ),
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
    ) -> ModernVBERTTextEncoder:
        hf_config = load_hf_config(checkpoint)
        config = ModernVBERTTextConfig.from_hf_config(hf_config)
        names = set(modernvbert_text_weight_map(config).values())
        tensors = load_safetensor_subset(checkpoint, names, dtype=parameter_dtype)
        return self.from_state_dict(
            config,
            tensors,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def state_dict(
        self,
        model: ModernVBERTTextEncoder,
    ) -> dict[str, jax.Array]:
        """Return Transformers-compatible text tensor names and values."""

        config = model.tower.config
        mapping = modernvbert_text_weight_map(config)
        shapes = _expected_text_shapes(config)
        _require_biasless(model)
        native: dict[str, jax.Array] = {
            "tower.token_embedding": model.tower.token_embedding,
            "tower.embedding_norm.weight": model.tower.embedding_norm.weight,
            "tower.final_norm.weight": model.tower.final_norm.weight,
        }
        for index, layer in enumerate(model.tower.layers):
            prefix = f"tower.layers.{index}."
            native.update(
                {
                    prefix + "attention.qkv.weight": layer.attention.qkv.weight,
                    prefix + "attention.output.weight": layer.attention.output.weight,
                    prefix + "mlp_norm.weight": layer.mlp_norm.weight,
                    prefix + "mlp.input.weight": layer.mlp.input.weight,
                    prefix + "mlp.output.weight": layer.mlp.output.weight,
                }
            )
            if index > 0:
                if layer.attention_norm is None:
                    raise ValueError(
                        "ModernVBERT layers after layer zero require attention_norm"
                    )
                native[prefix + "attention_norm.weight"] = layer.attention_norm.weight
        if set(native) != set(mapping):
            missing = set(mapping).difference(native)
            extra = set(native).difference(mapping)
            raise ValueError(
                "native ModernVBERT tree does not match its weight map: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, value in native.items():
            if value.shape != shapes[name]:
                raise ValueError(
                    f"native ModernVBERT parameter {name} has shape {value.shape}; "
                    f"expected {shapes[name]}"
                )
        return {mapping[name]: value for name, value in native.items()}


def _vision_native_state(model: ModernVBERTEncoder) -> dict[str, jax.Array]:
    native = {
        "vision.patch_embedding.weight": model.vision.patch_embedding.weight,
        "vision.patch_embedding.bias": model.vision.patch_embedding.bias,
        "vision.position_embedding": model.vision.position_embedding,
        "vision.final_norm.weight": model.vision.final_norm.weight,
        "vision.final_norm.bias": model.vision.final_norm.bias,
        "connector.weight": model.connector.weight,
    }
    for index, layer in enumerate(model.vision.layers):
        prefix = f"vision.layers.{index}."
        native.update(
            {
                prefix + "attention_norm.weight": layer.attention_norm.weight,
                prefix + "attention_norm.bias": layer.attention_norm.bias,
                prefix + "attention.query.weight": layer.attention.query.weight,
                prefix + "attention.query.bias": layer.attention.query.bias,
                prefix + "attention.key.weight": layer.attention.key.weight,
                prefix + "attention.key.bias": layer.attention.key.bias,
                prefix + "attention.value.weight": layer.attention.value.weight,
                prefix + "attention.value.bias": layer.attention.value.bias,
                prefix + "attention.output.weight": layer.attention.output.weight,
                prefix + "attention.output.bias": layer.attention.output.bias,
                prefix + "mlp_norm.weight": layer.mlp_norm.weight,
                prefix + "mlp_norm.bias": layer.mlp_norm.bias,
                prefix + "mlp.input.weight": layer.mlp.input.weight,
                prefix + "mlp.input.bias": layer.mlp.input.bias,
                prefix + "mlp.output.weight": layer.mlp.output.weight,
                prefix + "mlp.output.bias": layer.mlp.output.bias,
            }
        )
    if any(value is None for value in native.values()):
        raise ValueError("ModernVBERT vision parameters require their upstream biases")
    return native  # type: ignore[return-value]


@dataclass(frozen=True)
class ModernVBERTCheckpointAdapter:
    """Load and export ModernVBERT's executed text, vision, and connector path.

    The upstream SigLIP pooling head is intentionally excluded: ModernVBERT
    computes it and then discards it before projecting patch tokens. Omitting
    that dead path makes the native forward both faithful and smaller.
    """

    model_id: str = MODERNVBERT_MODEL_ID
    revision: str = MODERNVBERT_REVISION

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("checkpoint model_id and revision must be non-empty")

    def from_state_dict(
        self,
        config: ModernVBERTConfig,
        state_dict: Mapping[str, Any],
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
    ) -> ModernVBERTEncoder:
        text = ModernVBERTTextCheckpointAdapter(
            model_id=self.model_id,
            revision=self.revision,
        ).from_state_dict(
            config.text,
            state_dict,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )
        mapping = modernvbert_vision_weight_map(config.vision)
        shapes = _expected_vision_shapes(config)

        def get(native_name: str) -> jax.Array:
            return _array(
                state_dict,
                mapping[native_name],
                parameter_dtype,
                shapes[native_name],
            )

        layers = []
        for index in range(config.vision.num_hidden_layers):
            prefix = f"vision.layers.{index}."
            layers.append(
                SigLIPVisionLayer(
                    attention_norm=LayerNorm(
                        weight=get(prefix + "attention_norm.weight"),
                        bias=get(prefix + "attention_norm.bias"),
                        epsilon=config.vision.norm_epsilon,
                    ),
                    attention=SigLIPVisionAttention(
                        query=Linear(
                            weight=get(prefix + "attention.query.weight"),
                            bias=get(prefix + "attention.query.bias"),
                        ),
                        key=Linear(
                            weight=get(prefix + "attention.key.weight"),
                            bias=get(prefix + "attention.key.bias"),
                        ),
                        value=Linear(
                            weight=get(prefix + "attention.value.weight"),
                            bias=get(prefix + "attention.value.bias"),
                        ),
                        output=Linear(
                            weight=get(prefix + "attention.output.weight"),
                            bias=get(prefix + "attention.output.bias"),
                        ),
                    ),
                    mlp_norm=LayerNorm(
                        weight=get(prefix + "mlp_norm.weight"),
                        bias=get(prefix + "mlp_norm.bias"),
                        epsilon=config.vision.norm_epsilon,
                    ),
                    mlp=SigLIPVisionMLP(
                        input=Linear(
                            weight=get(prefix + "mlp.input.weight"),
                            bias=get(prefix + "mlp.input.bias"),
                        ),
                        output=Linear(
                            weight=get(prefix + "mlp.output.weight"),
                            bias=get(prefix + "mlp.output.bias"),
                        ),
                    ),
                )
            )
        vision = SigLIPVisionTower(
            patch_embedding=PatchEmbedding(
                weight=get("vision.patch_embedding.weight"),
                bias=get("vision.patch_embedding.bias"),
                patch_size=config.vision.patch_size,
            ),
            position_embedding=get("vision.position_embedding"),
            layers=tuple(layers),
            final_norm=LayerNorm(
                weight=get("vision.final_norm.weight"),
                bias=get("vision.final_norm.bias"),
                epsilon=config.vision.norm_epsilon,
            ),
            config=config.vision,
        )
        return ModernVBERTEncoder(
            text=text,
            vision=vision,
            connector=Linear(weight=get("connector.weight")),
            metadata=EncoderMetadata(
                model_id=self.model_id,
                revision=self.revision,
                output_dimension=config.text.hidden_size,
                routes=frozenset(Route),
                modalities=frozenset({Modality.TEXT, Modality.IMAGE, Modality.FUSED}),
            ),
            config=config,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def load(
        self,
        checkpoint: str | Path,
        *,
        parameter_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        attention_implementation: AttentionImplementation = "xla",
    ) -> ModernVBERTEncoder:
        hf_config = load_hf_config(checkpoint)
        config = ModernVBERTConfig.from_hf_config(hf_config)
        text_names = modernvbert_text_weight_map(config.text).values()
        vision_names = modernvbert_vision_weight_map(config.vision).values()
        tensors = load_safetensor_subset(
            checkpoint,
            set(text_names) | set(vision_names),
            dtype=parameter_dtype,
        )
        return self.from_state_dict(
            config,
            tensors,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            attention_implementation=attention_implementation,
        )

    def state_dict(self, model: ModernVBERTEncoder) -> dict[str, jax.Array]:
        """Return HF names for all tensors used by the native forward."""

        state = ModernVBERTTextCheckpointAdapter(
            model_id=self.model_id,
            revision=self.revision,
        ).state_dict(model.text)
        mapping = modernvbert_vision_weight_map(model.config.vision)
        shapes = _expected_vision_shapes(model.config)
        native = _vision_native_state(model)
        if set(native) != set(mapping):
            missing = set(mapping).difference(native)
            extra = set(native).difference(mapping)
            raise ValueError(
                "native ModernVBERT vision tree does not match its weight map: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, value in native.items():
            if value.shape != shapes[name]:
                raise ValueError(
                    f"native ModernVBERT parameter {name} has shape {value.shape}; "
                    f"expected {shapes[name]}"
                )
        state.update({mapping[name]: value for name, value in native.items()})
        return state
