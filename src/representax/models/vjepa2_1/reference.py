"""Conversion from the official Meta V-JEPA 2.1 reference state layout."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .model import (
    VJEPA2_1Encoder,
    VJEPA2_1Layer,
    VJEPA2_1LayerStack,
    VJEPA2_1Model,
    VJEPA2_1Predictor,
)


def _normalize_state(state: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for name, value in state.items():
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = jnp.asarray(value)
    return normalized


def _load_layer(
    layer: VJEPA2_1Layer,
    state: Mapping[str, Any],
    prefix: str,
) -> VJEPA2_1Layer:
    return eqx.tree_at(
        lambda value: (
            value.attention_norm.weight,
            value.attention_norm.bias,
            value.attention.qkv.weight,
            value.attention.qkv.bias,
            value.attention.output.weight,
            value.attention.output.bias,
            value.mlp_norm.weight,
            value.mlp_norm.bias,
            value.mlp.up.weight,
            value.mlp.up.bias,
            value.mlp.down.weight,
            value.mlp.down.bias,
        ),
        layer,
        (
            state[f"{prefix}.norm1.weight"],
            state[f"{prefix}.norm1.bias"],
            state[f"{prefix}.attn.qkv.weight"],
            state[f"{prefix}.attn.qkv.bias"],
            state[f"{prefix}.attn.proj.weight"],
            state[f"{prefix}.attn.proj.bias"],
            state[f"{prefix}.norm2.weight"],
            state[f"{prefix}.norm2.bias"],
            state[f"{prefix}.mlp.fc1.weight"],
            state[f"{prefix}.mlp.fc1.bias"],
            state[f"{prefix}.mlp.fc2.weight"],
            state[f"{prefix}.mlp.fc2.bias"],
        ),
    )


def load_reference_encoder(
    encoder: VJEPA2_1Encoder,
    state: Mapping[str, Any],
) -> VJEPA2_1Encoder:
    """Load one official online or target encoder state mapping."""

    state = _normalize_state(state)
    image_weight = state["patch_embed_img.proj.weight"]
    if image_weight.shape[2] != 1:
        raise ValueError("reference image tokenizer must use one-frame tubelets")
    layers = tuple(
        _load_layer(encoder.layers.layer(index), state, f"blocks.{index}")
        for index in range(encoder.layers.depth)
    )
    norms = eqx.tree_at(
        lambda value: (value.weight, value.bias),
        encoder.supervision_norms,
        (
            jnp.stack(
                tuple(
                    state[f"norms_block.{index}.weight"]
                    for index in range(len(encoder.config.supervision_layers))
                )
            ),
            jnp.stack(
                tuple(
                    state[f"norms_block.{index}.bias"]
                    for index in range(len(encoder.config.supervision_layers))
                )
            ),
        ),
    )
    return eqx.tree_at(
        lambda value: (
            value.image_patch_weight,
            value.image_patch_bias,
            value.video_patch_weight,
            value.video_patch_bias,
            value.image_modality_embedding,
            value.video_modality_embedding,
            value.layers,
            value.supervision_norms,
        ),
        encoder,
        (
            image_weight[:, :, 0],
            state["patch_embed_img.proj.bias"],
            state["patch_embed.proj.weight"],
            state["patch_embed.proj.bias"],
            state["img_mod_embed"].reshape((-1,)),
            state["video_mod_embed"].reshape((-1,)),
            VJEPA2_1LayerStack.from_layers(layers),
            norms,
        ),
    )


def load_reference_predictor(
    predictor: VJEPA2_1Predictor,
    state: Mapping[str, Any],
) -> VJEPA2_1Predictor:
    """Load the official dense/deep predictor state mapping."""

    state = _normalize_state(state)
    layers = tuple(
        _load_layer(
            predictor.layers.layer(index),
            state,
            f"predictor_blocks.{index}",
        )
        for index in range(predictor.layers.depth)
    )
    return eqx.tree_at(
        lambda value: (
            value.fuse_in.weight,
            value.fuse_in.bias,
            value.fuse_out.weight,
            value.fuse_out.bias,
            value.mask_tokens,
            value.image_modality_embedding,
            value.video_modality_embedding,
            value.layers,
            value.final_norm.weight,
            value.final_norm.bias,
            value.target_projection.weight,
            value.target_projection.bias,
            value.context_projection.weight,
            value.context_projection.bias,
        ),
        predictor,
        (
            state["predictor_embed.0.weight"],
            state["predictor_embed.0.bias"],
            state["predictor_embed.2.weight"],
            state["predictor_embed.2.bias"],
            jnp.stack(
                tuple(
                    state[f"mask_tokens.{index}"].reshape((-1,))
                    for index in range(predictor.mask_tokens.shape[0])
                )
            ),
            state["img_mod_embed"].reshape((-1,)),
            state["video_mod_embed"].reshape((-1,)),
            VJEPA2_1LayerStack.from_layers(layers),
            state["predictor_norm.weight"],
            state["predictor_norm.bias"],
            state["predictor_proj.weight"],
            state["predictor_proj.bias"],
            state["predictor_proj_context.weight"],
            state["predictor_proj_context.bias"],
        ),
    )


def load_reference_state(
    model: VJEPA2_1Model,
    *,
    encoder: Mapping[str, Any],
    predictor: Mapping[str, Any],
    target_encoder: Mapping[str, Any] | None = None,
) -> VJEPA2_1Model:
    """Convert the three official state dictionaries into native Equinox."""

    online = load_reference_encoder(model.online, encoder)
    target = load_reference_encoder(
        model.target,
        encoder if target_encoder is None else target_encoder,
    )
    dense_predictor = load_reference_predictor(model.predictor, predictor)
    return eqx.tree_at(
        lambda value: (value.online, value.predictor, value.target),
        model,
        (online, dense_predictor, target),
    )


def read_reference_checkpoint(path: str | Path) -> Mapping[str, Any]:
    """Read an official PyTorch checkpoint without making Torch a dependency."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - parity environment only
        raise ImportError(
            "reading official V-JEPA checkpoints requires torch"
        ) from error
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_reference_checkpoint(
    model: VJEPA2_1Model,
    path: str | Path,
) -> VJEPA2_1Model:
    """Load a complete official training checkpoint into one native model."""

    checkpoint = read_reference_checkpoint(path)
    missing = {"encoder", "predictor"} - checkpoint.keys()
    if missing:
        raise ValueError(
            "official V-JEPA checkpoint is missing " + ", ".join(sorted(missing))
        )
    target = checkpoint.get("target_encoder")
    return load_reference_state(
        model,
        encoder=checkpoint["encoder"],
        predictor=checkpoint["predictor"],
        target_encoder=target,
    )


__all__ = [
    "load_reference_encoder",
    "load_reference_checkpoint",
    "load_reference_predictor",
    "load_reference_state",
    "read_reference_checkpoint",
]
