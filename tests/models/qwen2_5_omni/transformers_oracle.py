"""Generate a deterministic Transformers 5.6 Qwen2.5-Omni parity oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    from transformers import (
        Qwen2_5OmniAudioEncoderConfig,
        Qwen2_5OmniTextConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniThinkerForConditionalGeneration,
        Qwen2_5OmniVisionEncoderConfig,
    )

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "Qwen2.5-Omni parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(31)
    text_config = cast(Any, Qwen2_5OmniTextConfig)(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 1, 2],
        },
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        layer_types=["full_attention", "full_attention"],
        pad_token_id=0,
    )
    vision_config = cast(Any, Qwen2_5OmniVisionEncoderConfig)(
        depth=2,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        window_size=8,
        fullatt_block_indexes=[1],
    )
    audio_config = cast(Any, Qwen2_5OmniAudioEncoderConfig)(
        d_model=16,
        encoder_ffn_dim=24,
        encoder_layers=2,
        encoder_attention_heads=2,
        num_mel_bins=4,
        max_source_positions=32,
        n_window=4,
        output_dim=16,
    )
    config = cast(Any, Qwen2_5OmniThinkerConfig)(
        text_config=text_config,
        vision_config=vision_config,
        audio_config=audio_config,
        pad_token_id=0,
        image_token_index=1,
        video_token_index=2,
        audio_token_index=3,
        vision_start_token_id=4,
        vision_end_token_id=5,
        audio_start_token_id=6,
        audio_end_token_id=7,
        position_id_per_seconds=25,
        seconds_per_chunk=2,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    config.audio_config._attn_implementation = "eager"
    model = Qwen2_5OmniThinkerForConditionalGeneration(config).eval().float()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor([[8, 4, 1, 1, 1, 1, 5, 6, 3, 3, 7, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixels = torch.arange(16 * 24, dtype=torch.float32).reshape(16, 24) / 100
    image_grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    audio_features = torch.arange(4 * 8, dtype=torch.float32).reshape(1, 4, 8) / 100
    feature_mask = torch.ones((1, 8), dtype=torch.long)
    objective = torch.linspace(-0.5, 0.5, config.text_config.hidden_size)

    position_ids, _ = model.get_rope_index(
        input_ids,
        image_grid,
        None,
        attention_mask,
        False,
        feature_mask.sum(-1),
        None,
    )
    with torch.no_grad():
        image_features = model.get_image_features(
            pixels,
            image_grid_thw=image_grid,
        ).pooler_output
        audio_embeddings = model.get_audio_features(
            audio_features,
            feature_attention_mask=feature_mask,
        ).last_hidden_state

    def forward(pixel_values, input_features):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid,
            input_features=input_features,
            feature_attention_mask=feature_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    with torch.no_grad():
        hidden = forward(pixels, audio_features)
    differentiable_pixels = pixels.clone().requires_grad_(True)
    differentiable_audio = audio_features.clone().requires_grad_(True)
    differentiated_hidden = forward(differentiable_pixels, differentiable_audio)
    input_loss = torch.sum(differentiated_hidden * objective)
    pixel_gradient, audio_gradient = torch.autograd.grad(
        input_loss, (differentiable_pixels, differentiable_audio)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    parameter_gradients = {}
    training_losses = []
    participating_names = set()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        training_hidden = forward(pixels, audio_features)
        parameter_loss = torch.sum(training_hidden * objective)
        training_losses.append(parameter_loss.detach().numpy())
        parameter_loss.backward()
        if step == 0:
            parameter_gradients = {
                "parameter_gradient__" + name: value.grad.detach().numpy()
                for name, value in model.named_parameters()
                if value.grad is not None
            }
            participating_names = {
                name
                for name, value in model.named_parameters()
                if value.grad is not None
            }
        optimizer.step()
    updated_parameters = {
        "updated_parameter__" + name: value.detach().numpy()
        for name, value in model.named_parameters()
        if name in participating_names
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        pixel_values=pixels.numpy(),
        image_grid_thw=image_grid.numpy(),
        input_features=audio_features.numpy(),
        feature_attention_mask=feature_mask.numpy(),
        position_ids=position_ids.numpy(),
        image_features=image_features.numpy(),
        audio_embeddings=audio_embeddings.numpy(),
        hidden=hidden.numpy(),
        pixel_gradient=pixel_gradient.detach().numpy(),
        audio_gradient=audio_gradient.detach().numpy(),
        parameter_loss=training_losses[0],
        training_losses=np.asarray(training_losses),
        **parameter_gradients,
        **updated_parameters,
    )


if __name__ == "__main__":
    main()
