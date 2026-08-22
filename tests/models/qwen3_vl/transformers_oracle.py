"""Generate a deterministic Transformers 5.3 Qwen3-VL parity oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            "Qwen3-VL parity requires transformers==5.3.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(29)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "mrope_section": [1, 1, 0],
            },
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,
            "tie_word_embeddings": True,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 12,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
        tie_word_embeddings=True,
    )
    model = Qwen3VLForConditionalGeneration(config).eval().float()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor([[1, 28, 29, 31, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    modality_ids = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.long)
    pixels = torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24) / 100
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    objective = torch.linspace(-0.5, 0.5, config.text_config.hidden_size)

    def forward(pixel_values):
        return model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            mm_token_type_ids=modality_ids,
        ).last_hidden_state

    with torch.no_grad():
        hidden = forward(pixels)
    differentiable_pixels = pixels.clone().requires_grad_(True)
    pixel_hidden = forward(differentiable_pixels)
    pixel_loss = torch.sum(pixel_hidden * objective)
    (pixel_gradient,) = torch.autograd.grad(pixel_loss, differentiable_pixels)

    model.zero_grad(set_to_none=True)
    training_hidden = forward(pixels)
    parameter_loss = torch.sum(training_hidden * objective)
    parameter_loss.backward()
    parameter_gradients = {
        "parameter_gradient__model." + name: value.grad.detach().numpy()
        for name, value in model.model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    optimizer.step()
    updated = {
        "updated_parameter__model." + name: value.detach().numpy()
        for name, value in model.model.named_parameters()
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        mm_token_type_ids=modality_ids.numpy(),
        pixel_values=pixels.numpy(),
        image_grid_thw=grid.numpy(),
        hidden=hidden.numpy(),
        pixel_gradient=pixel_gradient.detach().numpy(),
        parameter_loss=parameter_loss.detach().numpy(),
        **parameter_gradients,
        **updated,
    )


if __name__ == "__main__":
    main()
