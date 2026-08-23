"""Generate one deterministic Transformers 5.6 LLaVA-NeXT oracle."""

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
    from transformers import LlavaNextConfig, LlavaNextModel

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "LLaVA-NeXT parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(67)
    config = LlavaNextConfig(
        image_token_index=63,
        image_grid_pinpoints=[[8, 8]],
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        projector_hidden_act="gelu",
        text_config={
            "model_type": "llama",
            "vocab_size": 64,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rope_theta": 10_000.0,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 0,
        },
        vision_config={
            "model_type": "clip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
        },
    )
    model = LlavaNextModel(config).eval().float()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor([[1, *([63] * 10), 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    pixels = (
        torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(1, 2, 3, 8, 8) / 100
    )
    image_sizes = torch.tensor([[8, 8]], dtype=torch.long)
    objective = torch.linspace(-0.5, 0.5, input_ids.numel() * 8).reshape(
        1, input_ids.shape[1], 8
    )

    def forward(pixel_values):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        ).last_hidden_state

    with torch.no_grad():
        hidden = forward(pixels)
    differentiable_pixels = pixels.clone().requires_grad_(True)
    pixel_loss = torch.sum(forward(differentiable_pixels) * objective)
    (pixel_gradient,) = torch.autograd.grad(pixel_loss, differentiable_pixels)

    model.zero_grad(set_to_none=True)
    parameter_loss = torch.sum(forward(pixels) * objective)
    parameter_loss.backward()
    gradients = {
        "parameter_gradient__" + name: value.grad.detach().numpy()
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    optimizer.step()
    updated = {
        "updated_parameter__" + name: value.detach().numpy()
        for name, value in model.named_parameters()
        if value.grad is not None
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        pixel_values=pixels.numpy(),
        image_sizes=image_sizes.numpy(),
        hidden=hidden.numpy(),
        pixel_gradient=pixel_gradient.detach().numpy(),
        parameter_loss=parameter_loss.detach().numpy(),
        **gradients,
        **updated,
    )


if __name__ == "__main__":
    main()
