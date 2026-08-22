"""Generate deterministic Transformers 5.6 CLIP parity artifacts."""

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
    import torch.nn.functional as functional
    import transformers
    from transformers import CLIPConfig, CLIPModel

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "CLIP parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(37)
    config = cast(Any, CLIPConfig)(
        projection_dim=6,
        logit_scale_init_value=2.6592,
        text_config={
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 8,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
        },
    )
    config._attn_implementation = "eager"
    model = CLIPModel(config).eval().float()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(arguments.output_directory, safe_serialization=True)

    input_ids = torch.tensor(
        ((0, 4, 5, 2, 1, 1), (0, 3, 2, 1, 1, 1)),
        dtype=torch.long,
    )
    attention_mask = (input_ids != 1).long()
    pixels = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) / 255
    objective = torch.linspace(-0.5, 0.5, config.projection_dim)

    def representations(pixel_values):
        text = model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).pooler_output
        image = model.get_image_features(pixel_values=pixel_values).pooler_output
        return (
            functional.normalize(text, dim=-1),
            functional.normalize(image, dim=-1),
            functional.normalize(text + image, dim=-1),
        )

    with torch.no_grad():
        text, image, composed = representations(pixels)
    differentiable_pixels = pixels.clone().requires_grad_(True)
    _, _, differentiated = representations(differentiable_pixels)
    input_loss = torch.sum(differentiated * objective)
    (pixel_gradient,) = torch.autograd.grad(input_loss, differentiable_pixels)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    parameter_gradients = {}
    participating_names = set()
    training_losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, _, current = representations(pixels)
        loss = torch.sum(current * objective)
        training_losses.append(loss.detach().numpy())
        loss.backward()
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
    updated = {
        "updated_parameter__" + name: value.detach().numpy()
        for name, value in model.named_parameters()
        if name in participating_names
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        pixel_values=pixels.numpy(),
        text=text.numpy(),
        image=image.numpy(),
        composed=composed.numpy(),
        pixel_gradient=pixel_gradient.detach().numpy(),
        parameter_loss=training_losses[0],
        training_losses=np.asarray(training_losses),
        **parameter_gradients,
        **updated,
    )


if __name__ == "__main__":
    main()
