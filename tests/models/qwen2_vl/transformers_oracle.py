"""Generate deterministic Transformers 5.6 Qwen2/Qwen2.5-VL oracles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mapped(name: str) -> str:
    return (
        "model." + name.removeprefix("language_model.")
        if name.startswith("language_model.")
        else name
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--generation", choices=("qwen2_vl", "qwen2_5_vl"), required=True
    )
    arguments = parser.parse_args()

    import torch
    import transformers
    from safetensors.torch import save_file
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLModel,
        Qwen2VLConfig,
        Qwen2VLModel,
    )

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "Qwen2-VL parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(41 if arguments.generation == "qwen2_vl" else 43)
    common: dict[str, Any] = dict(
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=28,
        vision_end_token_id=31,
        pad_token_id=0,
    )
    text = {
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
    }
    if arguments.generation == "qwen2_5_vl":
        config = Qwen2_5_VLConfig(
            text_config=text,
            vision_config={
                "depth": 2,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_heads": 2,
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "out_hidden_size": 8,
                "window_size": 8,
                "fullatt_block_indexes": [1],
            },
            **common,
        )
        model = Qwen2_5_VLModel(config).eval().float()
    else:
        config = Qwen2VLConfig(
            text_config=text,
            vision_config={
                "depth": 2,
                "embed_dim": 8,
                "hidden_size": 8,
                "num_heads": 2,
                "mlp_ratio": 4,
                "hidden_act": "quick_gelu",
                "in_channels": 3,
                "patch_size": 2,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
            },
            **common,
        )
        model = Qwen2VLModel(config).eval().float()

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    (arguments.output_directory / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    save_file(
        {
            _mapped(name): value.detach().contiguous()
            for name, value in model.state_dict().items()
        },
        arguments.output_directory / "model.safetensors",
    )
    input_ids = torch.tensor([[1, 28, 29, 31, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    token_types = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.long)
    pixels = torch.arange(4 * 24, dtype=torch.float32).reshape(4, 24) / 100
    grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
    objective = torch.linspace(-0.5, 0.5, 8)

    def forward(pixel_values):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            mm_token_type_ids=token_types,
        ).last_hidden_state

    with torch.no_grad():
        hidden = forward(pixels)
    differentiable_pixels = pixels.clone().requires_grad_(True)
    pixel_hidden = forward(differentiable_pixels)
    pixel_loss = torch.sum(pixel_hidden * objective)
    (pixel_gradient,) = torch.autograd.grad(pixel_loss, differentiable_pixels)

    model.zero_grad(set_to_none=True)
    parameter_loss = torch.sum(forward(pixels) * objective)
    parameter_loss.backward()
    gradients = {
        "parameter_gradient__" + _mapped(name): value.grad.detach().numpy()
        for name, value in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    optimizer.step()
    updated = {
        "updated_parameter__" + _mapped(name): value.detach().numpy()
        for name, value in model.named_parameters()
    }
    np.savez(
        arguments.output_directory / "oracle.npz",
        input_ids=input_ids.numpy(),
        attention_mask=attention_mask.numpy(),
        mm_token_type_ids=token_types.numpy(),
        pixel_values=pixels.numpy(),
        image_grid_thw=grid.numpy(),
        hidden=hidden.numpy(),
        pixel_gradient=pixel_gradient.detach().numpy(),
        parameter_loss=parameter_loss.detach().numpy(),
        **gradients,
        **updated,
    )


if __name__ == "__main__":
    main()
