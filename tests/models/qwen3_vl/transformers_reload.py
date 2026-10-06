"""Reload an exported Qwen3-VL checkpoint in pinned Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    from transformers import Qwen3VLForConditionalGeneration

    reference = np.load(arguments.inputs)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(arguments.checkpoint)
        .eval()
        .float()
    )
    with torch.no_grad():
        hidden = model.model(
            input_ids=torch.from_numpy(reference["input_ids"]),
            attention_mask=torch.from_numpy(reference["attention_mask"]),
            pixel_values=torch.from_numpy(reference["pixel_values"]),
            image_grid_thw=torch.from_numpy(reference["image_grid_thw"]),
            mm_token_type_ids=torch.from_numpy(reference["mm_token_type_ids"]),
        ).last_hidden_state
    np.savez(arguments.output, hidden=hidden.numpy())


if __name__ == "__main__":
    main()
