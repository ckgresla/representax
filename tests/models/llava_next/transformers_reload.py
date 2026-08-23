"""Reload a native LLaVA-NeXT export in Transformers."""

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
    from transformers import LlavaNextModel

    model = LlavaNextModel.from_pretrained(
        arguments.checkpoint, local_files_only=True
    ).eval()
    values = np.load(arguments.inputs)
    with torch.no_grad():
        hidden = model(
            input_ids=torch.from_numpy(values["input_ids"]),
            attention_mask=torch.from_numpy(values["attention_mask"]),
            pixel_values=torch.from_numpy(values["pixel_values"]),
            image_sizes=torch.from_numpy(values["image_sizes"]),
        ).last_hidden_state
    np.savez(arguments.output, hidden=hidden.numpy())


if __name__ == "__main__":
    main()
