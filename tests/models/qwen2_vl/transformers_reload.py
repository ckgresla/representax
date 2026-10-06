"""Reload a native Qwen2/Qwen2.5-VL export with Transformers 5.6."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--generation", choices=("qwen2_vl", "qwen2_5_vl"), required=True
    )
    arguments = parser.parse_args()

    import torch
    import transformers

    if transformers.__version__ != "5.6.0":
        raise RuntimeError("Qwen2-VL reload requires transformers==5.6.0")
    model_type = (
        transformers.Qwen2VLModel
        if arguments.generation == "qwen2_vl"
        else transformers.Qwen2_5_VLModel
    )
    model = model_type.from_pretrained(arguments.checkpoint).eval().float()
    values = np.load(arguments.inputs)
    with torch.no_grad():
        hidden = model(
            input_ids=torch.from_numpy(values["input_ids"]),
            attention_mask=torch.from_numpy(values["attention_mask"]),
            mm_token_type_ids=torch.from_numpy(values["mm_token_type_ids"]),
            pixel_values=torch.from_numpy(values["pixel_values"]),
            image_grid_thw=torch.from_numpy(values["image_grid_thw"]),
        ).last_hidden_state
    np.savez(arguments.output, hidden=hidden.numpy())


if __name__ == "__main__":
    main()
