"""Reload a native CLIP export through pinned Transformers."""

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
    import torch.nn.functional as functional
    import transformers
    from transformers import CLIPModel

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "CLIP reload requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    values = np.load(arguments.inputs)
    model = (
        CLIPModel.from_pretrained(
            arguments.checkpoint,
            attn_implementation="eager",
        )
        .eval()
        .float()
    )
    with torch.no_grad():
        text = model.get_text_features(
            input_ids=torch.from_numpy(values["input_ids"]),
            attention_mask=torch.from_numpy(values["attention_mask"]),
        ).pooler_output
        image = model.get_image_features(
            pixel_values=torch.from_numpy(values["pixel_values"])
        ).pooler_output
        composed = functional.normalize(text + image, dim=-1)
    np.savez(arguments.output, composed=composed.numpy())


if __name__ == "__main__":
    main()
