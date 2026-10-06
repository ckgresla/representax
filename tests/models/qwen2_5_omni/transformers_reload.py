"""Reload a native Qwen2.5-Omni export through pinned Transformers."""

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
    import transformers
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "Qwen2.5-Omni reload requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    reference = np.load(arguments.inputs)
    model = (
        Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            arguments.checkpoint,
            attn_implementation="eager",
        )
        .eval()
        .float()
    )
    with torch.no_grad():
        output = model(
            input_ids=torch.from_numpy(reference["input_ids"]),
            attention_mask=torch.from_numpy(reference["attention_mask"]),
            pixel_values=torch.from_numpy(reference["pixel_values"]),
            image_grid_thw=torch.from_numpy(reference["image_grid_thw"]),
            input_features=torch.from_numpy(reference["input_features"]),
            feature_attention_mask=torch.from_numpy(
                reference["feature_attention_mask"]
            ),
            output_hidden_states=True,
            return_dict=True,
        )
    np.savez(arguments.output, hidden=output.hidden_states[-1].numpy())


if __name__ == "__main__":
    main()
