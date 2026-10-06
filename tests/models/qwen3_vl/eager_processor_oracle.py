"""Record Eager Embed V1 preprocessing from the pinned Transformers processor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        arguments.checkpoint,
        padding_side="left",
    )
    pixels = (np.arange(56 * 84 * 3, dtype=np.uint32).reshape(56, 84, 3) % 251).astype(
        np.uint8
    )
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "A deterministic document page."},
            ],
        }
    ]
    rendered = (
        processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<|endoftext|>"
    )
    features = processor(
        text=[rendered],
        images=[Image.fromarray(pixels)],
        padding=True,
        return_tensors="pt",
    )
    np.savez(
        arguments.output,
        source_pixels=pixels,
        **{name: value.detach().cpu().numpy() for name, value in features.items()},
    )


if __name__ == "__main__":
    main()
