"""Emit deterministic upstream Qwen2.5-Omni preprocessing arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np


def _artifacts():
    from PIL import Image

    pixels = (np.arange(56 * 56 * 3, dtype=np.uint32).reshape(56, 56, 3) % 251).astype(
        np.uint8
    )
    return (
        Image.fromarray(pixels),
        np.sin(np.linspace(0, 200, 16_000, dtype=np.float32)),
        np.stack([np.roll(pixels, index, axis=1) for index in range(4)]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    import transformers
    from transformers import Qwen2_5OmniProcessor

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "Qwen2.5-Omni preprocessing parity requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    processor = Qwen2_5OmniProcessor.from_pretrained(arguments.checkpoint)
    image, audio, video = _artifacts()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "audio"},
                {"type": "video"},
                {"type": "text", "text": "A deterministic multimodal sample."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        cast(Any, conversation),
        chat_template="sentence_transformers",
        add_generation_prompt=True,
        tokenize=False,
    )
    values = processor(
        text=[text],
        images=[image],
        audio=[audio],
        videos=[video],
        padding=True,
        return_tensors="np",
        use_audio_in_video=False,
    )
    np.savez(
        arguments.output, **{name: np.asarray(value) for name, value in values.items()}
    )


if __name__ == "__main__":
    main()
