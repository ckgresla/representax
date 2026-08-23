"""Reload one exported BidirLM Omni checkpoint in Sentence Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    oracle = np.load(arguments.oracle)
    value = [
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.fromarray(oracle["source_pixels"]),
                    },
                    {
                        "type": "audio",
                        "audio": {
                            "array": oracle["source_audio"],
                            "sampling_rate": 16_000,
                        },
                    },
                    {"type": "text", "text": "Compare these signals."},
                ],
            }
        ]
    ]
    model = SentenceTransformer(
        str(arguments.checkpoint),
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        device="cuda",
    )
    with torch.inference_mode():
        output = model.encode(
            value,
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, output=output)


if __name__ == "__main__":
    main()
