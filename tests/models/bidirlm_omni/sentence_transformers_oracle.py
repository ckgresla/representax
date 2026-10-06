"""Generate the pinned Sentence Transformers BidirLM Omni oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor


def _fixtures() -> tuple[
    np.ndarray,
    Image.Image,
    tuple[Image.Image, ...],
    dict[str, Any],
]:
    pixels = np.arange(64 * 96 * 3, dtype=np.uint8).reshape((64, 96, 3))
    video = tuple(
        Image.fromarray(np.roll(pixels, frame * 4, axis=1)) for frame in range(4)
    )
    sample_rate = 16_000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    audio = {
        "array": np.sin(2 * np.pi * 440 * time).astype(np.float32),
        "sampling_rate": sample_rate,
    }
    return pixels, Image.fromarray(pixels), video, audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.manual_seed(131)
    torch.set_float32_matmul_precision("highest")
    processor = AutoProcessor.from_pretrained(
        arguments.checkpoint, trust_remote_code=True
    )
    model = SentenceTransformer(
        str(arguments.checkpoint),
        trust_remote_code=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
        device="cuda",
    )
    pixels, image, video, audio = _fixtures()
    cases = {
        "text": [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "A precise red square."}],
                }
            ]
        ],
        "image": [[{"role": "user", "content": [{"type": "image", "image": image}]}]],
        "video": [[{"role": "user", "content": [{"type": "video", "video": video}]}]],
        "audio": [[{"role": "user", "content": [{"type": "audio", "audio": audio}]}]],
        "composed": [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": "Compare these signals."},
                    ],
                }
            ]
        ],
    }
    output: dict[str, np.ndarray] = {
        "source_pixels": pixels,
        "source_video": np.stack(tuple(np.asarray(frame) for frame in video)),
        "source_audio": audio["array"],
    }
    for name, values in cases.items():
        features = processor.apply_chat_template(
            values,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )
        for field, value in features.items():
            if hasattr(value, "shape"):
                output[f"{name}__{field}"] = np.asarray(value)
        with torch.inference_mode():
            output[f"{name}__output"] = model.encode(
                values,
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **cast(Any, output))


if __name__ == "__main__":
    main()
