"""Generate maximum-precision Jina v5 Omni reference embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, Qwen2TokenizerFast, Qwen2VLImageProcessor

from representax.core import Route
from representax.models.jina_v5 import (
    JinaV5OmniConfig,
    prepare_jina_v5_omni_features,
)


def _cases() -> dict[str, object]:
    image = np.arange(56 * 56 * 3, dtype=np.uint8).reshape(56, 56, 3)
    video = np.stack((image, image[::-1]))
    audio = np.sin(np.arange(16_000, dtype=np.float32) / 100)
    return {
        "text": "one two",
        "image": {"image": image},
        "audio": {"audio": audio},
        "video": {"video": video},
        "fused": {
            "text": "one",
            "image": image,
            "audio": audio,
            "video": video,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint
    config = JinaV5OmniConfig.from_hf_config(
        json.loads((checkpoint / "config.json").read_text())
    )
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        checkpoint, local_files_only=True, padding_side="right"
    )
    image_processor = Qwen2VLImageProcessor.from_pretrained(
        checkpoint,
        local_files_only=True,
        min_pixels=56 * 56,
        max_pixels=56 * 56,
    )
    model = (
        AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
            modality="omni",
        )
        .float()
        .cuda()
        .eval()
    )
    outputs = {}
    for name, sample in _cases().items():
        features = prepare_jina_v5_omni_features(
            [sample],
            route=Route.QUERY,
            checkpoint=checkpoint,
            config=config,
            tokenizer=tokenizer,
            image_processor=image_processor,
            maximum_sequence_length=128,
            video_min_pixels=56 * 56,
            video_max_pixels=56 * 56,
        )
        inputs = {
            key: torch.from_numpy(np.asarray(value)).cuda()
            for key, value in features.items()
            if not key.startswith("_") and key != "video_second_per_grid"
        }
        with torch.inference_mode():
            outputs[name] = model.embed(**inputs).float().cpu().numpy()
    np.savez(arguments.output, **outputs)


if __name__ == "__main__":
    main()
