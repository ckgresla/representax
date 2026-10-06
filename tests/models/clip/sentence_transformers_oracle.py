"""Generate real-checkpoint Sentence Transformers 5.6.1 CLIP evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _image():
    from PIL import Image

    pixels = (
        np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3) % 251
    ).astype(np.uint8)
    return Image.fromarray(pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--composition", action="store_true")
    arguments = parser.parse_args()

    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPProcessor

    if sentence_transformers.__version__ != "5.6.1":
        raise RuntimeError(
            "CLIP parity requires sentence-transformers==5.6.1; "
            f"found {sentence_transformers.__version__}"
        )
    source = arguments.checkpoint
    if (source / "0_CLIPModel" / "config.json").is_file():
        source = source / "0_CLIPModel"
    processor = CLIPProcessor.from_pretrained(source, local_files_only=True)
    image = _image()
    caption = "A deterministic caption."
    tokens = processor.tokenizer(
        [caption],
        padding="max_length",
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
        return_tensors="np",
    )
    images = processor.image_processor(images=[image], return_tensors="np")

    model = SentenceTransformer(
        str(arguments.checkpoint),
        trust_remote_code=True,
        local_files_only=True,
        device="cpu",
        model_kwargs={"attn_implementation": "eager", "dtype": torch.float32},
    )
    values: dict[str, np.ndarray] = {
        "input_ids": np.asarray(tokens["input_ids"]),
        "attention_mask": np.asarray(tokens["attention_mask"]),
        "pixel_values": np.asarray(images["pixel_values"]),
        "text": np.asarray(model.encode([caption], convert_to_numpy=True)),
        "image": np.asarray(model.encode([image], convert_to_numpy=True)),
    }
    if arguments.composition:
        values["composed"] = np.asarray(
            model.encode(
                [{"text": caption, "image": image}],
                convert_to_numpy=True,
            )
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.composition:
        np.savez(
            arguments.output,
            input_ids=values["input_ids"],
            attention_mask=values["attention_mask"],
            pixel_values=values["pixel_values"],
            text=values["text"],
            image=values["image"],
            composed=values["composed"],
        )
    else:
        np.savez(
            arguments.output,
            input_ids=values["input_ids"],
            attention_mask=values["attention_mask"],
            pixel_values=values["pixel_values"],
            text=values["text"],
            image=values["image"],
        )


if __name__ == "__main__":
    main()
