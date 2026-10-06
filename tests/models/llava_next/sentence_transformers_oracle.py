"""Run pinned real LLaVA-NeXT artifacts through Sentence Transformers 5.6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _image():
    from PIL import Image

    pixels = (
        np.arange(64 * 512 * 3, dtype=np.uint32).reshape((64, 512, 3)) % 251
    ).astype(np.uint8)
    return Image.fromarray(pixels), pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("bge", "e5"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sentence-transformers-version", default="5.6.1")
    arguments = parser.parse_args()

    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    if (
        sentence_transformers.__version__
        != arguments.expected_sentence_transformers_version
    ):
        raise RuntimeError(
            "LLaVA-NeXT acceptance requires sentence-transformers=="
            f"{arguments.expected_sentence_transformers_version}; "
            f"found {sentence_transformers.__version__}"
        )
    torch.set_float32_matmul_precision("highest")
    model = SentenceTransformer(
        str(arguments.checkpoint),
        device="cuda",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    image, source_pixels = _image()
    instruction = "Retrieve the visual document that answers the question."
    text = "A chart with deterministic colored pixels."
    sample = {"text": text, "image": image}
    encode_kwargs = {"prompt": instruction} if arguments.variant == "bge" else {}
    embedding = model.encode(
        sample,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **encode_kwargs,
    )
    arrays = {
        "source_pixels": source_pixels,
        "embedding": np.asarray(embedding),
    }
    for case, value in (("text", text), ("image", image), ("composed", sample)):
        features = model[0].preprocess([value], **encode_kwargs)
        for name in ("input_ids", "attention_mask", "pixel_values", "image_sizes"):
            if name not in features:
                continue
            key = f"{case}__{name}"
            arrays[key] = features[name].detach().cpu().float().numpy()
            if name in {"input_ids", "attention_mask", "image_sizes"}:
                arrays[key] = arrays[key].astype(np.int64)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **arrays)
    arguments.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "sentence_transformers": sentence_transformers.__version__,
                "variant": arguments.variant,
                "embedding_shape": list(np.asarray(embedding).shape),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
