"""Pinned upstream Llama Nemotron VL preprocessing and inference oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor


def _image() -> tuple[np.ndarray, Image.Image]:
    pixels = (
        np.arange(64 * 512 * 3, dtype=np.uint32).reshape((64, 512, 3)) % 251
    ).astype(np.uint8)
    return pixels, Image.fromarray(pixels)


def _arrays(prefix: str, values, output: dict[str, np.ndarray]) -> None:
    for name in ("input_ids", "attention_mask", "pixel_values"):
        value = values.get(name)
        if value is not None:
            output[f"{prefix}__{name}"] = value.detach().float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("embedding", "reranking"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    torch.manual_seed(97)
    torch.set_float32_matmul_precision("highest")
    pixels, image = _image()
    processor = AutoProcessor.from_pretrained(
        arguments.checkpoint,
        trust_remote_code=True,
        max_input_tiles=6,
        use_thumbnail=True,
    )
    model_class = (
        AutoModel
        if arguments.mode == "embedding"
        else AutoModelForSequenceClassification
    )
    model = (
        model_class.from_pretrained(
            arguments.checkpoint,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    query = "Which item contains a chart?"
    document = "A deterministic chart."
    if arguments.mode == "embedding":
        batches = {
            "query": processor.process_queries([query]),
            "text": processor.process_documents([{"image": "", "text": document}]),
            "image": processor.process_documents([{"image": image, "text": ""}]),
            "composed": processor.process_documents(
                [{"image": image, "text": document}]
            ),
        }
    else:

        def pair(image_value, text: str):
            prompt = processor.prompt_template_question_passage(query, text)
            return processor.process_query_documents(
                [{"image": image_value, "text": prompt}]
            )

        batches = {
            "text": pair("", document),
            "image": pair(image, ""),
            "composed": pair(image, document),
        }

    output: dict[str, np.ndarray] = {"source_pixels": pixels}
    with torch.inference_mode():
        for case, batch in batches.items():
            _arrays(case, batch, output)
            inputs = {
                name: value.cuda()
                for name, value in batch.items()
                if name in {"input_ids", "attention_mask", "pixel_values"}
                and value is not None
            }
            if arguments.mode == "embedding":
                result = model._embed_batch(inputs)
            else:
                result = model(**inputs, return_dict=True).logits
            output[f"{case}__output"] = result.float().cpu().numpy()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.output, **cast(Any, output))


if __name__ == "__main__":
    main()
