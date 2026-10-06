"""Reload a native Llama Nemotron VL export in its upstream HF runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification

    config = json.loads((arguments.checkpoint / "config.json").read_text())
    reranking = config["model_type"] == "llama_nemotron_vl_rerank"
    model_class = AutoModelForSequenceClassification if reranking else AutoModel
    model = (
        model_class.from_pretrained(
            arguments.checkpoint,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    values = np.load(arguments.inputs)
    inputs = {
        "input_ids": torch.from_numpy(values[f"{arguments.case}__input_ids"])
        .long()
        .cuda(),
        "attention_mask": torch.from_numpy(values[f"{arguments.case}__attention_mask"])
        .long()
        .cuda(),
    }
    pixels = values.get(f"{arguments.case}__pixel_values")
    if pixels is not None:
        inputs["pixel_values"] = torch.from_numpy(pixels).to(
            device="cuda", dtype=torch.bfloat16
        )
    with torch.no_grad():
        output = (
            model(**inputs, return_dict=True).logits
            if reranking
            else model._embed_batch(inputs)
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output, output.float().cpu().numpy())


if __name__ == "__main__":
    main()
