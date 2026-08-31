"""Reload an exported scalar Qwen3 reward model in pinned Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    dtype = torch.float32 if arguments.dtype == "float32" else torch.bfloat16
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            arguments.checkpoint,
            local_files_only=True,
            dtype=dtype,
        )
        .to(arguments.device)
        .eval()
    )
    with np.load(arguments.inputs) as inputs:
        input_ids = torch.from_numpy(inputs["input_ids"]).long().to(arguments.device)
        attention_mask = (
            torch.from_numpy(inputs["attention_mask"]).long().to(arguments.device)
        )
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    np.savez(arguments.output, logits=logits.float().cpu().numpy())


if __name__ == "__main__":
    main()
