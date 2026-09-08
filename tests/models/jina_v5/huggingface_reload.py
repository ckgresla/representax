"""Verify a Representax-exported Jina checkpoint in its upstream runtime."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from transformers import AutoModel, Qwen2TokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        checkpoint,
        local_files_only=True,
        padding_side="right",
    )
    features = tokenizer(
        ["Query: a quiet harbor"],
        padding=True,
        return_tensors="pt",
    )
    model = (
        AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
            modality="text",
        )
        .cuda()
        .eval()
    )
    features = {name: value.cuda() for name, value in features.items()}
    with torch.inference_mode():
        embedding = model.embed(**features).float().cpu().numpy()
    if embedding.ndim != 2 or not np.all(np.isfinite(embedding)):
        raise RuntimeError("exported Jina checkpoint produced invalid embeddings")


if __name__ == "__main__":
    main()
