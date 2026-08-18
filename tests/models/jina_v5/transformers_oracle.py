"""Generate the pinned Sentence Transformers 5.6.1 Jina Small text oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    model = SentenceTransformer(
        str(arguments.checkpoint),
        local_files_only=True,
        trust_remote_code=True,
        model_kwargs={"modality": "text"},
        model_card_data=None,
    ).eval()
    model.max_seq_length = 32
    texts = [
        "A small native JAX encoder.",
        "The same checkpoint must produce the same representation.",
        "Query: Which library owns this tensor?",
        "Document: Representax owns the native Equinox implementation.",
    ]
    features = model.tokenize(texts)
    attention_mask = features["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 0)
    with torch.inference_mode():
        pooled = model(features)["sentence_embedding"]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output,
        input_ids=features["input_ids"].cpu().numpy(),
        attention_mask=attention_mask.cpu().numpy(),
        position_ids=position_ids.cpu().numpy(),
        pooled=pooled.float().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
