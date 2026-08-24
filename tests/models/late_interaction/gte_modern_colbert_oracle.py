"""Generate the pinned real-model oracle from isolated PyLate 1.6.0."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

QUERIES = (
    "What makes late interaction useful for retrieval?",
    "How does ColBERT score a document?",
)
DOCUMENTS = (
    "ColBERT retains contextual token vectors and applies MaxSim at retrieval time.",
    "A dense encoder instead produces one vector for the complete document.",
)


def generate(checkpoint: Path, output_directory: Path) -> None:
    """Run the upstream model once and persist inputs and projected tokens."""

    import torch
    from pylate import models

    if version("pylate") != "1.6.0":
        raise RuntimeError("the real-model oracle requires pylate==1.6.0")
    model = models.ColBERT(
        model_name_or_path=str(checkpoint),
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
        model_kwargs={
            "attn_implementation": "eager",
            "torch_dtype": torch.float32,
        },
    )
    model.eval()
    arrays: dict[str, Any] = {}
    for name, texts, is_query in (
        ("query", QUERIES, True),
        ("document", DOCUMENTS, False),
    ):
        features = model.tokenize(list(texts), is_query=is_query, pad=True)
        features = {key: value.to(model.device) for key, value in features.items()}
        with torch.no_grad():
            output = model.forward(features)
        projected = output["token_embeddings"]
        if is_query:
            valid = (
                torch.ones_like(features["attention_mask"], dtype=torch.bool)
                if model.do_query_expansion
                else output["attention_mask"].bool()
            )
        else:
            valid = model.skiplist_mask(features["input_ids"], model.skiplist)
            valid &= output["attention_mask"].bool()
        normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
        normalized = torch.where(valid[..., None], normalized, 0.0)
        arrays[f"{name}_input_ids"] = features["input_ids"].cpu().numpy()
        arrays[f"{name}_attention_mask"] = features["attention_mask"].cpu().numpy()
        arrays[f"{name}_projected"] = projected.cpu().numpy()
        arrays[f"{name}_normalized"] = normalized.cpu().numpy()
        arrays[f"{name}_valid"] = valid.cpu().numpy()

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_directory / "oracle.npz", **arrays)
    metadata = {
        "checkpoint": "lightonai/GTE-ModernColBERT-v1",
        "checkpoint_revision": checkpoint.name,
        "queries": list(QUERIES),
        "documents": list(DOCUMENTS),
        "pylate_version": version("pylate"),
        "attention_implementation": "eager",
        "sentence_transformers_version": version("sentence-transformers"),
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.checkpoint, arguments.output_directory)


if __name__ == "__main__":
    main()
