"""Generate full-string embeddings with the pinned upstream oracle."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np

ORACLE_VERSION = "5.6.1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    version = importlib.metadata.version("sentence-transformers")
    if version != ORACLE_VERSION:
        raise RuntimeError(
            f"dense parity requires sentence-transformers=={ORACLE_VERSION}; "
            f"found {version}"
        )

    import sentence_transformers
    import torch

    from tests.models.upstream import (
        configure_torch_float32_highest,
        transformers_tacet,
    )

    texts = json.loads(arguments.texts.read_text())
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise TypeError("oracle texts must be a JSON array of strings")
    configure_torch_float32_highest()
    with transformers_tacet():
        model = sentence_transformers.SentenceTransformer(
            str(arguments.checkpoint),
            device="cuda",
            local_files_only=True,
        )
    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    np.save(arguments.output, np.asarray(embeddings, dtype=np.float32))
    arguments.metadata.write_text(
        json.dumps(
            {
                "sentence_transformers": version,
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "count": len(texts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
