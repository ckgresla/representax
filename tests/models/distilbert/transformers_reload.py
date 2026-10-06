"""Reload a native DistilBERT export with Transformers 5.6."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    import torch
    import transformers
    from transformers import DistilBertModel

    if transformers.__version__ != "5.6.0":
        raise RuntimeError(
            "DistilBERT reload requires transformers==5.6.0; "
            f"found {transformers.__version__}"
        )
    values = np.load(arguments.inputs)
    model = DistilBertModel.from_pretrained(arguments.checkpoint).eval()
    with torch.no_grad():
        hidden = model(
            input_ids=torch.from_numpy(values["input_ids"]),
            attention_mask=torch.from_numpy(values["attention_mask"]),
        )[0]
    np.savez(arguments.output, hidden=hidden.numpy())


if __name__ == "__main__":
    main()
