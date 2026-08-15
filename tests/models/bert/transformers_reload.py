"""Reload a native BERT export through pinned Transformers and execute it."""

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
    from transformers import BertModel

    if transformers.__version__ != "5.3.0":
        raise RuntimeError(
            "BERT reload requires transformers==5.3.0; "
            f"found {transformers.__version__}"
        )
    reference = np.load(arguments.inputs)
    model = BertModel.from_pretrained(arguments.checkpoint).eval()
    with torch.no_grad():
        output = model(
            input_ids=torch.from_numpy(reference["input_ids"]),
            attention_mask=torch.from_numpy(reference["attention_mask"]),
            token_type_ids=torch.from_numpy(reference["token_type_ids"]),
        )
    np.savez(
        arguments.output,
        hidden=output.last_hidden_state.numpy(),
        pooler=output.pooler_output.numpy(),
    )


if __name__ == "__main__":
    main()
