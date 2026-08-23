"""Generate real Sentence Transformers preprocessing and score oracles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from sentence_transformers.util import batch_to_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    pairs: Any = [
        (
            "Which planet is known as the Red Planet?",
            "Mars is often called the Red Planet because of iron oxide.",
        ),
        (
            "Which planet is known as the Red Planet?",
            "Venus has a dense carbon-dioxide atmosphere.",
        ),
    ]
    model = CrossEncoder(
        str(arguments.checkpoint),
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    prompt = None
    if model.default_prompt_name is not None:
        prompt = model.prompts[model.default_prompt_name]
    features = model.preprocess(pairs, prompt=prompt)
    model.eval()
    features = batch_to_device(features, model.device)
    activation = model.activation_fn
    if activation is None:
        raise RuntimeError("CrossEncoder did not resolve an inference activation")
    with torch.no_grad():
        raw_scores = model.forward(features)["scores"].squeeze(-1)
        scores = activation(raw_scores)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.output,
        input_ids=features["input_ids"].cpu().numpy(),
        attention_mask=features["attention_mask"].cpu().numpy(),
        logits=raw_scores.float().cpu().numpy(),
        scores=scores.float().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
