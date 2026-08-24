"""Generate the pinned PyLate tensor oracle used by repository parity tests."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


def generate(output_directory: Path) -> None:
    """Run PyLate once and persist all inputs, outputs, and gradients."""

    import torch
    from pylate.scores import colbert_scores

    if version("pylate") != "1.6.0":
        raise RuntimeError("the late-interaction oracle requires pylate==1.6.0")

    query_values = np.abs(
        np.random.default_rng(7).normal(size=(3, 4, 5)).astype(np.float32)
    )
    document_values = np.abs(
        np.random.default_rng(8).normal(size=(3, 6, 5)).astype(np.float32)
    )
    query_values /= np.linalg.norm(query_values, axis=-1, keepdims=True)
    document_values /= np.linalg.norm(document_values, axis=-1, keepdims=True)
    query_valid = np.asarray(
        [[True, True, True, False], [True, True, False, False], [True] * 4]
    )
    document_valid = np.asarray(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    positive_mask = np.eye(3, dtype=np.bool_)
    temperature = np.asarray(0.07, dtype=np.float32)

    queries = torch.tensor(query_values, requires_grad=True)
    documents = torch.tensor(document_values, requires_grad=True)
    scores = colbert_scores(
        queries,
        documents,
        queries_mask=torch.tensor(query_valid),
        documents_mask=torch.tensor(document_valid),
        backend="torch",
    )
    loss = torch.nn.functional.cross_entropy(
        scores / float(temperature),
        torch.arange(3),
    )
    loss.backward()
    if queries.grad is None or documents.grad is None:
        raise RuntimeError("PyLate oracle did not produce representation gradients")

    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "oracle.npz",
        query_values=query_values,
        document_values=document_values,
        query_valid=query_valid,
        document_valid=document_valid,
        positive_mask=positive_mask,
        temperature=temperature,
        scores=scores.detach().numpy(),
        loss=loss.detach().numpy(),
        query_gradient=queries.grad.numpy(),
        document_gradient=documents.grad.numpy(),
    )
    metadata = {
        "backend": "torch",
        "oracle": "pylate.scores.colbert_scores + torch cross_entropy",
        "pylate_version": version("pylate"),
        "sentence_transformers_version": version("sentence-transformers"),
        "source": "https://github.com/lightonai/pylate/tree/1.6.0",
        "torch_version": version("torch"),
        "transformers_version": version("transformers"),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.output_directory)


if __name__ == "__main__":
    main()
