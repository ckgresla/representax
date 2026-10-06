"""Evaluate one revision-pinned NanoBEIR split through generic IR.

NanoBEIR is a collection of BEIR-format datasets, not a Representax evaluator.
This example resolves its queries, corpus, and qrels as three native Grain
``MapDataset`` sources, then maps those records into the ordinary
``InformationRetrievalEvaluator``. No intermediate dataset is created.

Run with ``pip install -e '.[hf]'`` and, optionally, a local model snapshot:

    python -m examples.evaluation.nanobeir \
      --dataset NanoMSMARCO \
      --model /immutable/path/to/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from representax import data
from representax.evaluation import beir_evaluation
from representax.evaluation.retrieval import (
    InformationRetrievalEvaluator,
    RetrievalEvaluationBatch,
)
from representax.models import SentenceEncoder
from representax.models.processing import Processor
from representax.train import EvaluationRunner

NANOBEIR_DATASET_ID = "sentence-transformers/NanoBEIR-en"
NANOBEIR_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

NanoBEIRSourceKind = Literal["queries", "corpus", "qrels"]


def identity(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Preserve canonical BEIR rows when Grain resolves the source."""

    return record


def nanobeir_source(
    kind: NanoBEIRSourceKind,
    *,
    dataset: str,
) -> data.DataDistributionConfig:
    """Declare one immutable NanoBEIR source as a one-source distribution."""

    return data.mix(
        data.source(
            f"hf://{NANOBEIR_DATASET_ID}",
            revision=NANOBEIR_REVISION,
            subset=kind,
            split=dataset,
            map=identity,
            name=kind,
        ),
        shuffle=False,
    )


def nanobeir_evaluation(
    processor: Processor,
    *,
    dataset: str = "NanoMSMARCO",
    batch_size: int = 64,
) -> tuple[InformationRetrievalEvaluator, Iterable[RetrievalEvaluationBatch]]:
    """Build generic IR evaluation from lazy NanoBEIR Grain sources."""

    queries = data.build_dataset(nanobeir_source("queries", dataset=dataset))
    corpus = data.build_dataset(nanobeir_source("corpus", dataset=dataset))
    qrels = data.build_dataset(nanobeir_source("qrels", dataset=dataset))
    return beir_evaluation(
        queries=queries,
        corpus=corpus,
        qrels=qrels,
        processor=processor,
        batch_size=batch_size,
        name=dataset,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NanoMSMARCO")
    parser.add_argument("--model", default=MINILM_MODEL_ID)
    parser.add_argument("--model-revision", default=MINILM_REVISION)
    parser.add_argument("--batch-size", type=int, default=64)
    arguments = parser.parse_args()

    model, processor = SentenceEncoder.load_from_hf(
        arguments.model,
        revision=arguments.model_revision,
    )
    evaluator, batches = nanobeir_evaluation(
        processor,
        dataset=arguments.dataset,
        batch_size=arguments.batch_size,
    )
    result = EvaluationRunner(evaluator).run(model, batches)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
