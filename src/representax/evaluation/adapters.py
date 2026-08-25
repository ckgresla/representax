"""Dataset-format adapters into native evaluator batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from representax.models.processing import Processor

from .retrieval import (
    InformationRetrievalEvaluator,
    RetrievalEvaluationBatch,
    RetrievalInputKind,
    retrieval_evaluation_batch,
)


def nanobeir_evaluation(
    *,
    queries: Sequence[Mapping[str, Any]],
    corpus: Sequence[Mapping[str, Any]],
    qrels: Sequence[Mapping[str, Any]],
    processor: Processor,
    batch_size: int,
    query_id_field: str = "_id",
    query_text_field: str = "text",
    document_id_field: str = "_id",
    document_text_field: str = "text",
    qrel_query_field: str = "query-id",
    qrel_document_field: str = "corpus-id",
    qrel_score_field: str = "score",
    name: str = "nanobeir",
) -> tuple[InformationRetrievalEvaluator, tuple[RetrievalEvaluationBatch, ...]]:
    """Adapt NanoBEIR/BEIR-style records without creating an intermediate dataset."""

    if batch_size <= 0:
        raise ValueError("NanoBEIR batch_size must be positive")
    external_ids = [
        *(str(row[query_id_field]) for row in queries),
        *(str(row[document_id_field]) for row in corpus),
    ]
    id_map = {value: index for index, value in enumerate(dict.fromkeys(external_ids))}
    relevant: dict[int, set[int]] = {}
    for row in qrels:
        if float(row.get(qrel_score_field, 1.0)) <= 0:
            continue
        query_id = id_map[str(row[qrel_query_field])]
        document_id = id_map[str(row[qrel_document_field])]
        relevant.setdefault(query_id, set()).add(document_id)

    def batches(
        records: Sequence[Mapping[str, Any]],
        *,
        id_field: str,
        text_field: str,
        kind: RetrievalInputKind,
    ) -> list[RetrievalEvaluationBatch]:
        output = []
        for start in range(0, len(records), batch_size):
            rows = records[start : start + batch_size]
            output.append(
                retrieval_evaluation_batch(
                    processor([row[text_field] for row in rows]),
                    jnp.asarray(
                        [id_map[str(row[id_field])] for row in rows],
                        dtype=jnp.int32,
                    ),
                    kind=kind,
                )
            )
        return output

    evaluation_batches = (
        *batches(
            queries,
            id_field=query_id_field,
            text_field=query_text_field,
            kind="query",
        ),
        *batches(
            corpus,
            id_field=document_id_field,
            text_field=document_text_field,
            kind="document",
        ),
    )
    return (
        InformationRetrievalEvaluator(relevant_documents=relevant, name=name),
        evaluation_batches,
    )


__all__ = ["nanobeir_evaluation"]
