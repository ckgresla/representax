"""Dataset-format adapters into native evaluator batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol

import jax.numpy as jnp

from representax.core import Route

if TYPE_CHECKING:
    from representax.models.processing import Processor

from .retrieval import (
    InformationRetrievalEvaluator,
    RetrievalEvaluationBatch,
    RetrievalInputKind,
    retrieval_evaluation_batch,
)


class RecordDataset(Protocol):
    """The random-access subset shared by sequences and Grain MapDatasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: Any) -> Any: ...


Records = Sequence[Mapping[str, Any]] | RecordDataset


def beir_evaluation(
    *,
    queries: Records,
    corpus: Records,
    qrels: Records,
    processor: Processor,
    batch_size: int,
    query_id_field: str = "_id",
    query_text_field: str = "text",
    document_id_field: str = "_id",
    document_text_field: str = "text",
    qrel_query_field: str = "query-id",
    qrel_document_field: str = "corpus-id",
    qrel_score_field: str = "score",
    name: str = "retrieval",
) -> tuple[InformationRetrievalEvaluator, Iterable[RetrievalEvaluationBatch]]:
    """Adapt random-access BEIR records without an intermediate dataset.

    ``queries``, ``corpus``, and ``qrels`` may be native Grain ``MapDataset``
    objects built from local, Hugging Face, or custom sources. Only identifiers
    and relevance judgments are retained on the host; text is processed lazily
    in fixed-size query batches followed by corpus batches.
    """

    if batch_size <= 0:
        raise ValueError("BEIR batch_size must be positive")
    query_rows = tuple(queries[index] for index in range(len(queries)))
    document_ids = tuple(
        str(corpus[index][document_id_field]) for index in range(len(corpus))
    )
    external_ids = [
        *(str(row[query_id_field]) for row in query_rows),
        *document_ids,
    ]
    id_map = {value: index for index, value in enumerate(dict.fromkeys(external_ids))}
    relevant: dict[int, set[int]] = {}
    for index in range(len(qrels)):
        row = qrels[index]
        if float(row.get(qrel_score_field, 1.0)) <= 0:
            continue
        query_id = id_map[str(row[qrel_query_field])]
        document_values = row[qrel_document_field]
        if not isinstance(document_values, (list, tuple, set, frozenset)):
            document_values = (document_values,)
        relevant.setdefault(query_id, set()).update(
            id_map[str(document_id)] for document_id in document_values
        )

    def batches(
        records: Records,
        *,
        id_field: str,
        text_field: str,
        kind: RetrievalInputKind,
        route: Route,
    ) -> Iterator[RetrievalEvaluationBatch]:
        for start in range(0, len(records), batch_size):
            stop = min(start + batch_size, len(records))
            rows = tuple(records[index] for index in range(start, stop))
            padding = batch_size - len(rows)
            texts = tuple(row[text_field] for row in rows) + ("",) * padding
            identifiers = tuple(id_map[str(row[id_field])] for row in rows)
            yield retrieval_evaluation_batch(
                processor(texts, route=route),
                jnp.asarray(
                    (*identifiers, *(-1 for _ in range(padding))),
                    dtype=jnp.int32,
                ),
                kind=kind,
                valid=jnp.asarray(
                    (True,) * len(rows) + (False,) * padding,
                    dtype=jnp.bool_,
                ),
            )

    evaluation_batches = chain(
        batches(
            queries,
            id_field=query_id_field,
            text_field=query_text_field,
            kind="query",
            route=Route.QUERY,
        ),
        batches(
            corpus,
            id_field=document_id_field,
            text_field=document_text_field,
            kind="document",
            route=Route.DOCUMENT,
        ),
    )
    return (
        InformationRetrievalEvaluator(relevant_documents=relevant, name=name),
        evaluation_batches,
    )


__all__ = ["RecordDataset", "beir_evaluation"]
