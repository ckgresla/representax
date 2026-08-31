"""Dataset-format adapters into native evaluator batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any, Protocol

import jax.numpy as jnp

from representax.core import Route
from representax.data import DataSourceConfig, source
from representax.tasks.classification import pair_classification_batch
from representax.tasks.triplet import labeled_examples_batch

if TYPE_CHECKING:
    from representax.models.processing import Processor

from .representation import EvaluationSplit, labeled_evaluation_batch
from .retrieval import (
    InformationRetrievalEvaluator,
    RetrievalEvaluationBatch,
    RetrievalInputKind,
    retrieval_evaluation_batch,
)


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationSource:
    """One immutable upstream dataset used by a canonical evaluator gate."""

    name: str
    source: DataSourceConfig
    expected_rows: int
    license: str | None


SPRINT_DUPLICATE_QUESTIONS = CanonicalEvaluationSource(
    name="sprint_duplicate_questions",
    source=source(
        "hf://mteb/sprintduplicatequestions-pairclassification",
        map="representax.data.identity",
        revision="d66bd1f72af766a5cc4b0ca5e00c162f89e8cc46",
        split="test",
    ),
    expected_rows=101_000,
    license=None,
)
BANKING77_TRAIN = CanonicalEvaluationSource(
    name="banking77_train",
    source=source(
        "hf://mteb/banking77",
        map="representax.data.identity",
        revision="18072d2685ea682290f7b8924d94c62acc19c0b2",
        split="train",
    ),
    expected_rows=9_993,
    license="MIT",
)
BANKING77_TEST = CanonicalEvaluationSource(
    name="banking77_test",
    source=BANKING77_TRAIN.source.model_copy(update={"split": "test"}),
    expected_rows=3_076,
    license="MIT",
)
TWENTY_NEWSGROUPS = CanonicalEvaluationSource(
    name="twenty_newsgroups",
    source=source(
        "hf://mteb/twentynewsgroups-clustering",
        map="representax.data.identity",
        revision="6125ec4e24fa026cec8a478383ee943acfbd5449",
        split="test",
    ),
    expected_rows=10,
    license=None,
)
CIFAR100_TRAIN = CanonicalEvaluationSource(
    name="cifar100_train",
    source=source(
        "hf://uoft-cs/cifar100",
        map="representax.data.identity",
        revision="aadb3af77e9048adbea6b47c21a81e47dd092ae5",
        split="train",
    ),
    expected_rows=50_000,
    license=None,
)
CIFAR100_TEST = CanonicalEvaluationSource(
    name="cifar100_test",
    source=CIFAR100_TRAIN.source.model_copy(update={"split": "test"}),
    expected_rows=10_000,
    license=None,
)


class RecordDataset(Protocol):
    """The random-access subset shared by sequences and Grain MapDatasets."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: Any) -> Any: ...


Records = Sequence[Mapping[str, Any]] | RecordDataset


class PackedColumns:
    """Expose one upstream row of equal-length columns as lazy records."""

    def __init__(self, records: Records, *fields: str) -> None:
        if len(records) != 1:
            raise ValueError("packed-column datasets must contain exactly one row")
        row = records[0]
        columns = {field: row[field] for field in fields}
        lengths = {len(value) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError("packed dataset columns must have equal lengths")
        self._columns = columns
        self._length = lengths.pop()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: Any) -> Mapping[str, Any]:
        return {name: values[index] for name, values in self._columns.items()}


def _padded_rows(
    records: Records,
    indices: Sequence[int],
    *,
    batch_size: int,
) -> tuple[tuple[Mapping[str, Any], ...], int]:
    rows = tuple(records[index] for index in indices)
    if not rows:
        raise ValueError("evaluation batches cannot be empty")
    padding = batch_size - len(rows)
    return (*rows, *(rows[-1] for _ in range(padding))), padding


def pair_classification_batches(
    records: Records,
    *,
    processor: Processor,
    batch_size: int,
    left_field: str = "sent1",
    right_field: str = "sent2",
    label_field: str = "labels",
) -> Iterator[Any]:
    """Stream fixed-shape pair-classification batches from original records."""

    if batch_size <= 0:
        raise ValueError("pair-classification batch_size must be positive")
    values: Records = records
    if len(records) == 1 and isinstance(records[0].get(left_field), list):
        values = PackedColumns(records, left_field, right_field, label_field)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        rows, padding = _padded_rows(
            values,
            tuple(range(start, stop)),
            batch_size=batch_size,
        )
        valid_count = batch_size - padding
        yield pair_classification_batch(
            left=processor(tuple(row[left_field] for row in rows)),
            right=processor(tuple(row[right_field] for row in rows)),
            labels=jnp.asarray(tuple(int(row[label_field]) for row in rows)),
            valid=jnp.asarray(
                (True,) * valid_count + (False,) * padding,
                dtype=jnp.bool_,
            ),
        )


def _split_train_validation(
    records: Records,
    *,
    label_field: str,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = tuple(int(records[index][label_field]) for index in range(len(records)))
    train, validation = next(
        StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=seed,
        ).split(range(len(labels)), labels)
    )
    return (
        tuple(int(value) for value in train),
        tuple(int(value) for value in validation),
    )


def labeled_evaluation_batches(
    *,
    train: Records,
    test: Records,
    processor: Processor,
    batch_size: int,
    example_field: str = "text",
    label_field: str = "label",
    validation: Records | None = None,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> Iterator[Any]:
    """Stream deterministic train/validation/test frozen-representation batches."""

    if batch_size <= 0:
        raise ValueError("labeled evaluation batch_size must be positive")
    if validation is None:
        train_indices, validation_indices = _split_train_validation(
            train,
            label_field=label_field,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        splits = (
            (train, train_indices, EvaluationSplit.TRAIN),
            (train, validation_indices, EvaluationSplit.VALIDATION),
            (test, tuple(range(len(test))), EvaluationSplit.TEST),
        )
    else:
        splits = (
            (train, tuple(range(len(train))), EvaluationSplit.TRAIN),
            (
                validation,
                tuple(range(len(validation))),
                EvaluationSplit.VALIDATION,
            ),
            (test, tuple(range(len(test))), EvaluationSplit.TEST),
        )
    for records, indices, split in splits:
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            rows, padding = _padded_rows(
                records,
                selected,
                batch_size=batch_size,
            )
            valid_count = batch_size - padding
            yield labeled_evaluation_batch(
                examples=processor(tuple(row[example_field] for row in rows)),
                labels=jnp.asarray(tuple(int(row[label_field]) for row in rows)),
                split=split,
                valid=jnp.asarray(
                    (True,) * valid_count + (False,) * padding,
                    dtype=jnp.bool_,
                ),
            )


def clustering_samples(
    records: Records,
    *,
    sentence_field: str = "sentences",
    label_field: str = "labels",
) -> Iterator[PackedColumns]:
    """Yield each upstream clustering sample without flattening or copying it."""

    for index in range(len(records)):
        yield PackedColumns((records[index],), sentence_field, label_field)


def clustering_evaluation_batches(
    records: Records,
    *,
    processor: Processor,
    batch_size: int,
    sentence_field: str = "sentences",
    label_field: str = "labels",
) -> Iterator[Any]:
    """Stream one fixed-shape clustering sample into native labeled batches."""

    if batch_size <= 0:
        raise ValueError("clustering batch_size must be positive")
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        rows, padding = _padded_rows(
            records,
            tuple(range(start, stop)),
            batch_size=batch_size,
        )
        valid_count = batch_size - padding
        yield labeled_examples_batch(
            examples=processor(tuple(row[sentence_field] for row in rows)),
            labels=jnp.asarray(tuple(int(row[label_field]) for row in rows)),
            valid=jnp.asarray(
                (True,) * valid_count + (False,) * padding,
                dtype=jnp.bool_,
            ),
        )


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


__all__ = [
    "BANKING77_TEST",
    "BANKING77_TRAIN",
    "CIFAR100_TEST",
    "CIFAR100_TRAIN",
    "SPRINT_DUPLICATE_QUESTIONS",
    "TWENTY_NEWSGROUPS",
    "CanonicalEvaluationSource",
    "PackedColumns",
    "RecordDataset",
    "beir_evaluation",
    "clustering_evaluation_batches",
    "clustering_samples",
    "labeled_evaluation_batches",
    "pair_classification_batches",
]
