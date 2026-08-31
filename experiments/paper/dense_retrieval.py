"""Experiment-specific adapters for the paired dense-retrieval job."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from representax.core import Route
from representax.evaluation import retrieval_evaluation_batch


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationRow:
    """One ordered query or document consumed by configured evaluation."""

    identifier: int
    text: str
    kind: str
    valid: bool = True


class RetrievalEvaluationCollator:
    """Build homogeneous query or document batches with model preprocessing."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, rows: Sequence[RetrievalEvaluationRow]) -> Any:
        kinds = {row.kind for row in rows}
        if len(kinds) != 1:
            raise ValueError("retrieval evaluation batches must be homogeneous")
        kind = kinds.pop()
        if kind not in {"query", "document"}:
            raise ValueError(f"unknown retrieval evaluation kind {kind!r}")
        route = Route.QUERY if kind == "query" else Route.DOCUMENT
        return retrieval_evaluation_batch(
            self.processor(tuple(row.text for row in rows), route=route),
            jnp.asarray(tuple(row.identifier for row in rows), dtype=jnp.int32),
            kind=kind,
            valid=jnp.asarray(tuple(row.valid for row in rows), dtype=jnp.bool_),
        )


def evaluation_rows(
    queries: Iterable[tuple[int, str]],
    documents: Iterable[tuple[int, str]],
    *,
    batch_size: int,
) -> tuple[RetrievalEvaluationRow, ...]:
    """Order and pad queries before documents for the streaming IR reducer."""

    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    output: list[RetrievalEvaluationRow] = []
    for kind, values in (("query", tuple(queries)), ("document", tuple(documents))):
        output.extend(
            RetrievalEvaluationRow(identifier=identifier, text=text, kind=kind)
            for identifier, text in values
        )
        remainder = len(values) % batch_size
        if remainder:
            output.extend(
                RetrievalEvaluationRow(
                    identifier=-1,
                    text="",
                    kind=kind,
                    valid=False,
                )
                for _ in range(batch_size - remainder)
            )
    return tuple(output)


def fixed_rows_resolver(rows: Sequence[Any]) -> Any:
    """Return an artifact resolver for one immutable in-process row sequence."""

    frozen = tuple(rows)

    def resolve(_config: Any) -> tuple[Any, ...]:
        return frozen

    return resolve


def event_metrics(path: Any, event: str) -> Mapping[str, float]:
    """Read the most recent matching row from the canonical metric stream."""

    import json

    values = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("event") == event:
                values = {
                    name: float(value) for name, value in record["metrics"].items()
                }
    if values is None:
        raise ValueError(f"run did not emit canonical {event!r} metrics")
    return values


__all__ = [
    "RetrievalEvaluationCollator",
    "RetrievalEvaluationRow",
    "evaluation_rows",
    "event_metrics",
    "fixed_rows_resolver",
]
