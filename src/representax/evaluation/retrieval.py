"""Streaming information-retrieval evaluation over encoded query and corpus batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import Encoder, Route, encode

RetrievalInputKind = Literal["query", "document"]
RetrievalScoreFunction = Literal["cosine", "dot"]
RETRIEVAL_SCORE_FUNCTIONS: tuple[RetrievalScoreFunction, ...] = ("cosine", "dot")


class RetrievalEvaluationBatch(eqx.Module):
    """One homogeneous, model-ready query or document batch."""

    inputs: Any
    ids: Int[Array, " batch"]
    valid: Bool[Array, " batch"]
    kind: RetrievalInputKind = eqx.field(static=True)


class RetrievalBatchOutput(eqx.Module):
    """Representations and integer artifact IDs emitted by one compiled batch."""

    embeddings: Float[Array, "batch representation"]
    ids: Int[Array, " batch"]
    valid: Bool[Array, " batch"]
    kind: RetrievalInputKind = eqx.field(static=True)


def retrieval_evaluation_batch(
    inputs: Any,
    ids: Int[Array, " batch"],
    *,
    kind: RetrievalInputKind,
    valid: Bool[Array, " batch"] | None = None,
) -> RetrievalEvaluationBatch:
    """Construct one typed retrieval batch after source IDs have been integerized."""

    ids = jnp.asarray(ids)
    if ids.ndim != 1 or not jnp.issubdtype(ids.dtype, jnp.integer):
        raise TypeError("retrieval IDs must be a rank-one integer array")
    if kind not in {"query", "document"}:
        raise ValueError("retrieval batch kind must be 'query' or 'document'")
    resolved_valid = (
        jnp.ones(ids.shape, dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    if resolved_valid.shape != ids.shape:
        raise ValueError("retrieval validity must have one value per ID")
    return RetrievalEvaluationBatch(
        inputs=inputs,
        ids=ids,
        valid=resolved_valid,
        kind=kind,
    )


def _positive_ks(values: Sequence[int], name: str) -> tuple[int, ...]:
    resolved = tuple(values)
    if not resolved or any(value <= 0 for value in resolved):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} values must be unique")
    return resolved


def _dcg(relevance: Sequence[int], k: int) -> float:
    return float(
        sum(value / np.log2(index + 2) for index, value in enumerate(relevance[:k]))
    )


def information_retrieval_metrics(
    ranked_document_ids: np.ndarray,
    query_ids: np.ndarray,
    relevant_documents: Mapping[int, frozenset[int] | set[int]],
    *,
    accuracy_at_k: Sequence[int] = (1, 3, 5, 10),
    precision_recall_at_k: Sequence[int] = (1, 3, 5, 10),
    mrr_at_k: Sequence[int] = (10,),
    ndcg_at_k: Sequence[int] = (10,),
    map_at_k: Sequence[int] = (100,),
) -> dict[str, float]:
    """Compute binary-relevance IR metrics with Sentence Transformers semantics."""

    ranked = np.asarray(ranked_document_ids)
    queries = np.asarray(query_ids)
    if ranked.ndim != 2:
        raise ValueError("ranked document IDs must have shape [query, rank]")
    if queries.shape != (ranked.shape[0],):
        raise ValueError("query IDs must match ranked result rows")
    if not len(queries):
        raise ValueError("information retrieval requires at least one query")
    accuracy_ks = _positive_ks(accuracy_at_k, "accuracy_at_k")
    precision_recall_ks = _positive_ks(
        precision_recall_at_k,
        "precision_recall_at_k",
    )
    mrr_ks = _positive_ks(mrr_at_k, "mrr_at_k")
    ndcg_ks = _positive_ks(ndcg_at_k, "ndcg_at_k")
    map_ks = _positive_ks(map_at_k, "map_at_k")

    totals = {
        **{f"accuracy@{k}": 0.0 for k in accuracy_ks},
        **{f"precision@{k}": 0.0 for k in precision_recall_ks},
        **{f"recall@{k}": 0.0 for k in precision_recall_ks},
        **{f"mrr@{k}": 0.0 for k in mrr_ks},
        **{f"ndcg@{k}": 0.0 for k in ndcg_ks},
        **{f"map@{k}": 0.0 for k in map_ks},
    }
    for query_id, row in zip(queries.tolist(), ranked, strict=True):
        relevant = frozenset(relevant_documents.get(int(query_id), ()))
        if not relevant:
            raise ValueError(f"query {query_id!r} has no relevant documents")
        if len(set(row.tolist())) != len(row):
            raise ValueError(f"query {query_id!r} has duplicate ranked documents")

        for k in accuracy_ks:
            totals[f"accuracy@{k}"] += float(
                any(int(document_id) in relevant for document_id in row[:k])
            )
        for k in precision_recall_ks:
            correct = sum(int(document_id) in relevant for document_id in row[:k])
            totals[f"precision@{k}"] += correct / k
            totals[f"recall@{k}"] += correct / len(relevant)
        for k in mrr_ks:
            totals[f"mrr@{k}"] += next(
                (
                    1.0 / rank
                    for rank, document_id in enumerate(row[:k], start=1)
                    if int(document_id) in relevant
                ),
                0.0,
            )
        for k in ndcg_ks:
            predicted = [int(document_id) in relevant for document_id in row[:k]]
            totals[f"ndcg@{k}"] += _dcg(predicted, k) / _dcg(
                [1] * len(relevant),
                k,
            )
        for k in map_ks:
            correct = 0
            precision_sum = 0.0
            for rank, document_id in enumerate(row[:k], start=1):
                if int(document_id) in relevant:
                    correct += 1
                    precision_sum += correct / rank
            totals[f"map@{k}"] += precision_sum / min(k, len(relevant))
    return {name: value / len(queries) for name, value in totals.items()}


def _scores(
    queries: np.ndarray,
    documents: np.ndarray,
    function: RetrievalScoreFunction,
) -> np.ndarray:
    if function == "dot":
        return queries @ documents.T
    if function == "cosine":
        query_norm = np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
        document_norm = np.maximum(
            np.linalg.norm(documents, axis=1, keepdims=True),
            1e-12,
        )
        return (queries / query_norm) @ (documents / document_norm).T
    raise ValueError(f"unsupported retrieval score function {function!r}")


def _top_k(
    scores: np.ndarray,
    document_ids: np.ndarray,
    *,
    k: int,
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.broadcast_to(document_ids[None, :], scores.shape)
    if previous is not None:
        scores = np.concatenate((previous[0], scores), axis=1)
        ids = np.concatenate((previous[1], ids), axis=1)
    width = min(k, scores.shape[1])
    if width < scores.shape[1]:
        indices = np.argpartition(-scores, width - 1, axis=1)[:, :width]
        scores = np.take_along_axis(scores, indices, axis=1)
        ids = np.take_along_axis(ids, indices, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return (
        np.take_along_axis(scores, order, axis=1),
        np.take_along_axis(ids, order, axis=1),
    )


@dataclass(frozen=True, slots=True)
class _RetrievalAccumulator:
    query_embeddings: tuple[np.ndarray, ...] = ()
    query_ids: tuple[np.ndarray, ...] = ()
    queries: np.ndarray | None = None
    resolved_query_ids: np.ndarray | None = None
    rankings: tuple[tuple[RetrievalScoreFunction, np.ndarray, np.ndarray], ...] = ()
    document_ids: frozenset[int] = frozenset()
    corpus_started: bool = False


@dataclass(frozen=True, slots=True)
class InformationRetrievalEvaluator:
    """Stream corpus batches through bounded top-k state after encoding queries."""

    relevant_documents: Mapping[int, frozenset[int] | set[int]]
    name: str = "retrieval"
    score_functions: tuple[RetrievalScoreFunction, ...] = ("cosine",)
    main_score_function: RetrievalScoreFunction = "cosine"
    accuracy_at_k: tuple[int, ...] = (1, 3, 5, 10)
    precision_recall_at_k: tuple[int, ...] = (1, 3, 5, 10)
    mrr_at_k: tuple[int, ...] = (10,)
    ndcg_at_k: tuple[int, ...] = (10,)
    map_at_k: tuple[int, ...] = (100,)
    query_route: Route = Route.QUERY
    document_route: Route = Route.DOCUMENT

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.score_functions or len(set(self.score_functions)) != len(
            self.score_functions
        ):
            raise ValueError("retrieval score functions must be non-empty and unique")
        invalid = set(self.score_functions) - set(RETRIEVAL_SCORE_FUNCTIONS)
        if invalid:
            raise ValueError(f"unsupported retrieval score functions: {invalid}")
        if self.main_score_function not in self.score_functions:
            raise ValueError("main_score_function must be one of score_functions")
        for values, name in (
            (self.accuracy_at_k, "accuracy_at_k"),
            (self.precision_recall_at_k, "precision_recall_at_k"),
            (self.mrr_at_k, "mrr_at_k"),
            (self.ndcg_at_k, "ndcg_at_k"),
            (self.map_at_k, "map_at_k"),
        ):
            _positive_ks(values, name)
        if not self.relevant_documents:
            raise ValueError("retrieval evaluation requires relevance judgments")
        if any(not documents for documents in self.relevant_documents.values()):
            raise ValueError("every evaluated query requires a relevant document")

    @property
    def primary_metric(self) -> str:
        return (
            f"valid/{self.name}/{self.main_score_function}_ndcg@{max(self.ndcg_at_k)}"
        )

    @property
    def _maximum_k(self) -> int:
        return max(
            *self.accuracy_at_k,
            *self.precision_recall_at_k,
            *self.mrr_at_k,
            *self.ndcg_at_k,
            *self.map_at_k,
        )

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: RetrievalEvaluationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> RetrievalBatchOutput:
        if not isinstance(batch, RetrievalEvaluationBatch):
            raise TypeError("information retrieval requires RetrievalEvaluationBatch")
        if not isinstance(model, Encoder):
            raise TypeError("information retrieval requires an Encoder")
        route = self.query_route if batch.kind == "query" else self.document_route
        return RetrievalBatchOutput(
            embeddings=encode(model, batch.inputs, route=route, key=key),
            ids=batch.ids,
            valid=batch.valid,
            kind=batch.kind,
        )

    def initialize(self) -> _RetrievalAccumulator:
        return _RetrievalAccumulator()

    def accumulate(
        self,
        accumulator: _RetrievalAccumulator,
        output: RetrievalBatchOutput,
    ) -> _RetrievalAccumulator:
        valid = np.asarray(output.valid, dtype=bool)
        embeddings = np.asarray(output.embeddings)[valid]
        ids = np.asarray(output.ids)[valid]
        if not np.all(np.isfinite(embeddings)):
            raise ValueError("retrieval embeddings must be finite")
        if output.kind == "query":
            if accumulator.corpus_started:
                raise ValueError(
                    "all retrieval query batches must precede corpus batches"
                )
            return _RetrievalAccumulator(
                query_embeddings=(*accumulator.query_embeddings, embeddings),
                query_ids=(*accumulator.query_ids, ids),
            )
        if not accumulator.query_embeddings and accumulator.queries is None:
            raise ValueError("retrieval corpus batches require preceding query batches")
        if not len(ids):
            return accumulator
        incoming_ids = frozenset(int(value) for value in ids)
        if len(incoming_ids) != len(ids) or incoming_ids & accumulator.document_ids:
            raise ValueError("retrieval document IDs must be globally unique")

        if accumulator.queries is None or accumulator.resolved_query_ids is None:
            queries = np.concatenate(accumulator.query_embeddings)
            query_ids = np.concatenate(accumulator.query_ids)
            if len(set(query_ids.tolist())) != len(query_ids):
                raise ValueError("retrieval query IDs must be globally unique")
            active = np.asarray(
                [
                    bool(self.relevant_documents.get(int(query_id), ()))
                    for query_id in query_ids
                ],
                dtype=bool,
            )
            queries = queries[active]
            query_ids = query_ids[active]
            if not len(query_ids):
                raise ValueError("no encoded queries have relevance judgments")
        else:
            queries = accumulator.queries
            query_ids = accumulator.resolved_query_ids

        previous = {
            function: (scores, ranked_ids)
            for function, scores, ranked_ids in accumulator.rankings
        }
        rankings = tuple(
            (
                function,
                *_top_k(
                    _scores(queries, embeddings, function),
                    ids,
                    k=self._maximum_k,
                    previous=previous.get(function),
                ),
            )
            for function in self.score_functions
        )
        return _RetrievalAccumulator(
            queries=queries,
            resolved_query_ids=query_ids,
            rankings=rankings,
            document_ids=accumulator.document_ids | incoming_ids,
            corpus_started=True,
        )

    def finalize(self, accumulator: _RetrievalAccumulator) -> Mapping[str, float]:
        if not accumulator.rankings or accumulator.resolved_query_ids is None:
            raise ValueError("retrieval evaluation received no corpus batches")
        metrics: dict[str, float] = {}
        for function, _, ranked_ids in accumulator.rankings:
            values = information_retrieval_metrics(
                ranked_ids,
                accumulator.resolved_query_ids,
                self.relevant_documents,
                accuracy_at_k=self.accuracy_at_k,
                precision_recall_at_k=self.precision_recall_at_k,
                mrr_at_k=self.mrr_at_k,
                ndcg_at_k=self.ndcg_at_k,
                map_at_k=self.map_at_k,
            )
            metrics.update(
                {
                    f"valid/{self.name}/{function}_{name}": value
                    for name, value in values.items()
                }
            )
        return metrics


__all__ = [
    "InformationRetrievalEvaluator",
    "RETRIEVAL_SCORE_FUNCTIONS",
    "RetrievalBatchOutput",
    "RetrievalEvaluationBatch",
    "RetrievalInputKind",
    "RetrievalScoreFunction",
    "information_retrieval_metrics",
    "retrieval_evaluation_batch",
]
