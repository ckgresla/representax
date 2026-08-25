"""Paired similarity evaluation compatible with Sentence Transformers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Encoder, Route, encode
from representax.tasks.pairwise import PairwiseBatch

SimilarityFunction = Literal["cosine", "dot", "euclidean", "manhattan"]
SIMILARITY_FUNCTIONS: tuple[SimilarityFunction, ...] = (
    "cosine",
    "dot",
    "euclidean",
    "manhattan",
)


class SimilarityBatchOutput(eqx.Module):
    """Aligned representations and labels emitted by one compiled batch."""

    left: Float[Array, "pair representation"]
    right: Float[Array, "pair representation"]
    labels: Float[Array, " pair"]
    valid: Bool[Array, " pair"]


@dataclass(frozen=True, slots=True)
class _SimilarityAccumulator:
    left: tuple[np.ndarray, ...] = ()
    right: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Assign one-based average ranks, matching SciPy's default tie policy."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _pairwise_scores(
    left: np.ndarray,
    right: np.ndarray,
    function: SimilarityFunction,
) -> np.ndarray:
    if function == "cosine":
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    if function == "dot":
        return np.sum(left * right, axis=1)
    if function == "euclidean":
        return -np.linalg.norm(left - right, axis=1)
    if function == "manhattan":
        return -np.sum(np.abs(left - right), axis=1)
    raise ValueError(f"unsupported similarity function {function!r}")


def similarity_metrics(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    similarity_functions: Sequence[SimilarityFunction] = SIMILARITY_FUNCTIONS,
) -> dict[str, float]:
    """Compute paired Pearson/Spearman metrics on host representations."""

    left = np.asarray(left)
    right = np.asarray(right)
    labels = np.asarray(labels)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("paired embeddings must share shape [pair, representation]")
    if labels.shape != (left.shape[0],):
        raise ValueError("labels must have one scalar per embedding pair")
    if left.shape[0] < 2:
        raise ValueError("embedding similarity requires at least two valid pairs")
    if not (
        np.all(np.isfinite(left))
        and np.all(np.isfinite(right))
        and np.all(np.isfinite(labels))
    ):
        raise ValueError("embedding similarity inputs must be finite")
    functions = tuple(similarity_functions)
    if not functions:
        raise ValueError("at least one similarity function is required")
    if len(set(functions)) != len(functions):
        raise ValueError("similarity functions must be unique")
    invalid = tuple(
        function for function in functions if function not in SIMILARITY_FUNCTIONS
    )
    if invalid:
        raise ValueError(f"unsupported similarity functions: {invalid}")

    metrics: dict[str, float] = {}
    for function in functions:
        scores = _pairwise_scores(left, right, function)
        metrics[f"pearson_{function}"] = _pearson(labels, scores)
        metrics[f"spearman_{function}"] = _pearson(
            _rankdata(labels),
            _rankdata(scores),
        )
    if len(functions) > 1:
        metrics["pearson_max"] = max(
            metrics[f"pearson_{function}"] for function in functions
        )
        metrics["spearman_max"] = max(
            metrics[f"spearman_{function}"] for function in functions
        )
    return metrics


@dataclass(frozen=True, slots=True)
class SimilarityEvaluator:
    """Corpus-level correlations for aligned labeled embedding pairs."""

    name: str = "similarity"
    similarity_functions: tuple[SimilarityFunction, ...] = SIMILARITY_FUNCTIONS
    main_similarity: SimilarityFunction | None = None
    left_route: Route = Route.GENERIC
    right_route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if not self.similarity_functions:
            raise ValueError("at least one similarity function is required")
        if len(set(self.similarity_functions)) != len(self.similarity_functions):
            raise ValueError("similarity functions must be unique")
        invalid = tuple(
            function
            for function in self.similarity_functions
            if function not in SIMILARITY_FUNCTIONS
        )
        if invalid:
            raise ValueError(f"unsupported similarity functions: {invalid}")
        if (
            self.main_similarity is not None
            and self.main_similarity not in self.similarity_functions
        ):
            raise ValueError("main_similarity must be one of similarity_functions")

    @property
    def primary_metric(self) -> str:
        suffix = (
            f"spearman_{self.main_similarity}"
            if self.main_similarity is not None
            else (
                "spearman_max"
                if len(self.similarity_functions) > 1
                else f"spearman_{self.similarity_functions[0]}"
            )
        )
        return f"valid/{self.name}/{suffix}"

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: PairwiseBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> SimilarityBatchOutput:
        if not isinstance(batch, PairwiseBatch):
            raise TypeError("embedding similarity requires a PairwiseBatch")
        if not isinstance(model, Encoder):
            raise TypeError("embedding similarity requires an Encoder")
        if key is None:
            left_key = right_key = None
        else:
            left_key, right_key = jax.random.split(key)
        return SimilarityBatchOutput(
            left=encode(model, batch.left, route=self.left_route, key=left_key),
            right=encode(model, batch.right, route=self.right_route, key=right_key),
            labels=batch.labels,
            valid=batch.valid,
        )

    def initialize(self) -> _SimilarityAccumulator:
        return _SimilarityAccumulator()

    def accumulate(
        self,
        accumulator: _SimilarityAccumulator,
        output: SimilarityBatchOutput,
    ) -> _SimilarityAccumulator:
        return _SimilarityAccumulator(
            left=(*accumulator.left, np.asarray(output.left)),
            right=(*accumulator.right, np.asarray(output.right)),
            labels=(*accumulator.labels, np.asarray(output.labels)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(
        self,
        accumulator: _SimilarityAccumulator,
    ) -> Mapping[str, float]:
        if not accumulator.left:
            raise ValueError("similarity evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        metrics = similarity_metrics(
            np.concatenate(accumulator.left)[valid],
            np.concatenate(accumulator.right)[valid],
            np.concatenate(accumulator.labels)[valid],
            similarity_functions=self.similarity_functions,
        )
        return {
            f"valid/{self.name}/{metric}": value for metric, value in metrics.items()
        }


__all__ = [
    "SimilarityBatchOutput",
    "SimilarityEvaluator",
    "SIMILARITY_FUNCTIONS",
    "SimilarityFunction",
    "similarity_metrics",
]
