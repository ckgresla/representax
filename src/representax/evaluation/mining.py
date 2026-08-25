"""Paraphrase-mining evaluation."""

from __future__ import annotations

import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import Encoder, Route, encode


class MiningEvaluationBatch(eqx.Module):
    inputs: Any
    ids: Int[Array, " batch"]
    valid: Bool[Array, " batch"]


class MiningBatchOutput(eqx.Module):
    embeddings: Float[Array, "batch dimension"]
    ids: Int[Array, " batch"]
    valid: Bool[Array, " batch"]


@dataclass(frozen=True, slots=True)
class _MiningAccumulator:
    embeddings: tuple[np.ndarray, ...] = ()
    ids: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def _normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


@dataclass(frozen=True, slots=True)
class ParaphraseMiningEvaluator:
    """Mine a bounded set of highest-scoring pairs and evaluate duplicate recovery."""

    duplicate_pairs: frozenset[tuple[int, int]]
    name: str = "paraphrase"
    max_pairs: int = 100_000
    block_size: int = 1024
    route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if self.max_pairs <= 0 or self.block_size <= 0:
            raise ValueError("paraphrase bounds must be positive")
        canonical = frozenset(tuple(sorted(pair)) for pair in self.duplicate_pairs)
        object.__setattr__(self, "duplicate_pairs", canonical)

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/average_precision"

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: MiningEvaluationBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> MiningBatchOutput:
        if not isinstance(model, Encoder):
            raise TypeError("paraphrase mining requires an Encoder")
        return MiningBatchOutput(
            embeddings=encode(model, batch.inputs, route=self.route, key=key),
            ids=batch.ids,
            valid=batch.valid,
        )

    def initialize(self) -> _MiningAccumulator:
        return _MiningAccumulator()

    def accumulate(
        self, accumulator: _MiningAccumulator, output: MiningBatchOutput
    ) -> _MiningAccumulator:
        return _MiningAccumulator(
            embeddings=(*accumulator.embeddings, np.asarray(output.embeddings)),
            ids=(*accumulator.ids, np.asarray(output.ids)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _MiningAccumulator) -> Mapping[str, float]:
        if not accumulator.embeddings:
            raise ValueError("paraphrase mining received no batches")
        valid = np.concatenate(accumulator.valid)
        embeddings = _normalize(np.concatenate(accumulator.embeddings)[valid])
        ids = np.concatenate(accumulator.ids)[valid]
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("paraphrase IDs must be unique")
        heap: list[tuple[float, int, int]] = []
        for start in range(0, len(ids), self.block_size):
            stop = min(start + self.block_size, len(ids))
            scores = embeddings[start:stop] @ embeddings.T
            for row in range(stop - start):
                absolute = start + row
                for column in range(absolute + 1, len(ids)):
                    candidate = (float(scores[row, column]), absolute, column)
                    if len(heap) < self.max_pairs:
                        heapq.heappush(heap, candidate)
                    elif candidate[0] > heap[0][0]:
                        heapq.heapreplace(heap, candidate)
        ranked = sorted(heap, reverse=True)
        truth = self.duplicate_pairs
        hits = 0
        precision_sum = 0.0
        best_f1 = 0.0
        best_threshold = float("inf")
        for rank, (score, left, right) in enumerate(ranked, start=1):
            if tuple(sorted((int(ids[left]), int(ids[right])))) in truth:
                hits += 1
                precision_sum += hits / rank
            precision = hits / rank
            recall = hits / max(len(truth), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = score
        prefix = f"valid/{self.name}"
        return {
            f"{prefix}/average_precision": precision_sum / max(len(truth), 1),
            f"{prefix}/best_f1": best_f1,
            f"{prefix}/best_threshold": best_threshold,
            f"{prefix}/recall": hits / max(len(truth), 1),
            f"{prefix}/mined_pairs": float(len(ranked)),
        }


__all__ = [
    "MiningBatchOutput",
    "MiningEvaluationBatch",
    "ParaphraseMiningEvaluator",
]
