"""Deterministic frozen-embedding clustering evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import Encoder, Route, encode
from representax.tasks.triplet import LabeledExamplesBatch

from .representation import normalize_embeddings


class ClusteringBatchOutput(eqx.Module):
    embeddings: Float[Array, "example representation"]
    labels: Int[Array, " example"]
    valid: Bool[Array, " example"]


@dataclass(frozen=True, slots=True)
class _ClusteringAccumulator:
    embeddings: tuple[np.ndarray, ...] = ()
    labels: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    clusters: int | None = None,
    normalization: Literal["none", "l2"] = "l2",
    batch_size: int = 1024,
    max_iterations: int = 100,
    n_init: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Fit MiniBatchKMeans and report label-permutation-invariant geometry."""

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import (
            adjusted_rand_score,
            completeness_score,
            homogeneity_score,
            normalized_mutual_info_score,
            v_measure_score,
        )
    except ImportError as error:  # pragma: no cover - exercised by package smoke
        raise ImportError(
            "clustering evaluation requires the mandatory scikit-learn dependency; "
            "reinstall Representax"
        ) from error
    values = normalize_embeddings(embeddings, normalization)
    labels = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or labels.shape != (len(values),) or not len(values):
        raise ValueError("clustering requires aligned non-empty embeddings and labels")
    cluster_count = len(np.unique(labels)) if clusters is None else clusters
    if cluster_count <= 1 or cluster_count > len(values):
        raise ValueError("cluster count must be between two and the example count")
    predicted = MiniBatchKMeans(
        n_clusters=cluster_count,
        batch_size=batch_size,
        max_iter=max_iterations,
        n_init=n_init,
        random_state=seed,
    ).fit_predict(values)
    return {
        "v_measure": float(v_measure_score(labels, predicted)),
        "homogeneity": float(homogeneity_score(labels, predicted)),
        "completeness": float(completeness_score(labels, predicted)),
        "adjusted_rand": float(adjusted_rand_score(labels, predicted)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(labels, predicted)
        ),
    }


@dataclass(frozen=True, slots=True)
class ClusteringEvaluator:
    """Evaluate unsupervised geometry of frozen encoder representations."""

    name: str = "clustering"
    clusters: int | None = None
    normalization: Literal["none", "l2"] = "l2"
    batch_size: int = 1024
    max_iterations: int = 100
    n_init: int = 10
    seed: int = 0
    route: Route = Route.GENERIC

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation name must be non-empty")
        if self.clusters is not None and self.clusters <= 1:
            raise ValueError("clusters must exceed one or be inferred")
        if min(self.batch_size, self.max_iterations, self.n_init) <= 0:
            raise ValueError("clustering execution values must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/v_measure"

    def evaluate_batch(
        self,
        model: Any,
        batch: LabeledExamplesBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> ClusteringBatchOutput:
        if not isinstance(batch, LabeledExamplesBatch):
            raise TypeError("clustering requires a LabeledExamplesBatch")
        if not isinstance(model, Encoder):
            raise TypeError("clustering requires an Encoder")
        return ClusteringBatchOutput(
            embeddings=encode(model, batch.examples, route=self.route, key=key),
            labels=batch.labels,
            valid=batch.valid,
        )

    def initialize(self) -> _ClusteringAccumulator:
        return _ClusteringAccumulator()

    def accumulate(
        self,
        accumulator: _ClusteringAccumulator,
        output: ClusteringBatchOutput,
    ) -> _ClusteringAccumulator:
        return _ClusteringAccumulator(
            embeddings=(*accumulator.embeddings, np.asarray(output.embeddings)),
            labels=(*accumulator.labels, np.asarray(output.labels)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _ClusteringAccumulator) -> Mapping[str, float]:
        if not accumulator.embeddings:
            raise ValueError("clustering evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        metrics = clustering_metrics(
            np.concatenate(accumulator.embeddings)[valid],
            np.concatenate(accumulator.labels)[valid],
            clusters=self.clusters,
            normalization=self.normalization,
            batch_size=self.batch_size,
            max_iterations=self.max_iterations,
            n_init=self.n_init,
            seed=self.seed,
        )
        return {f"valid/{self.name}/{name}": value for name, value in metrics.items()}


__all__ = ["ClusteringBatchOutput", "ClusteringEvaluator", "clustering_metrics"]
