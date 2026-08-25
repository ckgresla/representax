"""Explicit-triplet embedding evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.core import Encoder, Route, encode
from representax.tasks.triplet import ExplicitTripletBatch

TripletDistance = Literal["cosine", "euclidean", "manhattan"]


class TripletBatchOutput(eqx.Module):
    anchor: Float[Array, "batch dimension"]
    positive: Float[Array, "batch dimension"]
    negative: Float[Array, "batch dimension"]
    valid: Bool[Array, " batch"]


@dataclass(frozen=True, slots=True)
class _TripletAccumulator:
    anchor: tuple[np.ndarray, ...] = ()
    positive: tuple[np.ndarray, ...] = ()
    negative: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


def _distance(left: np.ndarray, right: np.ndarray, kind: TripletDistance) -> np.ndarray:
    if kind == "euclidean":
        return np.linalg.norm(left - right, axis=-1)
    if kind == "manhattan":
        return np.sum(np.abs(left - right), axis=-1)
    if kind == "cosine":
        denominator = np.maximum(
            np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1), 1e-12
        )
        return 1.0 - np.sum(left * right, axis=-1) / denominator
    raise ValueError(f"unsupported triplet distance {kind!r}")


@dataclass(frozen=True, slots=True)
class TripletEvaluator:
    name: str = "triplet"
    distance: TripletDistance = "cosine"
    anchor_route: Route = Route.GENERIC
    positive_route: Route = Route.GENERIC
    negative_route: Route = Route.GENERIC

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/accuracy"

    def evaluate_batch(
        self,
        model: eqx.Module,
        batch: ExplicitTripletBatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> TripletBatchOutput:
        if not isinstance(batch, ExplicitTripletBatch) or not isinstance(
            model, Encoder
        ):
            raise TypeError("triplet evaluation requires an Encoder and triplet batch")
        keys = (None, None, None) if key is None else jax.random.split(key, 3)
        return TripletBatchOutput(
            anchor=encode(model, batch.anchor, route=self.anchor_route, key=keys[0]),
            positive=encode(
                model, batch.positive, route=self.positive_route, key=keys[1]
            ),
            negative=encode(
                model, batch.negative, route=self.negative_route, key=keys[2]
            ),
            valid=batch.valid,
        )

    def initialize(self) -> _TripletAccumulator:
        return _TripletAccumulator()

    def accumulate(
        self, accumulator: _TripletAccumulator, output: TripletBatchOutput
    ) -> _TripletAccumulator:
        return _TripletAccumulator(
            anchor=(*accumulator.anchor, np.asarray(output.anchor)),
            positive=(*accumulator.positive, np.asarray(output.positive)),
            negative=(*accumulator.negative, np.asarray(output.negative)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _TripletAccumulator) -> Mapping[str, float]:
        if not accumulator.anchor:
            raise ValueError("triplet evaluation received no batches")
        valid = np.concatenate(accumulator.valid)
        anchor = np.concatenate(accumulator.anchor)[valid]
        positive = np.concatenate(accumulator.positive)[valid]
        negative = np.concatenate(accumulator.negative)[valid]
        positive_distance = _distance(anchor, positive, self.distance)
        negative_distance = _distance(anchor, negative, self.distance)
        prefix = f"valid/{self.name}"
        return {
            f"{prefix}/accuracy": float(np.mean(positive_distance < negative_distance)),
            f"{prefix}/positive_distance": float(np.mean(positive_distance)),
            f"{prefix}/negative_distance": float(np.mean(negative_distance)),
            f"{prefix}/distance_margin": float(
                np.mean(negative_distance - positive_distance)
            ),
        }


__all__ = ["TripletBatchOutput", "TripletDistance", "TripletEvaluator"]
