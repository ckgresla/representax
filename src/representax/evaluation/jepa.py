"""LeJEPA validation and collapse diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from representax.tasks.jepa import JEPABatch, LeJEPATask


class JEPABatchOutput(eqx.Module):
    projections: Float[Array, "batch view dimension"]
    valid: Bool[Array, "batch view"]


@dataclass(frozen=True, slots=True)
class _JEPAAccumulator:
    projections: tuple[np.ndarray, ...] = ()
    valid: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True, slots=True)
class JEPAEvaluator:
    """Measure view invariance and representation collapse without training noise."""

    task: LeJEPATask
    name: str = "jepa"

    @property
    def primary_metric(self) -> str:
        return f"valid/{self.name}/invariance"

    def evaluate_batch(
        self,
        model: Any,
        batch: JEPABatch,
        *,
        key: PRNGKeyArray | None = None,
    ) -> JEPABatchOutput:
        return JEPABatchOutput(
            projections=self.task.representations(model, batch, key=key),
            valid=batch.valid,
        )

    def initialize(self) -> _JEPAAccumulator:
        return _JEPAAccumulator()

    def accumulate(
        self, accumulator: _JEPAAccumulator, output: JEPABatchOutput
    ) -> _JEPAAccumulator:
        return _JEPAAccumulator(
            projections=(*accumulator.projections, np.asarray(output.projections)),
            valid=(*accumulator.valid, np.asarray(output.valid, dtype=bool)),
        )

    def finalize(self, accumulator: _JEPAAccumulator) -> Mapping[str, float]:
        if not accumulator.projections:
            raise ValueError("JEPA evaluation received no batches")
        projections = np.concatenate(accumulator.projections)
        valid = np.concatenate(accumulator.valid)
        sample_means = np.sum(projections * valid[..., None], axis=1) / np.maximum(
            np.sum(valid, axis=1, keepdims=True), 1
        )
        centered_views = projections - sample_means[:, None, :]
        invariance = np.sum(np.square(centered_views) * valid[..., None]) / max(
            int(np.sum(valid)) * projections.shape[-1], 1
        )
        flat = projections[valid]
        feature_std = np.std(flat, axis=0)
        centered = flat - np.mean(flat, axis=0, keepdims=True)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        off_diagonal = covariance - np.diag(np.diag(covariance))
        singular = np.linalg.svd(centered, compute_uv=False)
        probability = singular / max(float(np.sum(singular)), 1e-12)
        effective_rank = float(
            np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        )
        prefix = f"valid/{self.name}"
        return {
            f"{prefix}/invariance": float(invariance),
            f"{prefix}/feature_std_mean": float(np.mean(feature_std)),
            f"{prefix}/feature_std_min": float(np.min(feature_std)),
            f"{prefix}/covariance_off_diagonal_rms": float(
                np.sqrt(np.mean(np.square(off_diagonal)))
            ),
            f"{prefix}/effective_rank": effective_rank,
        }


__all__ = ["JEPABatchOutput", "JEPAEvaluator"]
