"""Serializable task and loss configuration for distillation."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from representax._config import FinitePositiveFloat
from representax.core import Route
from representax.tasks.config import LossConfig, TaskConfig

from .losses import EmbeddingDistillationDistance, ScoreSimilarity


class EmbeddingDistillationConfig(TaskConfig):
    """Student input columns and their semantic encoder routes."""

    kind: Literal["embedding_distillation"] = "embedding_distillation"
    routes: tuple[Route, ...] = (Route.GENERIC,)

    @model_validator(mode="after")
    def validate_routes(self) -> EmbeddingDistillationConfig:
        if not self.routes:
            raise ValueError("embedding distillation requires at least one route")
        return self


class MarginDistillationConfig(TaskConfig):
    """Query-document routes for teacher score-margin regression."""

    kind: Literal["margin_distillation"] = "margin_distillation"
    query_route: Route = Route.QUERY
    document_route: Route = Route.DOCUMENT


class DistributionDistillationConfig(TaskConfig):
    """Query-candidate routes for teacher distribution matching."""

    kind: Literal["distribution_distillation"] = "distribution_distillation"
    query_route: Route = Route.QUERY
    candidate_route: Route = Route.DOCUMENT


class EmbeddingDistillationLossConfig(LossConfig):
    """Offline teacher-embedding matching with one native distance."""

    kind: Literal["embedding_distillation"] = "embedding_distillation"
    distance: EmbeddingDistillationDistance = "cosine"


class MarginMSELossConfig(LossConfig):
    """MSE between student and teacher positive-minus-negative margins."""

    kind: Literal["margin_mse"] = "margin_mse"
    similarity: ScoreSimilarity = "dot"


class DistributionKLLossConfig(LossConfig):
    """Temperature-scaled KL divergence over candidate scores."""

    kind: Literal["distribution_kl"] = "distribution_kl"
    similarity: ScoreSimilarity = "dot"
    temperature: FinitePositiveFloat = 1.0


__all__ = [
    "DistributionDistillationConfig",
    "DistributionKLLossConfig",
    "EmbeddingDistillationConfig",
    "EmbeddingDistillationLossConfig",
    "MarginDistillationConfig",
    "MarginMSELossConfig",
]
