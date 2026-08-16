"""Native embedding, score-margin, and distribution distillation losses."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

EmbeddingDistillationDistance = Literal["mse", "l2", "cosine"]
ScoreSimilarity = Literal["dot", "cosine"]


class EmbeddingDistillationLossTerms(eqx.Module):
    """Per-column, per-row embedding matching terms."""

    loss: Float[Array, ""]
    row_losses: Float[Array, "column batch"]
    selected: Bool[Array, "column batch"]


class MarginDistillationLossTerms(eqx.Module):
    """Student and teacher positive-minus-negative score margins."""

    loss: Float[Array, ""]
    predicted_margins: Float[Array, "batch negative"]
    teacher_margins: Float[Array, "batch negative"]
    row_losses: Float[Array, "batch negative"]
    selected: Bool[Array, "batch negative"]


class DistributionDistillationLossTerms(eqx.Module):
    """Temperature-scaled student and teacher candidate distributions."""

    loss: Float[Array, ""]
    student_scores: Float[Array, "batch candidate"]
    teacher_probabilities: Float[Array, "batch candidate"]
    row_losses: Float[Array, " batch"]
    selected: Bool[Array, " batch"]


def _valid(count: int, valid: Bool[Array, " batch"] | None) -> Bool[Array, " batch"]:
    if valid is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    resolved = jnp.asarray(valid)
    if resolved.shape != (count,) or resolved.dtype != jnp.bool_:
        raise TypeError("valid must be a boolean vector matching batch rows")
    return resolved


def _masked_mean(
    values: Float[Array, " *row"],
    selected: Bool[Array, " *row"],
) -> Float[Array, ""]:
    count = jnp.sum(selected)
    return jnp.sum(jnp.where(selected, values, 0.0)) / jnp.maximum(count, 1).astype(
        jnp.float32
    )


def _embedding_columns(
    values: Float[Array, "column batch representation"],
    *,
    name: str,
) -> Float[Array, "column batch representation"]:
    resolved = jnp.asarray(values, dtype=jnp.float32)
    if resolved.ndim != 3:
        raise ValueError(f"{name} must have shape [column, batch, dimension]")
    return resolved


def embedding_distillation_loss_terms(
    student_embeddings: Float[Array, "column batch representation"],
    teacher_embeddings: Float[Array, "column batch representation"],
    *,
    valid: Bool[Array, " batch"] | None = None,
    distance: EmbeddingDistillationDistance = "cosine",
) -> EmbeddingDistillationLossTerms:
    """Match each student embedding column to its offline teacher target."""

    students = _embedding_columns(student_embeddings, name="student_embeddings")
    teachers = _embedding_columns(teacher_embeddings, name="teacher_embeddings")
    if students.shape != teachers.shape:
        raise ValueError(
            "student and teacher embeddings must have identical projected shapes"
        )
    resolved_valid = _valid(students.shape[1], valid)
    if distance == "mse":
        row_losses = jnp.mean(jnp.square(students - teachers), axis=2)
    elif distance == "l2":
        row_losses = jnp.linalg.norm(students - teachers, axis=2)
    elif distance == "cosine":
        student_norms = jnp.maximum(jnp.linalg.norm(students, axis=2), 1e-8)
        teacher_norms = jnp.maximum(jnp.linalg.norm(teachers, axis=2), 1e-8)
        row_losses = 1.0 - jnp.sum(students * teachers, axis=2) / (
            student_norms * teacher_norms
        )
    else:
        raise ValueError(f"unsupported embedding distillation distance {distance!r}")
    selected = jnp.broadcast_to(resolved_valid[None, :], row_losses.shape)
    return EmbeddingDistillationLossTerms(
        loss=_masked_mean(row_losses, selected),
        row_losses=row_losses,
        selected=selected,
    )


def aligned_score_similarity(
    left: Float[Array, "batch representation"],
    right: Float[Array, "batch representation"],
    *,
    similarity: ScoreSimilarity,
) -> Float[Array, " batch"]:
    """Dot or cosine similarity between aligned representation rows."""

    left = jnp.asarray(left, dtype=jnp.float32)
    right = jnp.asarray(right, dtype=jnp.float32)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("aligned score embeddings must be equal-shaped matrices")
    products = jnp.sum(left * right, axis=1)
    if similarity == "dot":
        return products
    if similarity == "cosine":
        left_norms = jnp.maximum(jnp.linalg.norm(left, axis=1), 1e-8)
        right_norms = jnp.maximum(jnp.linalg.norm(right, axis=1), 1e-8)
        return products / (left_norms * right_norms)
    raise ValueError(f"unsupported score similarity {similarity!r}")


def _candidate_scores(
    queries: Float[Array, "batch representation"],
    candidates: Float[Array, "candidate batch representation"],
    *,
    similarity: ScoreSimilarity,
) -> Float[Array, "batch candidate"]:
    candidates = jnp.asarray(candidates, dtype=jnp.float32)
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape [candidate, batch, dimension]")
    scores = [
        aligned_score_similarity(queries, candidate, similarity=similarity)
        for candidate in candidates
    ]
    return jnp.stack(scores, axis=1)


def margin_mse_loss_terms(
    queries: Float[Array, "batch representation"],
    positives: Float[Array, "batch representation"],
    negatives: Float[Array, "negative batch representation"],
    teacher_margins: Float[Array, "batch negative"],
    *,
    valid: Bool[Array, " batch"] | None = None,
    similarity: ScoreSimilarity = "dot",
) -> MarginDistillationLossTerms:
    """Regress student positive-minus-negative scores onto teacher margins."""

    queries = jnp.asarray(queries, dtype=jnp.float32)
    positives = jnp.asarray(positives, dtype=jnp.float32)
    negative_scores = _candidate_scores(queries, negatives, similarity=similarity)
    positive_scores = aligned_score_similarity(
        queries,
        positives,
        similarity=similarity,
    )
    predicted = positive_scores[:, None] - negative_scores
    targets = jnp.asarray(teacher_margins, dtype=jnp.float32)
    if targets.shape != predicted.shape:
        raise ValueError("teacher margins must match batch and negative dimensions")
    row_losses = jnp.square(predicted - targets)
    resolved_valid = _valid(predicted.shape[0], valid)
    selected = jnp.broadcast_to(resolved_valid[:, None], predicted.shape)
    return MarginDistillationLossTerms(
        loss=_masked_mean(row_losses, selected),
        predicted_margins=predicted,
        teacher_margins=targets,
        row_losses=row_losses,
        selected=selected,
    )


def distribution_kl_loss_terms(
    queries: Float[Array, "batch representation"],
    candidates: Float[Array, "candidate batch representation"],
    teacher_scores: Float[Array, "batch candidate"],
    *,
    valid: Bool[Array, " batch"] | None = None,
    similarity: ScoreSimilarity = "dot",
    temperature: float = 1.0,
) -> DistributionDistillationLossTerms:
    """Distill a teacher candidate distribution with temperature-scaled KL."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("distillation temperature must be finite and positive")
    student_scores = _candidate_scores(queries, candidates, similarity=similarity)
    targets = jnp.asarray(teacher_scores, dtype=jnp.float32)
    if targets.shape != student_scores.shape:
        raise ValueError("teacher scores must match batch and candidate dimensions")
    student_log_probabilities = jax.nn.log_softmax(
        student_scores / temperature,
        axis=1,
    )
    teacher_log_probabilities = jax.nn.log_softmax(targets / temperature, axis=1)
    teacher_probabilities = jnp.exp(teacher_log_probabilities)
    row_losses = jnp.sum(
        teacher_probabilities * (teacher_log_probabilities - student_log_probabilities),
        axis=1,
    ) * (temperature**2)
    selected = _valid(student_scores.shape[0], valid)
    return DistributionDistillationLossTerms(
        loss=_masked_mean(row_losses, selected),
        student_scores=student_scores,
        teacher_probabilities=teacher_probabilities,
        row_losses=row_losses,
        selected=selected,
    )


__all__ = [
    "DistributionDistillationLossTerms",
    "EmbeddingDistillationDistance",
    "EmbeddingDistillationLossTerms",
    "MarginDistillationLossTerms",
    "ScoreSimilarity",
    "aligned_score_similarity",
    "distribution_kl_loss_terms",
    "embedding_distillation_loss_terms",
    "margin_mse_loss_terms",
]
