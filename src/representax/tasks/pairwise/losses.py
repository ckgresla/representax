"""Native labeled-pair similarity, distance, and ranking objectives."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

PairDistance = Literal["cosine", "euclidean", "manhattan"]


class PairLossTerms(eqx.Module):
    """Auditable per-pair terms for regression and contrastive objectives."""

    loss: Float[Array, ""]
    scores: Float[Array, " pair"]
    row_losses: Float[Array, " pair"]
    selected: Bool[Array, " pair"]


class PairRankingTerms(eqx.Module):
    """Auditable score-ordering terms for CoSENT and AnglE."""

    loss: Float[Array, ""]
    similarity: Float[Array, " pair"]
    score_differences: Float[Array, "pair pair"]
    ordered_pairs: Bool[Array, "pair pair"]


def _pair_inputs(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
) -> tuple[
    Float[Array, "pair representation"],
    Float[Array, "pair representation"],
]:
    left = jnp.asarray(left, dtype=jnp.float32)
    right = jnp.asarray(right, dtype=jnp.float32)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("pair embeddings must be matrices")
    if left.shape != right.shape:
        raise ValueError("pair embedding matrices must have the same shape")
    return left, right


def _valid(
    pair_count: int,
    valid: Bool[Array, " pair"] | None,
) -> Bool[Array, " pair"]:
    if valid is None:
        return jnp.ones((pair_count,), dtype=jnp.bool_)
    resolved = jnp.asarray(valid)
    if resolved.shape != (pair_count,) or resolved.dtype != jnp.bool_:
        raise TypeError("valid must be a boolean vector matching pair rows")
    return resolved


def _labels(
    pair_count: int,
    labels: Float[Array, " pair"],
) -> Float[Array, " pair"]:
    resolved = jnp.asarray(labels, dtype=jnp.float32)
    if resolved.shape != (pair_count,):
        raise ValueError("labels must match pair rows")
    return resolved


def _masked_mean(
    values: Float[Array, " pair"],
    valid: Bool[Array, " pair"],
) -> Float[Array, ""]:
    count = jnp.sum(valid)
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(count, 1).astype(
        jnp.float32
    )


def pairwise_cosine_similarity(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
) -> Float[Array, " pair"]:
    """Cosine similarity between aligned rows using Torch's 1e-8 floor."""

    left, right = _pair_inputs(left, right)
    left_norm = jnp.maximum(jnp.linalg.norm(left, axis=1), 1e-8)
    right_norm = jnp.maximum(jnp.linalg.norm(right, axis=1), 1e-8)
    return jnp.sum(left * right, axis=1) / (left_norm * right_norm)


def pairwise_angle_similarity(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
) -> Float[Array, " pair"]:
    """AnglE's absolute normalized complex-space similarity."""

    left, right = _pair_inputs(left, right)
    if left.shape[1] % 2:
        left = jnp.pad(left, ((0, 0), (0, 1)))
        right = jnp.pad(right, ((0, 0), (0, 1)))
    a, b = jnp.split(left, 2, axis=1)
    c, d = jnp.split(right, 2, axis=1)
    denominator = jnp.sum(c**2 + d**2, axis=1, keepdims=True)
    real = (a * c + b * d) / denominator
    imaginary = (b * c - a * d) / denominator
    left_magnitude = jnp.sum(a**2 + b**2, axis=1, keepdims=True) ** 0.5
    right_magnitude = jnp.sum(c**2 + d**2, axis=1, keepdims=True) ** 0.5
    magnitude_ratio = left_magnitude / right_magnitude
    real = real / magnitude_ratio
    imaginary = imaginary / magnitude_ratio
    return jnp.abs(jnp.sum(jnp.concatenate((real, imaginary), axis=1), axis=1))


def pairwise_distance(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    *,
    metric: PairDistance,
) -> Float[Array, " pair"]:
    """Canonical aligned-row distance used by contrastive objectives."""

    left, right = _pair_inputs(left, right)
    if metric == "cosine":
        return 1.0 - pairwise_cosine_similarity(left, right)
    # Torch pairwise_distance adds eps componentwise before taking the norm.
    difference = left - right + jnp.asarray(1e-6, dtype=jnp.float32)
    if metric == "euclidean":
        return jnp.linalg.norm(difference, ord=2, axis=1)
    if metric == "manhattan":
        return jnp.linalg.norm(difference, ord=1, axis=1)
    raise ValueError(f"unsupported pair distance {metric!r}")


def cosine_regression_loss_terms(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    labels: Float[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
) -> PairLossTerms:
    """Mean squared error between pair cosine similarity and float labels."""

    similarity = pairwise_cosine_similarity(left, right)
    labels = _labels(similarity.shape[0], labels)
    resolved_valid = _valid(similarity.shape[0], valid)
    row_losses = jnp.square(similarity - labels)
    return PairLossTerms(
        loss=_masked_mean(row_losses, resolved_valid),
        scores=similarity,
        row_losses=row_losses,
        selected=resolved_valid,
    )


def contrastive_loss_terms(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    labels: Float[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
    metric: PairDistance = "cosine",
    margin: float = 0.5,
) -> PairLossTerms:
    """Hadsell contrastive loss over labeled positive and negative pairs."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("contrastive margin must be finite and positive")
    distances = pairwise_distance(left, right, metric=metric)
    labels = _labels(distances.shape[0], labels)
    resolved_valid = _valid(distances.shape[0], valid)
    row_losses = 0.5 * (
        labels * jnp.square(distances)
        + (1.0 - labels) * jnp.square(jax.nn.relu(margin - distances))
    )
    return PairLossTerms(
        loss=_masked_mean(row_losses, resolved_valid),
        scores=distances,
        row_losses=row_losses,
        selected=resolved_valid,
    )


def online_contrastive_loss_terms(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    labels: Float[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
    metric: PairDistance = "cosine",
    margin: float = 0.5,
) -> PairLossTerms:
    """Select hard positive and negative pairs before contrastive reduction."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("contrastive margin must be finite and positive")
    distances = pairwise_distance(left, right, metric=metric)
    labels = _labels(distances.shape[0], labels)
    resolved_valid = _valid(distances.shape[0], valid)
    positives = resolved_valid & (labels == 1.0)
    negatives = resolved_valid & (labels == 0.0)
    positive_count = jnp.sum(positives)
    negative_count = jnp.sum(negatives)

    positive_max = jnp.max(jnp.where(positives, distances, -jnp.inf))
    negative_min = jnp.min(jnp.where(negatives, distances, jnp.inf))
    positive_mean = jnp.sum(jnp.where(positives, distances, 0.0)) / jnp.maximum(
        positive_count, 1
    ).astype(jnp.float32)
    negative_mean = jnp.sum(jnp.where(negatives, distances, 0.0)) / jnp.maximum(
        negative_count, 1
    ).astype(jnp.float32)
    negative_cutoff = jnp.where(positive_count > 1, positive_max, negative_mean)
    positive_cutoff = jnp.where(negative_count > 1, negative_min, positive_mean)
    selected_negatives = negatives & (distances < negative_cutoff)
    selected_positives = positives & (distances > positive_cutoff)
    selected = selected_positives | selected_negatives
    row_losses = jnp.where(
        selected_positives,
        jnp.square(distances),
        jnp.where(
            selected_negatives,
            jnp.square(jax.nn.relu(margin - distances)),
            0.0,
        ),
    )
    return PairLossTerms(
        loss=jnp.sum(row_losses),
        scores=distances,
        row_losses=row_losses,
        selected=selected,
    )


def pair_ranking_loss_terms(
    left: Float[Array, "pair representation"],
    right: Float[Array, "pair representation"],
    labels: Float[Array, " pair"],
    *,
    valid: Bool[Array, " pair"] | None = None,
    scale: float = 20.0,
    similarity: Literal["cosine", "angle"] = "cosine",
) -> PairRankingTerms:
    """CoSENT score-ordering loss with cosine or AnglE similarity."""

    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("pair ranking scale must be finite and positive")
    if similarity == "cosine":
        pair_scores = pairwise_cosine_similarity(left, right)
    elif similarity == "angle":
        pair_scores = pairwise_angle_similarity(left, right)
    else:
        raise ValueError(f"unsupported pair ranking similarity {similarity!r}")
    labels = _labels(pair_scores.shape[0], labels)
    resolved_valid = _valid(pair_scores.shape[0], valid)
    scaled = pair_scores * jnp.asarray(scale, dtype=jnp.float32)
    differences = scaled[:, None] - scaled[None, :]
    ordered = (
        (labels[:, None] < labels[None, :])
        & resolved_valid[:, None]
        & resolved_valid[None, :]
    )
    candidates = jnp.where(ordered, differences, -jnp.inf).reshape(-1)
    loss = jax.nn.logsumexp(
        jnp.concatenate((jnp.zeros((1,), dtype=jnp.float32), candidates))
    )
    return PairRankingTerms(
        loss=loss,
        similarity=pair_scores,
        score_differences=differences,
        ordered_pairs=ordered,
    )


__all__ = [
    "PairDistance",
    "PairLossTerms",
    "PairRankingTerms",
    "contrastive_loss_terms",
    "cosine_regression_loss_terms",
    "online_contrastive_loss_terms",
    "pair_ranking_loss_terms",
    "pairwise_angle_similarity",
    "pairwise_cosine_similarity",
    "pairwise_distance",
]
