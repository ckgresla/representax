"""Native explicit and within-batch-mined triplet objectives."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

ExplicitTripletDistance = Literal["cosine", "euclidean", "manhattan"]
BatchTripletDistance = Literal["cosine", "euclidean", "squared_euclidean"]


class ExplicitTripletLossTerms(eqx.Module):
    """Auditable per-row terms for supplied triplets."""

    loss: Float[Array, ""]
    positive_distances: Float[Array, " triplet"]
    negative_distances: Float[Array, " triplet"]
    row_losses: Float[Array, " triplet"]
    selected: Bool[Array, " triplet"]


class BatchTripletLossTerms(eqx.Module):
    """Auditable terms for one in-batch mining policy."""

    loss: Float[Array, ""]
    pairwise_distances: Float[Array, "example example"]
    candidate_losses: Float[Array, " *candidate"]
    selected: Bool[Array, " *candidate"]


def _embedding_matrix(
    embeddings: Float[Array, "example representation"],
) -> Float[Array, "example representation"]:
    resolved = jnp.asarray(embeddings, dtype=jnp.float32)
    if resolved.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    return resolved


def _aligned_inputs(
    left: Float[Array, "row representation"],
    right: Float[Array, "row representation"],
) -> tuple[
    Float[Array, "row representation"],
    Float[Array, "row representation"],
]:
    left = _embedding_matrix(left)
    right = _embedding_matrix(right)
    if left.shape != right.shape:
        raise ValueError("aligned embedding matrices must have the same shape")
    return left, right


def _valid(count: int, valid: Bool[Array, " row"] | None) -> Bool[Array, " row"]:
    if valid is None:
        return jnp.ones((count,), dtype=jnp.bool_)
    resolved = jnp.asarray(valid)
    if resolved.shape != (count,) or resolved.dtype != jnp.bool_:
        raise TypeError("valid must be a boolean vector matching embedding rows")
    return resolved


def _labels(
    count: int,
    labels: Int[Array, " example"],
) -> Int[Array, " example"]:
    resolved = jnp.asarray(labels)
    if resolved.shape != (count,) or not jnp.issubdtype(resolved.dtype, jnp.integer):
        raise TypeError("labels must be an integer vector matching embedding rows")
    return resolved


def _masked_mean(
    values: Float[Array, " *row"],
    selected: Bool[Array, " *row"],
) -> Float[Array, ""]:
    count = jnp.sum(selected)
    return jnp.sum(jnp.where(selected, values, 0.0)) / jnp.maximum(count, 1).astype(
        jnp.float32
    )


def aligned_triplet_distance(
    left: Float[Array, "row representation"],
    right: Float[Array, "row representation"],
    *,
    metric: ExplicitTripletDistance,
) -> Float[Array, " row"]:
    """Distance between aligned rows for explicit triplet supervision."""

    left, right = _aligned_inputs(left, right)
    if metric == "cosine":
        left_norm = jnp.maximum(jnp.linalg.norm(left, axis=1), 1e-12)
        right_norm = jnp.maximum(jnp.linalg.norm(right, axis=1), 1e-12)
        return 1.0 - jnp.sum(left * right, axis=1) / (left_norm * right_norm)
    difference = left - right
    if metric == "euclidean":
        return jnp.sqrt(jnp.sum(jnp.square(difference), axis=1))
    if metric == "manhattan":
        return jnp.sum(jnp.abs(difference), axis=1)
    raise ValueError(f"unsupported explicit triplet distance {metric!r}")


def pairwise_triplet_distances(
    embeddings: Float[Array, "example representation"],
    *,
    metric: BatchTripletDistance,
) -> Float[Array, "example example"]:
    """Full within-batch distance matrix used by mining policies."""

    embeddings = _embedding_matrix(embeddings)
    if metric == "cosine":
        norms = jnp.maximum(jnp.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        normalized = embeddings / norms
        return 1.0 - normalized @ normalized.T
    products = embeddings @ embeddings.T
    square_norms = jnp.diag(products)
    squared = jnp.maximum(
        square_norms[None, :] - 2.0 * products + square_norms[:, None],
        0.0,
    )
    if metric == "squared_euclidean":
        return squared
    if metric == "euclidean":
        zero = squared == 0.0
        return (~zero).astype(jnp.float32) * jnp.sqrt(
            squared + zero.astype(jnp.float32) * 1e-16
        )
    raise ValueError(f"unsupported batch triplet distance {metric!r}")


def explicit_triplet_loss_terms(
    anchor: Float[Array, "triplet representation"],
    positive: Float[Array, "triplet representation"],
    negative: Float[Array, "triplet representation"],
    *,
    valid: Bool[Array, " triplet"] | None = None,
    metric: ExplicitTripletDistance = "euclidean",
    margin: float = 5.0,
) -> ExplicitTripletLossTerms:
    """Margin loss over aligned anchor-positive-negative rows."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("triplet margin must be finite and positive")
    positive_distances = aligned_triplet_distance(anchor, positive, metric=metric)
    negative_distances = aligned_triplet_distance(anchor, negative, metric=metric)
    if positive_distances.shape != negative_distances.shape:
        raise ValueError("positive and negative triplet rows must align")
    selected = _valid(positive_distances.shape[0], valid)
    row_losses = jax.nn.relu(positive_distances - negative_distances + margin)
    return ExplicitTripletLossTerms(
        loss=_masked_mean(row_losses, selected),
        positive_distances=positive_distances,
        negative_distances=negative_distances,
        row_losses=row_losses,
        selected=selected,
    )


def triplet_masks(
    labels: Int[Array, " example"],
    *,
    valid: Bool[Array, " example"] | None = None,
) -> tuple[
    Bool[Array, "example example"],
    Bool[Array, "example example"],
    Bool[Array, "example example example"],
]:
    """Return valid anchor-positive, anchor-negative, and triplet masks."""

    labels = jnp.asarray(labels)
    if labels.ndim != 1 or not jnp.issubdtype(labels.dtype, jnp.integer):
        raise TypeError("labels must be an integer vector")
    resolved_valid = _valid(labels.shape[0], valid)
    same_label = labels[:, None] == labels[None, :]
    distinct = ~jnp.eye(labels.shape[0], dtype=jnp.bool_)
    both_valid = resolved_valid[:, None] & resolved_valid[None, :]
    positive = both_valid & distinct & same_label
    negative = both_valid & ~same_label
    triplets = positive[:, :, None] & negative[:, None, :]
    return positive, negative, triplets


def batch_all_triplet_loss_terms(
    embeddings: Float[Array, "example representation"],
    labels: Int[Array, " example"],
    *,
    valid: Bool[Array, " example"] | None = None,
    metric: BatchTripletDistance = "euclidean",
    margin: float = 5.0,
) -> BatchTripletLossTerms:
    """Average every positive-loss valid triplet in the batch."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("triplet margin must be finite and positive")
    embeddings = _embedding_matrix(embeddings)
    labels = _labels(embeddings.shape[0], labels)
    distances = pairwise_triplet_distances(embeddings, metric=metric)
    _, _, valid_triplets = triplet_masks(labels, valid=valid)
    losses = jax.nn.relu(distances[:, :, None] - distances[:, None, :] + margin)
    selected = valid_triplets & (losses > 1e-16)
    return BatchTripletLossTerms(
        loss=_masked_mean(losses, selected),
        pairwise_distances=distances,
        candidate_losses=losses,
        selected=selected,
    )


def _hard_distances(
    distances: Float[Array, "example example"],
    positive: Bool[Array, "example example"],
    negative: Bool[Array, "example example"],
) -> tuple[
    Float[Array, " example"],
    Float[Array, " example"],
    Bool[Array, " example"],
]:
    hardest_positive = jnp.max(jnp.where(positive, distances, -jnp.inf), axis=1)
    hardest_negative = jnp.min(jnp.where(negative, distances, jnp.inf), axis=1)
    eligible = jnp.any(positive, axis=1) & jnp.any(negative, axis=1)
    return (
        jnp.where(eligible, hardest_positive, 0.0),
        jnp.where(eligible, hardest_negative, 0.0),
        eligible,
    )


def batch_hard_triplet_loss_terms(
    embeddings: Float[Array, "example representation"],
    labels: Int[Array, " example"],
    *,
    valid: Bool[Array, " example"] | None = None,
    metric: BatchTripletDistance = "euclidean",
    margin: float = 5.0,
    soft_margin: bool = False,
) -> BatchTripletLossTerms:
    """Use each anchor's farthest positive and nearest negative."""

    if not soft_margin and (not math.isfinite(margin) or margin <= 0):
        raise ValueError("triplet margin must be finite and positive")
    embeddings = _embedding_matrix(embeddings)
    labels = _labels(embeddings.shape[0], labels)
    distances = pairwise_triplet_distances(embeddings, metric=metric)
    positive, negative, _ = triplet_masks(labels, valid=valid)
    hardest_positive, hardest_negative, selected = _hard_distances(
        distances,
        positive,
        negative,
    )
    differences = hardest_positive - hardest_negative
    losses = (
        jax.nn.softplus(differences)
        if soft_margin
        else jax.nn.relu(differences + margin)
    )
    return BatchTripletLossTerms(
        loss=_masked_mean(losses, selected),
        pairwise_distances=distances,
        candidate_losses=losses,
        selected=selected,
    )


def batch_semi_hard_triplet_loss_terms(
    embeddings: Float[Array, "example representation"],
    labels: Int[Array, " example"],
    *,
    valid: Bool[Array, " example"] | None = None,
    metric: BatchTripletDistance = "euclidean",
    margin: float = 5.0,
) -> BatchTripletLossTerms:
    """Choose the nearest farther negative, falling back to the farthest negative."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("triplet margin must be finite and positive")
    embeddings = _embedding_matrix(embeddings)
    labels = _labels(embeddings.shape[0], labels)
    distances = pairwise_triplet_distances(embeddings, metric=metric)
    positive, negative, _ = triplet_masks(labels, valid=valid)

    farther_negative = negative[:, None, :] & (
        distances[:, None, :] > distances[:, :, None]
    )
    has_farther = jnp.any(farther_negative, axis=2)
    nearest_farther = jnp.min(
        jnp.where(farther_negative, distances[:, None, :], jnp.inf),
        axis=2,
    )
    farthest_negative = jnp.max(
        jnp.where(negative, distances, -jnp.inf),
        axis=1,
    )
    chosen_negative = jnp.where(
        has_farther,
        nearest_farther,
        farthest_negative[:, None],
    )
    eligible = positive & jnp.any(negative, axis=1)[:, None]
    losses = jax.nn.relu(distances - chosen_negative + margin)
    return BatchTripletLossTerms(
        loss=_masked_mean(losses, eligible),
        pairwise_distances=distances,
        candidate_losses=losses,
        selected=eligible,
    )


__all__ = [
    "BatchTripletDistance",
    "BatchTripletLossTerms",
    "ExplicitTripletDistance",
    "ExplicitTripletLossTerms",
    "aligned_triplet_distance",
    "batch_all_triplet_loss_terms",
    "batch_hard_triplet_loss_terms",
    "batch_semi_hard_triplet_loss_terms",
    "explicit_triplet_loss_terms",
    "pairwise_triplet_distances",
    "triplet_masks",
]
