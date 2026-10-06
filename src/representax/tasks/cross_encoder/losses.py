"""Pure-JAX pointwise, pairwise, and listwise ranking objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from .config import LambdaWeighting, ReductionLog, ScoreActivation


def activate_scores(
    logits: Float[Array, "*batch"], activation: ScoreActivation
) -> Float[Array, "*batch"]:
    if activation == "identity":
        return logits
    if activation == "sigmoid":
        return jax.nn.sigmoid(logits)
    if activation == "tanh":
        return jnp.tanh(logits)
    raise ValueError(f"unsupported score activation {activation!r}")


def _weighted_mean(
    values: Float[Array, "*batch"], valid: Bool[Array, "*batch"]
) -> Float[Array, ""]:
    weights = valid.astype(jnp.float32)
    return jnp.sum(jnp.where(valid, values, 0.0)) / jnp.maximum(jnp.sum(weights), 1.0)


def binary_cross_entropy(
    logits: Float[Array, " batch"],
    labels: Float[Array, " batch"],
    valid: Bool[Array, " batch"],
    *,
    positive_weight: float | None = None,
) -> Float[Array, ""]:
    """Stable BCE-with-logits over valid rows."""

    if logits.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("binary logits, labels, and validity must align")
    negative = (1.0 - labels) * jax.nn.softplus(logits)
    positive = labels * jax.nn.softplus(-logits)
    if positive_weight is not None:
        positive = positive * positive_weight
    return _weighted_mean(negative + positive, valid)


def multiclass_cross_entropy(
    logits: Float[Array, "batch class"],
    labels: Int[Array, " batch"],
    valid: Bool[Array, " batch"],
) -> Float[Array, ""]:
    if (
        logits.ndim != 2
        or labels.shape != logits.shape[:1]
        or valid.shape != labels.shape
    ):
        raise ValueError("multiclass logits, labels, and validity must align")
    losses = -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(logits.shape[0]), labels]
    return _weighted_mean(losses, valid)


def score_mse(
    scores: Float[Array, "*batch"],
    labels: Float[Array, "*batch"],
    valid: Bool[Array, "*batch"],
) -> Float[Array, ""]:
    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("scores, labels, and validity must align")
    return _weighted_mean(jnp.square(scores - labels), valid)


def cross_mnr_loss(
    logits: Float[Array, "query candidate"],
    positive_indices: Int[Array, " query"],
    valid: Bool[Array, "query candidate"],
    *,
    activation: ScoreActivation = "sigmoid",
    scale: float = 10.0,
) -> Float[Array, ""]:
    """Exact cross-encoder InfoNCE over a query/candidate score matrix."""

    if logits.shape != valid.shape or positive_indices.shape != logits.shape[:1]:
        raise ValueError("cross MNR logits, positives, and validity must align")
    scores = activate_scores(logits, activation) * scale
    scores = jnp.where(valid, scores, -jnp.inf)
    row_losses = -jax.nn.log_softmax(scores, axis=-1)[
        jnp.arange(scores.shape[0]), positive_indices
    ]
    query_valid = jnp.any(valid, axis=-1)
    return _weighted_mean(row_losses, query_valid)


def ranknet_loss(
    scores: Float[Array, "query document"],
    labels: Float[Array, "query document"],
    valid: Bool[Array, "query document"],
    *,
    sigma: float = 1.0,
    k: int | None = None,
    reduction_log: ReductionLog = "binary",
) -> Float[Array, ""]:
    """RankNet logistic loss over every strictly ordered target pair."""

    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("listwise scores, labels, and validity must align")
    return lambda_loss(
        scores,
        labels,
        valid,
        weighting="none",
        sigma=sigma,
        k=k,
        reduction_log=reduction_log,
    )


def lambda_loss(
    scores: Float[Array, "query document"],
    labels: Float[Array, "query document"],
    valid: Bool[Array, "query document"],
    *,
    weighting: LambdaWeighting = "ndcg_loss2pp",
    k: int | None = None,
    sigma: float = 1.0,
    reduction_log: ReductionLog = "binary",
    mu: float = 10.0,
    eps: float = 1e-10,
) -> Float[Array, ""]:
    """Sentence Transformers-compatible LambdaLoss family."""

    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("listwise scores, labels, and validity must align")
    width = scores.shape[1]
    cutoff = width if k is None else min(k, width)
    masked_scores = jnp.where(valid, scores, -1e16)
    masked_labels = jnp.where(valid, labels, -jnp.inf)
    predicted_order = jnp.argsort(masked_scores, axis=-1)[:, ::-1]
    scores_by_prediction = jnp.take_along_axis(masked_scores, predicted_order, axis=1)
    labels_by_prediction = jnp.take_along_axis(masked_labels, predicted_order, axis=1)
    labels_ideal = jnp.sort(masked_labels, axis=-1)[:, ::-1]
    differences = labels_by_prediction[:, :, None] - labels_by_prediction[:, None, :]
    pairs = jnp.isfinite(differences)
    if weighting != "ndcg_loss1":
        pairs = pairs & (differences > 0)
    at_k = jnp.arange(width) < cutoff
    pairs = pairs & at_k[:, None] & at_k[None, :]

    positions = jnp.arange(1, width + 1, dtype=jnp.float32)
    discount = jnp.log2(1.0 + positions)
    ideal_gain = jnp.exp2(jnp.maximum(labels_ideal, 0.0)) - 1.0
    maximum_dcg = jnp.maximum(
        jnp.sum((ideal_gain / discount)[:, :cutoff], axis=-1), eps
    )
    gain = (jnp.exp2(jnp.maximum(labels_by_prediction, 0.0)) - 1.0) / maximum_dcg[
        :, None
    ]
    gain_difference = jnp.abs(gain[:, :, None] - gain[:, None, :])

    if weighting == "none":
        weights = jnp.ones_like(gain_difference)
    elif weighting == "ndcg_loss1":
        weights = (gain / discount)[:, :, None]
    elif weighting in {"ndcg_loss2", "ndcg_loss2pp"}:
        distance = jnp.abs(
            jnp.arange(1, width + 1)[:, None] - jnp.arange(1, width + 1)[None, :]
        )
        left = jnp.take(1.0 / discount, jnp.maximum(distance - 1, 0))
        right = jnp.take(1.0 / discount, distance)
        delta = jnp.where(distance == 0, 0.0, jnp.abs(left - right))
        ndcg2 = delta[None, :, :] * gain_difference
        if weighting == "ndcg_loss2":
            weights = ndcg2
        else:
            rank_delta = jnp.abs(1.0 / discount[:, None] - 1.0 / discount[None, :])
            weights = mu * ndcg2 + rank_delta[None, :, :] * gain_difference
    elif weighting == "lambda_rank":
        rank_delta = jnp.abs(1.0 / discount[:, None] - 1.0 / discount[None, :])
        weights = rank_delta[None, :, :] * gain_difference
    else:
        raise ValueError(f"unsupported LambdaLoss weighting {weighting!r}")

    score_difference = jnp.clip(
        scores_by_prediction[:, :, None] - scores_by_prediction[:, None, :],
        -1e8,
        1e8,
    )
    probabilities = jnp.maximum(jax.nn.sigmoid(sigma * score_difference), eps)
    log_probabilities = weights * jnp.log(probabilities)
    if reduction_log == "binary":
        log_probabilities = log_probabilities / jnp.log(2.0)
    elif reduction_log != "natural":
        raise ValueError(f"unsupported logarithm reduction {reduction_log!r}")
    return _weighted_mean(-log_probabilities, pairs)


def listnet_loss(
    scores: Float[Array, "query document"],
    labels: Float[Array, "query document"],
    valid: Bool[Array, "query document"],
) -> Float[Array, ""]:
    """Top-one ListNet cross entropy over finite candidate lists."""

    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("listwise scores, labels, and validity must align")
    masked_scores = jnp.where(valid, scores, -jnp.inf)
    masked_labels = jnp.where(valid, labels, -jnp.inf)
    targets = jax.nn.softmax(masked_labels, axis=-1)
    row_losses = -jnp.sum(
        jnp.where(valid, targets * jax.nn.log_softmax(masked_scores, axis=-1), 0.0),
        axis=-1,
    )
    return jnp.mean(row_losses)


def list_mle_loss(
    scores: Float[Array, "query document"],
    labels: Float[Array, "query document"],
    valid: Bool[Array, "query document"],
    *,
    respect_input_order: bool = True,
    position_aware: bool = False,
) -> Float[Array, ""]:
    """ListMLE or position-aware ListMLE over padded candidate lists."""

    if scores.shape != labels.shape or valid.shape != labels.shape:
        raise ValueError("listwise scores, labels, and validity must align")
    if respect_input_order:
        ordered_scores = scores
        ordered_valid = valid
    else:
        order = jnp.argsort(jnp.where(valid, labels, -jnp.inf), axis=-1)[:, ::-1]
        ordered_scores = jnp.take_along_axis(scores, order, axis=1)
        ordered_valid = jnp.take_along_axis(valid, order, axis=1)
    # Sentence Transformers fills padded logits with 1e-16 before constructing
    # the suffix partition. Consequently their exp(0)-like padded terms remain
    # in earlier denominators even though the final log-probability is masked.
    masked = jnp.where(ordered_valid, ordered_scores, 1e-16)
    suffix_logsumexp = jnp.flip(
        jax.lax.associative_scan(jnp.logaddexp, jnp.flip(masked, axis=1), axis=1),
        axis=1,
    )
    terms = jnp.where(ordered_valid, masked - suffix_logsumexp, 0.0)
    if position_aware:
        ranks = jnp.arange(scores.shape[1], dtype=jnp.float32)[None, :]
        counts = jnp.sum(ordered_valid, axis=1, keepdims=True)
        weights = jnp.where(ordered_valid, jnp.exp2(counts - ranks) - 1.0, 0.0)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1e-10)
        terms = terms * weights
    return jnp.mean(-jnp.sum(terms, axis=1))


__all__ = [
    "activate_scores",
    "binary_cross_entropy",
    "cross_mnr_loss",
    "lambda_loss",
    "list_mle_loss",
    "listnet_loss",
    "multiclass_cross_entropy",
    "ranknet_loss",
    "score_mse",
]
