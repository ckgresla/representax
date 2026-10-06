"""Global orthogonal regularization terms."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class GORTerms(eqx.Module):
    loss: Float[Array, ""]
    mean_term: Float[Array, ""]
    second_moment_term: Float[Array, ""]
    column_mean_terms: Float[Array, " column"]
    column_second_moment_terms: Float[Array, " column"]


def _normalize(values: Array) -> Array:
    return values / jnp.maximum(
        jnp.linalg.norm(values, axis=-1, keepdims=True),
        jnp.asarray(1e-12, values.dtype),
    )


def global_orthogonal_regularization_terms(
    embeddings: tuple[Float[Array, "batch representation"], ...],
    *,
    valid: Bool[Array, " batch"] | None = None,
    similarity: Literal["cosine", "dot"] = "cosine",
    mean_weight: float = 1.0,
    second_moment_weight: float = 1.0,
    aggregation: Literal["mean", "sum"] = "mean",
) -> GORTerms:
    """Encourage zero-mean similarities and spherical second moments."""

    if not embeddings:
        raise ValueError("GOR requires at least one embedding column")
    batch_size = embeddings[0].shape[0]
    if any(value.ndim != 2 or value.shape[0] != batch_size for value in embeddings):
        raise ValueError("GOR embedding columns must be aligned matrices")
    if similarity not in {"cosine", "dot"}:
        raise ValueError("GOR similarity must be cosine or dot")
    if aggregation not in {"mean", "sum"}:
        raise ValueError("GOR aggregation must be mean or sum")
    if mean_weight == 0 and second_moment_weight == 0:
        raise ValueError("at least one GOR weight must be non-zero")
    resolved_valid = (
        jnp.ones((batch_size,), dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    pair_mask = resolved_valid[:, None] & resolved_valid[None, :]
    pair_mask = pair_mask & ~jnp.eye(batch_size, dtype=jnp.bool_)
    pair_count = jnp.maximum(jnp.sum(pair_mask), 1).astype(jnp.float32)
    mean_terms = []
    second_moment_terms = []
    for value in embeddings:
        compared = _normalize(value) if similarity == "cosine" else value
        similarities = compared @ compared.T
        similarities = jnp.where(pair_mask, similarities, 0.0)
        mean_terms.append(jnp.square(jnp.sum(similarities) / pair_count))
        second_moment = jnp.sum(jnp.square(similarities)) / pair_count
        second_moment_terms.append(jax.nn.relu(second_moment - 1.0 / value.shape[-1]))
    column_means = jnp.stack(mean_terms)
    column_seconds = jnp.stack(second_moment_terms)
    reduce = jnp.sum if aggregation == "sum" else jnp.mean
    mean_term = mean_weight * reduce(column_means)
    second_moment_term = second_moment_weight * reduce(column_seconds)
    return GORTerms(
        loss=mean_term + second_moment_term,
        mean_term=mean_term,
        second_moment_term=second_moment_term,
        column_mean_terms=column_means,
        column_second_moment_terms=column_seconds,
    )
