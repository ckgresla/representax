"""Retrieval task inputs with explicit positive relationships."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


class RetrievalBatch(eqx.Module):
    """Model-native query/document payloads and a dense positive relation."""

    query: Any
    document: Any
    positive_mask: jax.Array
    positive_weights: jax.Array | None
    query_valid: jax.Array
    document_valid: jax.Array

    def __post_init__(self) -> None:
        query_count, document_count = self.positive_mask.shape
        if self.positive_mask.ndim != 2 or self.positive_mask.dtype != jnp.bool_:
            raise TypeError("positive_mask must be a boolean matrix")
        if self.positive_weights is not None:
            if self.positive_weights.shape != self.positive_mask.shape:
                raise ValueError("positive_weights must match positive_mask")
            if not jnp.issubdtype(self.positive_weights.dtype, jnp.floating):
                raise TypeError("positive_weights must be floating point")
        if self.query_valid.shape != (query_count,):
            raise ValueError("query_valid must match the query dimension")
        if self.document_valid.shape != (document_count,):
            raise ValueError("document_valid must match the document dimension")
        if self.query_valid.dtype != jnp.bool_:
            raise TypeError("query_valid must be boolean")
        if self.document_valid.dtype != jnp.bool_:
            raise TypeError("document_valid must be boolean")
        query_leaves = [x for x in jax.tree.leaves(self.query) if eqx.is_array(x)]
        document_leaves = [x for x in jax.tree.leaves(self.document) if eqx.is_array(x)]
        if not query_leaves or not document_leaves:
            raise ValueError("query and document payloads must contain arrays")
        if any(x.ndim == 0 or x.shape[0] != query_count for x in query_leaves):
            raise ValueError("query payloads must be row-major")
        if any(x.ndim == 0 or x.shape[0] != document_count for x in document_leaves):
            raise ValueError("document payloads must be row-major")


def retrieval_batch(
    *,
    query: Any,
    document: Any,
    positive_mask: jax.Array,
    positive_weights: jax.Array | None = None,
    query_valid: jax.Array | None = None,
    document_valid: jax.Array | None = None,
) -> RetrievalBatch:
    """Build a fixed-shape retrieval batch with sensible validity defaults."""

    positive_mask = jnp.asarray(positive_mask, dtype=jnp.bool_)
    query_count, document_count = positive_mask.shape
    if query_valid is None:
        query_valid = jnp.ones((query_count,), dtype=jnp.bool_)
    if document_valid is None:
        document_valid = jnp.ones((document_count,), dtype=jnp.bool_)
    return RetrievalBatch(
        query=query,
        document=document,
        positive_mask=positive_mask,
        positive_weights=(
            None if positive_weights is None else jnp.asarray(positive_weights)
        ),
        query_valid=jnp.asarray(query_valid, dtype=jnp.bool_),
        document_valid=jnp.asarray(document_valid, dtype=jnp.bool_),
    )
