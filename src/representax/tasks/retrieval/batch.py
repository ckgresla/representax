"""Retrieval task inputs with explicit positive relationships."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jaxtyping import Array, Bool, Float


class RetrievalBatch(eqx.Module):
    """Model-native query/document payloads and a dense positive relation."""

    query: Any
    document: Any
    positive_mask: Bool[Array, "query document"]
    positive_weights: Float[Array, "query document"] | None
    query_valid: Bool[Array, " query"]
    document_valid: Bool[Array, " document"]

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


class ProcessLocalRetrievalBatch(eqx.Module):
    """Process-local payload rows with query relations to all global documents."""

    query: Any
    document: Any
    positive_mask: Bool[Array, "local_query global_document"]
    positive_weights: Float[Array, "local_query global_document"] | None
    query_valid: Bool[Array, " local_query"]
    document_valid: Bool[Array, " local_document"]

    def __post_init__(self) -> None:
        if self.positive_mask.ndim != 2 or self.positive_mask.dtype != jnp.bool_:
            raise TypeError("positive_mask must be a boolean matrix")
        local_query_count, global_document_count = self.positive_mask.shape
        if self.positive_weights is not None:
            if self.positive_weights.shape != self.positive_mask.shape:
                raise ValueError("positive_weights must match positive_mask")
            if not jnp.issubdtype(self.positive_weights.dtype, jnp.floating):
                raise TypeError("positive_weights must be floating point")
        if self.query_valid.shape != (local_query_count,):
            raise ValueError("query_valid must match the local query dimension")
        if self.query_valid.dtype != jnp.bool_:
            raise TypeError("query_valid must be boolean")

        query_leaves = [x for x in jax.tree.leaves(self.query) if eqx.is_array(x)]
        document_leaves = [x for x in jax.tree.leaves(self.document) if eqx.is_array(x)]
        if not query_leaves or not document_leaves:
            raise ValueError("query and document payloads must contain arrays")
        if any(x.ndim == 0 or x.shape[0] != local_query_count for x in query_leaves):
            raise ValueError("query payloads must contain process-local rows")
        local_document_count = document_leaves[0].shape[0]
        if any(
            x.ndim == 0 or x.shape[0] != local_document_count for x in document_leaves
        ):
            raise ValueError("document payloads must contain process-local rows")
        if self.document_valid.shape != (local_document_count,):
            raise ValueError("document_valid must match the local document dimension")
        if self.document_valid.dtype != jnp.bool_:
            raise TypeError("document_valid must be boolean")
        if global_document_count < local_document_count:
            raise ValueError(
                "positive_mask must address at least the local document rows"
            )


def retrieval_batch(
    *,
    query: Any,
    document: Any,
    positive_mask: Bool[Array, "query document"],
    positive_weights: Float[Array, "query document"] | None = None,
    query_valid: Bool[Array, " query"] | None = None,
    document_valid: Bool[Array, " document"] | None = None,
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


def process_local_retrieval_batch(
    *,
    query: Any,
    document: Any,
    positive_mask: Bool[Array, "local_query global_document"],
    positive_weights: Float[Array, "local_query global_document"] | None = None,
    query_valid: Bool[Array, " local_query"] | None = None,
    document_valid: Bool[Array, " local_document"] | None = None,
) -> ProcessLocalRetrievalBatch:
    """Build process-local rows that retain the global document relation axis."""

    positive_mask = jnp.asarray(positive_mask, dtype=jnp.bool_)
    local_query_count = positive_mask.shape[0]
    document_leaves = [
        value for value in jax.tree.leaves(document) if eqx.is_array(value)
    ]
    if not document_leaves:
        raise ValueError("document payloads must contain arrays")
    local_document_count = document_leaves[0].shape[0]
    if query_valid is None:
        query_valid = jnp.ones((local_query_count,), dtype=jnp.bool_)
    if document_valid is None:
        document_valid = jnp.ones((local_document_count,), dtype=jnp.bool_)
    return ProcessLocalRetrievalBatch(
        query=query,
        document=document,
        positive_mask=positive_mask,
        positive_weights=(
            None if positive_weights is None else jnp.asarray(positive_weights)
        ),
        query_valid=jnp.asarray(query_valid, dtype=jnp.bool_),
        document_valid=jnp.asarray(document_valid, dtype=jnp.bool_),
    )


def place_process_local_retrieval_batch(
    batch: ProcessLocalRetrievalBatch,
    sharding: NamedSharding,
) -> RetrievalBatch:
    """Assemble process-local retrieval rows with a leading-axis sharding."""

    def global_rows(tree: Any) -> Any:
        return jax.tree.map(
            lambda value: (
                jax.make_array_from_process_local_data(sharding, value)
                if eqx.is_array(value)
                else value
            ),
            tree,
            is_leaf=lambda value: value is None,
        )

    return RetrievalBatch(
        query=global_rows(batch.query),
        document=global_rows(batch.document),
        positive_mask=global_rows(batch.positive_mask),
        positive_weights=(
            None
            if batch.positive_weights is None
            else global_rows(batch.positive_weights)
        ),
        query_valid=global_rows(batch.query_valid),
        document_valid=global_rows(batch.document_valid),
    )


__all__ = [
    "ProcessLocalRetrievalBatch",
    "RetrievalBatch",
    "place_process_local_retrieval_batch",
    "process_local_retrieval_batch",
    "retrieval_batch",
]
