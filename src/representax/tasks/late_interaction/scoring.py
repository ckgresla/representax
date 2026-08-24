"""Exact pure-JAX late-interaction scoring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from representax.core import LateInteractionRepresentation


def _maxsim_document_block(
    queries: LateInteractionRepresentation,
    documents: LateInteractionRepresentation,
) -> Float[Array, "query document"]:
    similarities = jnp.einsum(
        "qtd,nsd->qnts",
        queries.values,
        documents.values,
        precision=jax.lax.Precision.HIGHEST,
    )
    similarities = jnp.where(
        documents.valid[None, :, None, :],
        similarities,
        -jnp.inf,
    )
    token_scores = jnp.max(similarities, axis=-1)
    has_document_token = jnp.any(documents.valid, axis=-1)
    token_scores = jnp.where(
        has_document_token[None, :, None],
        token_scores,
        0.0,
    )
    return jnp.sum(
        jnp.where(queries.valid[:, None, :], token_scores, 0.0),
        axis=-1,
        dtype=jnp.float32,
    )


def maxsim_scores(
    queries: LateInteractionRepresentation,
    documents: LateInteractionRepresentation,
    *,
    document_chunk_size: int | None = None,
) -> Float[Array, "query document"]:
    """Compute exact ColBERT MaxSim, optionally retaining one document tile."""

    if queries.values.shape[-1] != documents.values.shape[-1]:
        raise ValueError("query and document token dimensions must match")
    if document_chunk_size is None:
        return _maxsim_document_block(queries, documents)
    if document_chunk_size <= 0:
        raise ValueError("document_chunk_size must be positive when set")

    document_count = documents.values.shape[0]
    chunk_count = (document_count + document_chunk_size - 1) // document_chunk_size
    padded_count = chunk_count * document_chunk_size
    padding = padded_count - document_count
    values = jnp.pad(documents.values, ((0, padding), (0, 0), (0, 0)))
    valid = jnp.pad(documents.valid, ((0, padding), (0, 0)))
    value_chunks = values.reshape(
        chunk_count,
        document_chunk_size,
        *values.shape[1:],
    )
    valid_chunks = valid.reshape(chunk_count, document_chunk_size, valid.shape[1])

    def body(_: None, block: tuple[Array, Array]):
        block_values, block_valid = block
        return None, _maxsim_document_block(
            queries,
            LateInteractionRepresentation(values=block_values, valid=block_valid),
        )

    _, score_chunks = jax.lax.scan(
        jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable),
        None,
        (value_chunks, valid_chunks),
    )
    return jnp.transpose(score_chunks, (1, 0, 2)).reshape(
        queries.values.shape[0], padded_count
    )[:, :document_count]


__all__ = ["maxsim_scores"]
