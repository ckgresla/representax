"""GISTEmbed false-negative filtering over student and guide representations."""

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class GISTLossTerms(eqx.Module):
    """GIST objective and auditable row-level filtering statistics."""

    loss: Float[Array, ""]
    row_losses: Float[Array, " batch"]
    masked_candidates: Array


def _normalize(values: Float[Array, "batch representation"]) -> Array:
    values = values.astype(jnp.float32)
    return values / jnp.maximum(
        jnp.linalg.norm(values, axis=-1, keepdims=True),
        jnp.asarray(1e-12, values.dtype),
    )


def _validate(
    embeddings: tuple[Array, ...],
    guide_embeddings: tuple[Array, ...],
    valid: Bool[Array, " batch"],
    *,
    temperature: float,
    margin_strategy: str,
    margin: float,
) -> int:
    if len(embeddings) < 2 or len(embeddings) != len(guide_embeddings):
        raise ValueError("GIST requires aligned anchor, positive, and guide columns")
    batch_size = embeddings[0].shape[0]
    if any(
        value.ndim != 2
        or value.shape[0] != batch_size
        or not jnp.issubdtype(value.dtype, jnp.floating)
        for value in embeddings
    ):
        raise TypeError("GIST student embeddings must be aligned floating matrices")
    if any(
        value.ndim != 2
        or value.shape[0] != batch_size
        or not jnp.issubdtype(value.dtype, jnp.floating)
        for value in guide_embeddings
    ):
        raise TypeError("GIST guide embeddings must be aligned floating matrices")
    if any(value.shape[-1] != embeddings[0].shape[-1] for value in embeddings):
        raise ValueError("GIST student embeddings must share their dimension")
    if any(
        value.shape[-1] != guide_embeddings[0].shape[-1] for value in guide_embeddings
    ):
        raise ValueError("GIST guide embeddings must share their dimension")
    if valid.shape != (batch_size,) or valid.dtype != jnp.bool_:
        raise TypeError("GIST valid must be a boolean vector matching batch rows")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("GIST temperature must be finite and positive")
    if margin_strategy not in {"absolute", "relative"}:
        raise ValueError("GIST margin_strategy must be 'absolute' or 'relative'")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("GIST margin must be finite and non-negative")
    return batch_size


def _gist_rows(
    anchor_rows: Array,
    positive_rows: Array,
    guide_anchor_rows: Array,
    guide_positive_rows: Array,
    row_indices: Array,
    row_valid: Array,
    embeddings: tuple[Array, ...],
    guide_embeddings: tuple[Array, ...],
    candidate_valid: Array,
    *,
    temperature: float,
    margin_strategy: Literal["absolute", "relative"],
    margin: float,
    contrast_anchors: bool,
    contrast_positives: bool,
) -> tuple[Array, Array]:
    anchor, positive, *negatives = embeddings
    guide_anchor, guide_positive, *guide_negatives = guide_embeddings

    guided_positive_scores = jnp.sum(
        guide_anchor_rows * guide_positive_rows,
        axis=-1,
        keepdims=True,
    )
    threshold = (
        guided_positive_scores - margin
        if margin_strategy == "absolute"
        else guided_positive_scores - jnp.abs(guided_positive_scores) * margin
    )

    def score_block(rows, candidates, guide_rows, guide_candidates, *, protect):
        scores = rows @ candidates.T
        guide_scores = guide_rows @ guide_candidates.T
        masked = guide_scores > threshold
        if protect:
            positive_mask = (
                jnp.arange(candidates.shape[0])[None, :] == row_indices[:, None]
            )
            masked = masked & ~positive_mask
        masked = masked | ~candidate_valid[None, :]
        return jnp.where(masked, -jnp.inf, scores), jnp.sum(masked)

    score_blocks = []
    masked_count = jnp.asarray(0, dtype=jnp.int32)
    block, count = score_block(
        anchor_rows,
        positive,
        guide_anchor_rows,
        guide_positive,
        protect=True,
    )
    score_blocks.append(block)
    masked_count = masked_count + count
    if contrast_anchors:
        block, count = score_block(
            anchor_rows,
            anchor,
            guide_anchor_rows,
            guide_anchor,
            protect=False,
        )
        score_blocks.append(block)
        masked_count = masked_count + count
    if contrast_positives:
        block, count = score_block(
            positive_rows,
            positive,
            guide_positive_rows,
            guide_positive,
            protect=False,
        )
        score_blocks.append(block)
        masked_count = masked_count + count
    for negative, guide_negative in zip(negatives, guide_negatives, strict=True):
        block, count = score_block(
            anchor_rows,
            negative,
            guide_anchor_rows,
            guide_negative,
            protect=False,
        )
        score_blocks.append(block)
        masked_count = masked_count + count

    logits = jnp.concatenate(score_blocks, axis=1) / temperature
    logits = jnp.where(row_valid[:, None], logits, 0.0)
    row_losses = -jax.nn.log_softmax(logits, axis=1)[
        jnp.arange(logits.shape[0]), row_indices
    ]
    return jnp.where(row_valid, row_losses, 0.0), masked_count


def gist_loss_terms(
    embeddings: tuple[Float[Array, "batch representation"], ...],
    guide_embeddings: tuple[Float[Array, "batch guide"], ...],
    *,
    valid: Bool[Array, " batch"] | None = None,
    temperature: float = 0.01,
    margin_strategy: Literal["absolute", "relative"] = "absolute",
    margin: float = 0.0,
    contrast_anchors: bool = True,
    contrast_positives: bool = True,
    row_chunk_size: int | None = None,
) -> GISTLossTerms:
    """Compute direct or score-row-chunked GIST with identical semantics."""

    if not embeddings:
        raise ValueError("GIST embeddings must be non-empty")
    batch_size = embeddings[0].shape[0]
    resolved_valid = (
        jnp.ones((batch_size,), dtype=jnp.bool_)
        if valid is None
        else jnp.asarray(valid, dtype=jnp.bool_)
    )
    _validate(
        embeddings,
        guide_embeddings,
        resolved_valid,
        temperature=temperature,
        margin_strategy=margin_strategy,
        margin=margin,
    )
    normalized_embeddings = tuple(_normalize(value) for value in embeddings)
    normalized_guide_embeddings = tuple(_normalize(value) for value in guide_embeddings)
    chunk_size = batch_size if row_chunk_size is None else row_chunk_size
    if chunk_size <= 0:
        raise ValueError("GIST row_chunk_size must be positive")
    chunk_count = (batch_size + chunk_size - 1) // chunk_size
    padded_size = chunk_count * chunk_size
    padding = padded_size - batch_size

    def chunks(value: Array) -> Array:
        widths = ((0, padding),) + ((0, 0),) * (value.ndim - 1)
        return jnp.pad(value, widths).reshape(
            chunk_count,
            chunk_size,
            *value.shape[1:],
        )

    row_indices = jnp.pad(jnp.arange(batch_size), (0, padding)).reshape(
        chunk_count, chunk_size
    )
    row_valid = chunks(resolved_valid)
    anchor_chunks = chunks(normalized_embeddings[0])
    positive_chunks = chunks(normalized_embeddings[1])
    guide_anchor_chunks = chunks(normalized_guide_embeddings[0])
    guide_positive_chunks = chunks(normalized_guide_embeddings[1])

    def body(totals, values):
        rows, positives, guide_rows, guide_positives, indices, valid_rows = values
        row_losses, masked = _gist_rows(
            rows,
            positives,
            guide_rows,
            guide_positives,
            indices,
            valid_rows,
            normalized_embeddings,
            normalized_guide_embeddings,
            resolved_valid,
            temperature=temperature,
            margin_strategy=margin_strategy,
            margin=margin,
            contrast_anchors=contrast_anchors,
            contrast_positives=contrast_positives,
        )
        loss_sum, valid_count, masked_count = totals
        return (
            loss_sum + jnp.sum(row_losses),
            valid_count + jnp.sum(valid_rows),
            masked_count + masked,
        ), row_losses

    initial = (
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    (loss_sum, valid_count, masked_count), row_loss_chunks = jax.lax.scan(
        jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable),
        initial,
        (
            anchor_chunks,
            positive_chunks,
            guide_anchor_chunks,
            guide_positive_chunks,
            row_indices,
            row_valid,
        ),
    )
    return GISTLossTerms(
        loss=loss_sum / jnp.maximum(valid_count, 1).astype(jnp.float32),
        row_losses=row_loss_chunks.reshape(-1)[:batch_size],
        masked_candidates=masked_count,
    )
