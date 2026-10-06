"""Host-side fixed-count token masking for static-shape MLM batches."""

from __future__ import annotations

import math

import numpy as np


def mask_tokens(
    input_ids: np.ndarray,
    *,
    attention_mask: np.ndarray,
    special_tokens_mask: np.ndarray,
    vocabulary_size: int,
    mask_token_id: int,
    seed: int,
    probability: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corrupted IDs, target positions and labels with 80/10/10 corruption.

    Select ceil(probability * eligible_count) positions per nonempty row without
    replacement. This is fixed-count masking, not independent Bernoulli masking.
    Targets are padded to ceil(probability * sequence_length) with label -100.
    The caller derives a stable seed per batch/example and owns tokenizer IDs.
    """
    ids = np.asarray(input_ids)
    attention = np.asarray(attention_mask, dtype=bool)
    special = np.asarray(special_tokens_mask, dtype=bool)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("input_ids must be a two-dimensional integer array")
    if attention.shape != ids.shape or special.shape != ids.shape:
        raise ValueError("token masks must match input_ids")
    if not 0 < probability <= 1 or not math.isfinite(probability):
        raise ValueError("mask probability must be in (0, 1]")
    if vocabulary_size < 1 or not 0 <= mask_token_id < vocabulary_size:
        raise ValueError("mask token must be inside a nonempty vocabulary")
    if np.any((ids < 0) | (ids >= vocabulary_size)):
        raise ValueError("input token outside vocabulary")
    count = math.ceil(probability * ids.shape[1])
    positions = np.zeros((ids.shape[0], count), dtype=np.int32)
    labels = np.full(positions.shape, -100, dtype=np.int32)
    corrupted = ids.copy()
    rng = np.random.default_rng(seed)
    for row in range(ids.shape[0]):
        eligible = np.flatnonzero(attention[row] & ~special[row])
        selected = rng.choice(
            eligible, size=math.ceil(probability * len(eligible)), replace=False
        )
        size = len(selected)
        positions[row, :size] = selected
        labels[row, :size] = ids[row, selected]
        draw = rng.random(size)
        corrupted[row, selected[draw < 0.8]] = mask_token_id
        random_positions = selected[(draw >= 0.8) & (draw < 0.9)]
        corrupted[row, random_positions] = rng.integers(
            vocabulary_size, size=len(random_positions)
        )
    return corrupted, positions, labels
