"""Pickle-stable benchmark-only sampler factories."""

from __future__ import annotations

from typing import Any


def sequential_sentence_transformers_batches(
    dataset: Any,
    batch_size: int,
    drop_last: bool,
    valid_label_columns: list[str] | None = None,
    generator: Any = None,
    seed: int = 0,
) -> Any:
    """Build canonical batches over the dataset's already-fixed row order."""

    del generator
    from sentence_transformers.base.sampler import DefaultBatchSampler
    from torch.utils.data import SequentialSampler

    return DefaultBatchSampler(
        SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=valid_label_columns,
        seed=seed,
    )


__all__ = ["sequential_sentence_transformers_batches"]
