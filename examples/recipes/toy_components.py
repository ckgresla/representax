"""Importable preprocessing components for the toy retrieval recipe."""

from collections.abc import Sequence

import jax.numpy as jnp

from representax.tasks.retrieval import retrieval_batch


def to_features(record):
    """Keep the already canonical toy record unchanged."""

    return record


def collate(examples: Sequence[dict]):
    """Turn mapped records into the fixed-shape retrieval batch contract."""

    size = len(examples)
    return retrieval_batch(
        query=jnp.asarray([example["query"] for example in examples]),
        document=jnp.asarray([example["document"] for example in examples]),
        positive_mask=jnp.eye(size, dtype=jnp.bool_),
    )


__all__ = ["collate", "to_features"]
