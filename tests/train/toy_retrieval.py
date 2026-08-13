"""Deterministic nontrivial retrieval data for runtime acceptance tests."""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax.numpy as jnp

from representax.data import build_grain_iterator, mix, source
from representax.tasks.retrieval import retrieval_batch

TOY_BATCH_SIZE = 16
TOY_FEATURE_DIMENSION = 16
TOY_OUTPUT_DIMENSION = 8
TOY_STEPS = 8


def identity(record):
    return record


def toy_retrieval_records() -> list[dict[str, list[float]]]:
    """Build paired semantic views with independent, learnable nuisance features."""

    records = []
    for index in range(TOY_BATCH_SIZE * TOY_STEPS):
        concept = index % TOY_BATCH_SIZE
        view = index // TOY_BATCH_SIZE
        angle = 2.0 * math.pi * (concept + 0.5) / TOY_BATCH_SIZE
        semantic = [
            function(frequency * angle)
            for frequency in range(1, 5)
            for function in (math.sin, math.cos)
        ]
        query_nuisance = [
            1.5 * math.sin((concept + 1) * (view + 1) * (axis + 1) * 0.37)
            for axis in range(8)
        ]
        document_nuisance = [
            1.5 * math.cos((concept + 3) * (view + 2) * (axis + 1) * 0.29)
            for axis in range(8)
        ]
        records.append(
            {
                "query": semantic + query_nuisance,
                "document": semantic + document_nuisance,
            }
        )
    return records


def resolve_toy_retrieval(_artifact):
    return toy_retrieval_records()


def collate_retrieval(examples: Sequence[dict]):
    size = len(examples)
    return retrieval_batch(
        query=jnp.asarray([example["query"] for example in examples]),
        document=jnp.asarray([example["document"] for example in examples]),
        positive_mask=jnp.eye(size, dtype=jnp.bool_),
    )


def build_toy_retrieval_batches(*, seed: int = 23):
    artifact = source("memory://nontrivial-retrieval", map=identity)
    return build_grain_iterator(
        mix(artifact, shuffle=False, seed=seed),
        batch_size=TOY_BATCH_SIZE,
        batch_fn=collate_retrieval,
        num_threads=1,
        prefetch_buffer_size=1,
        resolvers={"memory": resolve_toy_retrieval},
        mappers={artifact.mapper: identity},
    )


__all__ = [
    "TOY_BATCH_SIZE",
    "TOY_FEATURE_DIMENSION",
    "TOY_OUTPUT_DIMENSION",
    "TOY_STEPS",
    "build_toy_retrieval_batches",
    "toy_retrieval_records",
]
