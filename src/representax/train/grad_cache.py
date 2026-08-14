"""Exact GradCache execution for the canonical MNR objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from representax.core import Encoder, LossOutput, Route, Task, encode
from representax.tasks.retrieval import MNRTask, RetrievalBatch

from .execution import _LOCAL_EXECUTION_CONTEXT, ExecutionContext


def _leading_batch_size(inputs: Any, *, role: str) -> int:
    leaves = [leaf for leaf in jax.tree.leaves(inputs) if eqx.is_array(leaf)]
    if not leaves:
        raise ValueError(f"{role} inputs must contain arrays")
    batch_size = leaves[0].shape[0]
    if any(leaf.ndim == 0 or leaf.shape[0] != batch_size for leaf in leaves):
        raise ValueError(f"{role} inputs must be row-major")
    return batch_size


def _pad_and_chunk(inputs: Any, *, batch_size: int, chunk_size: int) -> Any:
    leaves = jax.tree.leaves(inputs)
    if not leaves:
        raise ValueError("GradCache inputs must contain arrays")
    if any(not eqx.is_array(leaf) for leaf in leaves):
        raise TypeError("GradCache inputs must be array PyTrees")
    if any(leaf.ndim == 0 or leaf.shape[0] != batch_size for leaf in leaves):
        raise ValueError("GradCache inputs must be row-major")

    chunk_count = (batch_size + chunk_size - 1) // chunk_size
    padded_size = chunk_count * chunk_size
    padding = padded_size - batch_size

    def chunk(leaf: jax.Array) -> jax.Array:
        widths = ((0, padding),) + ((0, 0),) * (leaf.ndim - 1)
        padded = jnp.pad(leaf, widths)
        return padded.reshape((chunk_count, chunk_size, *leaf.shape[1:]))

    return jax.tree.map(chunk, inputs)


def _rematerialized_encode(
    model: Encoder,
    inputs: Any,
    *,
    route: Route,
    batch_size: int,
    chunk_size: int,
    key: jax.Array | None,
) -> jax.Array:
    """Encode chunks while retaining only representations across the forward scan."""

    chunks = _pad_and_chunk(inputs, batch_size=batch_size, chunk_size=chunk_size)
    chunk_count = (batch_size + chunk_size - 1) // chunk_size

    if key is None:

        def body(_: None, chunk: Any) -> tuple[None, jax.Array]:
            return None, encode(model, chunk, route=route)

        scan_inputs = chunks
    else:
        keys = jax.random.split(key, chunk_count)

        def body(_: None, values: tuple[Any, jax.Array]) -> tuple[None, jax.Array]:
            chunk, chunk_key = values
            return None, encode(model, chunk, route=route, key=chunk_key)

        scan_inputs = (chunks, keys)

    rematerialized_body = jax.checkpoint(
        body,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    _, encoded_chunks = jax.lax.scan(rematerialized_body, None, scan_inputs)
    embeddings = encoded_chunks.reshape((-1, encoded_chunks.shape[-1]))
    return embeddings[:batch_size]


@dataclass(frozen=True)
class GradCache:
    """Bound encoder-activation memory without changing MNR semantics."""

    query_chunk_size: int
    document_chunk_size: int | None = None
    representation_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.document_chunk_size is not None and self.document_chunk_size <= 0:
            raise ValueError("document_chunk_size must be positive when set")
        if (
            self.representation_chunk_size is not None
            and self.representation_chunk_size <= 0
        ):
            raise ValueError("representation_chunk_size must be positive when set")

    @property
    def resolved_document_chunk_size(self) -> int:
        return self.document_chunk_size or self.query_chunk_size

    @property
    def resolved_representation_chunk_size(self) -> int:
        return self.representation_chunk_size or self.query_chunk_size

    def validate(self, task: Task[Any]) -> None:
        if not isinstance(task, MNRTask):
            raise TypeError("GradCache currently requires MNRTask")

    def evaluate(
        self,
        task: Task[Any],
        model: eqx.Module,
        batch: Any,
        *,
        key: jax.Array | None,
        context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    ) -> LossOutput:
        if not isinstance(task, MNRTask) or not isinstance(batch, RetrievalBatch):
            raise TypeError("GradCache requires MNRTask and RetrievalBatch")
        axis_name = context.data_axis_name
        if axis_name is not None and key is not None:
            key = jax.random.fold_in(key, jax.lax.axis_index(axis_name))
        if key is None:
            query_key = document_key = None
        else:
            query_key, document_key = jax.random.split(key)
        query_count = _leading_batch_size(batch.query, role="query")
        document_count = _leading_batch_size(batch.document, role="document")
        queries = _rematerialized_encode(
            model,
            batch.query,
            route=Route.QUERY,
            batch_size=query_count,
            chunk_size=self.query_chunk_size,
            key=query_key,
        )
        documents = _rematerialized_encode(
            model,
            batch.document,
            route=Route.DOCUMENT,
            batch_size=document_count,
            chunk_size=self.resolved_document_chunk_size,
            key=document_key,
        )
        if axis_name is not None:
            queries = jax.lax.all_gather(
                queries,
                axis_name,
                axis=0,
                tiled=True,
            )
            documents = jax.lax.all_gather(
                documents,
                axis_name,
                axis=0,
                tiled=True,
            )
            positive_mask = jax.lax.all_gather(
                batch.positive_mask,
                axis_name,
                axis=0,
                tiled=True,
            )
            positive_weights = (
                None
                if batch.positive_weights is None
                else jax.lax.all_gather(
                    batch.positive_weights,
                    axis_name,
                    axis=0,
                    tiled=True,
                )
            )
            batch = RetrievalBatch(
                query=queries,
                document=documents,
                positive_mask=positive_mask,
                positive_weights=positive_weights,
                query_valid=jax.lax.all_gather(
                    batch.query_valid,
                    axis_name,
                    axis=0,
                    tiled=True,
                ),
                document_valid=jax.lax.all_gather(
                    batch.document_valid,
                    axis_name,
                    axis=0,
                    tiled=True,
                ),
            )
        output = task.loss_from_embeddings(
            queries,
            documents,
            batch,
            row_chunk_size=self.resolved_representation_chunk_size,
        )
        if axis_name is None:
            return output
        return LossOutput(
            loss=jax.lax.pmean(output.loss, axis_name),
            metrics=jax.tree.map(
                lambda value: jax.lax.pmean(value, axis_name),
                output.metrics,
            ),
        )
