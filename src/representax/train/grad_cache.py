"""Exact GradCache execution for representation-ranking objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray

from representax.core import (
    EncodeFunction,
    Encoder,
    LossOutput,
    Route,
    Scorer,
    Task,
    encode,
    encode_late_interaction,
)
from representax.core.sharding import (
    batch_to_scan,
    constrain_activation,
    scan_to_batch,
)
from representax.tasks.cross_encoder import CrossMNRBatch, CrossMNRTask
from representax.tasks.guided import GISTBatch, GISTTask
from representax.tasks.late_interaction import LateInteractionTask
from representax.tasks.modifiers import MatryoshkaTask
from representax.tasks.retrieval import MNRTask, RetrievalBatch

from .execution import _LOCAL_EXECUTION_CONTEXT, ExecutionContext


def _leading_batch_size(inputs: Any, *, role: str) -> int:
    leaves = [leaf for leaf in jax.tree.leaves(inputs) if eqx.is_array(leaf)]
    if not leaves:
        raise ValueError(f"{role} inputs must contain arrays")
    batch_size = leaves[0].shape[0]
    if batch_size == 0:
        raise ValueError(f"{role} inputs must contain at least one row")
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

    def chunk(leaf: Array) -> Array:
        # Repeat a real row rather than synthesizing an all-zero example. The
        # padded outputs are discarded either way, while an all-zero row can
        # have undefined model derivatives (for example through L2 norm at 0)
        # and poison the otherwise zero cotangent during replay.
        return batch_to_scan(
            leaf,
            local_chunk_size=chunk_size,
            pad_mode="edge",
        )

    return jax.tree.map(chunk, inputs)


def _rematerialized_encode(
    model: Any,
    inputs: Any,
    *,
    route: Route,
    batch_size: int,
    chunk_size: int,
    key: PRNGKeyArray | None,
    encode_fn: EncodeFunction = encode,
) -> Any:
    """Encode chunks while retaining only representations across the forward scan."""

    chunks = _pad_and_chunk(inputs, batch_size=batch_size, chunk_size=chunk_size)
    first_chunk = jax.tree.leaves(chunks)[0]
    chunk_count = first_chunk.shape[0]

    if key is None:

        def body(_: None, chunk: Any) -> tuple[None, Any]:
            return None, encode_fn(model, chunk, route=route)

        scan_inputs = chunks
    else:
        keys = jax.random.split(key, chunk_count)

        def body(
            _: None,
            values: tuple[Any, PRNGKeyArray],
        ) -> tuple[None, Any]:
            chunk, chunk_key = values
            return None, encode_fn(model, chunk, route=route, key=chunk_key)

        scan_inputs = (chunks, keys)

    rematerialized_body = jax.checkpoint(
        body,
        policy=jax.checkpoint_policies.nothing_saveable,
    )
    _, encoded_chunks = jax.lax.scan(rematerialized_body, None, scan_inputs)
    representations = jax.tree.map(
        lambda values: constrain_activation(
            scan_to_batch(
                values,
                batch_size=batch_size,
                local_chunk_size=chunk_size,
            )
        )[:batch_size],
        encoded_chunks,
    )
    return representations


def _gather_retrieval_rows(
    batch: RetrievalBatch,
    queries: Any,
    documents: Any,
    *,
    axis_name: str,
) -> tuple[Any, Any, RetrievalBatch]:
    def gather(tree: Any) -> Any:
        return jax.tree.map(
            lambda value: jax.lax.all_gather(
                value,
                axis_name,
                axis=0,
                tiled=True,
            ),
            tree,
        )

    queries = gather(queries)
    documents = gather(documents)
    return (
        queries,
        documents,
        RetrievalBatch(
            query=queries,
            document=documents,
            positive_mask=gather(batch.positive_mask),
            positive_weights=(
                None
                if batch.positive_weights is None
                else gather(batch.positive_weights)
            ),
            query_valid=gather(batch.query_valid),
            document_valid=gather(batch.document_valid),
        ),
    )


@dataclass(frozen=True)
class GradCache:
    """Bound encoder and score-row memory without changing loss semantics."""

    query_chunk_size: int
    document_chunk_size: int | None = None
    loss_row_chunk_size: int | None = None
    score_chunk_size: int | None = None

    def __post_init__(self) -> None:
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.document_chunk_size is not None and self.document_chunk_size <= 0:
            raise ValueError("document_chunk_size must be positive when set")
        if self.loss_row_chunk_size is not None and self.loss_row_chunk_size <= 0:
            raise ValueError("loss_row_chunk_size must be positive when set")
        if self.score_chunk_size is not None and self.score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive when set")

    @property
    def resolved_document_chunk_size(self) -> int:
        return self.document_chunk_size or self.query_chunk_size

    @property
    def resolved_loss_row_chunk_size(self) -> int:
        return self.loss_row_chunk_size or self.query_chunk_size

    @property
    def resolved_score_chunk_size(self) -> int:
        return self.score_chunk_size or self.query_chunk_size

    def validate(self, task: Task[Any]) -> None:
        base_task = task.task if isinstance(task, MatryoshkaTask) else task
        if isinstance(task, MatryoshkaTask) and isinstance(
            base_task, LateInteractionTask
        ):
            raise TypeError("Matryoshka does not apply to token-level representations")
        if not isinstance(
            base_task, (MNRTask, GISTTask, LateInteractionTask, CrossMNRTask)
        ):
            raise TypeError(
                "GradCache requires MNRTask, GISTTask, LateInteractionTask, "
                "CrossMNRTask, "
                "or a supported representation modifier"
            )

    def evaluate(
        self,
        task: Task[Any],
        model: eqx.Module,
        batch: Any,
        *,
        key: PRNGKeyArray | None,
        context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    ) -> LossOutput:
        modifier = task if isinstance(task, MatryoshkaTask) else None
        base_task = modifier.task if modifier is not None else task
        if isinstance(base_task, CrossMNRTask) and isinstance(batch, CrossMNRBatch):
            if modifier is not None:
                raise TypeError("representation modifiers do not apply to scorers")
            return base_task.loss_with_chunk_size(
                cast(Scorer, model),
                batch,
                key=key,
                chunk_size=self.resolved_score_chunk_size,
            )
        if modifier is None:
            representation_key = key
            modifier_key = None
        elif modifier.dimensions_per_step == -1 or modifier.dimensions_per_step >= len(
            modifier.dimensions
        ):
            representation_key = modifier_key = key
        elif key is None:
            raise ValueError("random Matryoshka sampling requires a JAX key")
        else:
            representation_key, modifier_key = jax.random.split(key)

        if isinstance(base_task, GISTTask) and isinstance(batch, GISTBatch):
            if context.data_axis_name is not None:
                raise NotImplementedError(
                    "distributed cached GIST is not implemented yet"
                )
            encoder = cast(Encoder, model)

            def cached_encode(
                candidate: Encoder,
                inputs: Any,
                *,
                route: Route,
                key: PRNGKeyArray | None = None,
            ) -> Float[Array, "batch representation"]:
                batch_size = _leading_batch_size(inputs, role=route.value)
                chunk_size = (
                    self.query_chunk_size
                    if route == Route.QUERY
                    else self.resolved_document_chunk_size
                )
                return _rematerialized_encode(
                    candidate,
                    inputs,
                    route=route,
                    batch_size=batch_size,
                    chunk_size=chunk_size,
                    key=key,
                )

            representations = base_task.representations(
                encoder,
                batch,
                key=representation_key,
                encode_fn=cached_encode,
            )
            if modifier is not None:
                return modifier.loss_from_representations(
                    representations,
                    batch,
                    key=modifier_key,
                    row_chunk_size=self.resolved_loss_row_chunk_size,
                )
            return base_task.loss_from_representations(
                representations, batch, row_chunk_size=self.resolved_loss_row_chunk_size
            )
        if isinstance(base_task, LateInteractionTask) and isinstance(
            batch, RetrievalBatch
        ):
            if modifier is not None:  # validate() rejects this before tracing.
                raise AssertionError("late interaction modifier passed validation")
            axis_name = context.data_axis_name
            if axis_name is not None and base_task.negative_scope != "global":
                raise NotImplementedError(
                    "distributed GradCache currently implements global negatives only"
                )
            if axis_name is not None and representation_key is not None:
                representation_key = jax.random.fold_in(
                    representation_key,
                    jax.lax.axis_index(axis_name),
                )
            if representation_key is None:
                query_key = document_key = None
            else:
                query_key, document_key = jax.random.split(representation_key)
            query_count = _leading_batch_size(batch.query, role="query")
            document_count = _leading_batch_size(batch.document, role="document")
            queries = _rematerialized_encode(
                model,
                batch.query,
                route=Route.QUERY,
                batch_size=query_count,
                chunk_size=self.query_chunk_size,
                key=query_key,
                encode_fn=encode_late_interaction,
            )
            documents = _rematerialized_encode(
                model,
                batch.document,
                route=Route.DOCUMENT,
                batch_size=document_count,
                chunk_size=self.resolved_document_chunk_size,
                key=document_key,
                encode_fn=encode_late_interaction,
            )
            if axis_name is not None:
                queries, documents, batch = _gather_retrieval_rows(
                    batch,
                    queries,
                    documents,
                    axis_name=axis_name,
                )
            output = base_task.loss_from_representations(
                (queries, documents),
                batch,
                row_chunk_size=self.resolved_loss_row_chunk_size,
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
        if not isinstance(base_task, MNRTask) or not isinstance(batch, RetrievalBatch):
            raise TypeError(
                "GradCache requires MNR/Retrieval or GIST/GuidedRetrieval contracts"
            )
        axis_name = context.data_axis_name
        if axis_name is not None and base_task.negative_scope != "global":
            raise NotImplementedError(
                "distributed GradCache currently implements global negatives only"
            )
        if axis_name is not None and representation_key is not None:
            representation_key = jax.random.fold_in(
                representation_key,
                jax.lax.axis_index(axis_name),
            )
        if representation_key is None:
            query_key = document_key = None
        else:
            query_key, document_key = jax.random.split(representation_key)
        encoder = cast(Encoder, model)
        query_count = _leading_batch_size(batch.query, role="query")
        document_count = _leading_batch_size(batch.document, role="document")
        queries = _rematerialized_encode(
            encoder,
            batch.query,
            route=Route.QUERY,
            batch_size=query_count,
            chunk_size=self.query_chunk_size,
            key=query_key,
        )
        documents = _rematerialized_encode(
            encoder,
            batch.document,
            route=Route.DOCUMENT,
            batch_size=document_count,
            chunk_size=self.resolved_document_chunk_size,
            key=document_key,
        )
        if axis_name is not None:
            queries, documents, batch = _gather_retrieval_rows(
                batch,
                queries,
                documents,
                axis_name=axis_name,
            )
        if modifier is None:
            output = base_task.loss_from_embeddings(
                queries,
                documents,
                batch,
                row_chunk_size=self.resolved_loss_row_chunk_size,
            )
        else:
            output = modifier.loss_from_representations(
                (queries, documents),
                batch,
                key=modifier_key,
                row_chunk_size=self.resolved_loss_row_chunk_size,
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
