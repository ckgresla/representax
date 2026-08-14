"""Builders from serializable execution configs to JAX runtime objects."""

from __future__ import annotations

from representax.config import GradCacheConfig

from .execution import Direct, LossExecution
from .grad_cache import GradCache


def build_loss_execution(config: GradCacheConfig | None) -> LossExecution:
    """Construct direct or GradCache execution from the training configuration."""

    if config is None:
        return Direct()
    if isinstance(config, GradCacheConfig):
        return GradCache(
            query_chunk_size=config.resolved_query_micro_batch_size,
            document_chunk_size=config.resolved_document_micro_batch_size,
            loss_row_chunk_size=config.resolved_loss_row_chunk_size,
        )
    raise TypeError(f"unsupported GradCache config {type(config).__name__}")


__all__ = ["build_loss_execution"]
