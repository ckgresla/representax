"""Builders from serializable execution configs to JAX runtime objects."""

from __future__ import annotations

from representax.config import GradCacheConfig, MegaBatchMiningConfig

from .execution import Direct, LossExecution
from .grad_cache import GradCache
from .mega_batch import MegaBatchMining


def build_loss_execution(
    config: GradCacheConfig | None,
    *,
    mega_batch_mining: MegaBatchMiningConfig | None = None,
) -> LossExecution:
    """Construct direct or GradCache execution from the training configuration."""

    if config is not None and mega_batch_mining is not None:
        raise ValueError("configure only one specialized loss execution")
    if config is None and mega_batch_mining is None:
        return Direct()
    if isinstance(config, GradCacheConfig):
        return GradCache(
            query_chunk_size=config.resolved_query_micro_batch_size,
            document_chunk_size=config.resolved_document_micro_batch_size,
            loss_row_chunk_size=config.resolved_loss_row_chunk_size,
        )
    if isinstance(mega_batch_mining, MegaBatchMiningConfig):
        return MegaBatchMining(
            micro_batch_size=mega_batch_mining.micro_batch_size,
            loss_row_chunk_size=mega_batch_mining.resolved_loss_row_chunk_size,
        )
    invalid = config if config is not None else mega_batch_mining
    raise TypeError(f"unsupported loss execution config {type(invalid).__name__}")


__all__ = ["build_loss_execution"]
