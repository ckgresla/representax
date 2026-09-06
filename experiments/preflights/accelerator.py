"""Small accelerator helpers shared by paper preflights."""

from __future__ import annotations

import os
from typing import Any, Literal

Platform = Literal["gpu", "tpu"]


def initialize_jax(platform: Platform) -> Any:
    import jax

    if platform == "tpu" and not jax.distributed.is_initialized():
        jax.distributed.initialize()
    if jax.default_backend() != platform:
        raise RuntimeError(
            f"requested {platform}, but JAX selected {jax.default_backend()}"
        )
    return jax


def data_parallel_job(
    job: Any,
    *,
    device_count: int,
    platform: Platform = "gpu",
    training_only: bool = False,
) -> Any:
    """Run fixed global work as replicated-state data parallelism."""

    from representax.config import BatchConfig, DDPConfig, MeshConfig

    global_batch = job.training.global_batch_size
    if global_batch % device_count:
        raise ValueError(
            f"global batch {global_batch} does not divide {device_count} devices"
        )
    local_batch = global_batch // device_count
    preferred_micro = min(job.training.batch.micro_batch_size, local_batch)
    while local_batch % preferred_micro:
        preferred_micro -= 1
    training = job.training.model_copy(
        update={
            "mesh": MeshConfig(
                axis_shapes=(device_count,),
                axis_names=("data",),
            ),
            "sharding": DDPConfig(axis="data"),
            "batch": BatchConfig(
                micro_batch_size=preferred_micro,
                gradient_accumulation_steps=local_batch // preferred_micro,
            ),
        }
    )
    logging = job.logging.model_copy(
        update={"accelerator": job.logging.accelerator and platform == "gpu"}
    )
    updates = {"training": training, "logging": logging}
    if training_only:
        from representax.config import ExportConfig

        updates.update(
            checkpointing=None,
            evaluation=None,
            export=ExportConfig(enabled=False),
        )
    return job.model_copy(update=updates)


def torch_is_tpu() -> bool:
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def torch_synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch_is_tpu():
        import torch_xla

        torch_xla.sync(wait=True)


def torch_reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def torch_empty_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def torch_device_report() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        return {
            "device": torch.cuda.get_device_name(),
            "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    if torch_is_tpu():
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        return {
            "device": str(xm.xla_device()),
            "device_count": int(xr.world_size()),
            "process_count": int(xr.process_count()),
            "peak_device_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "device": "cpu",
        "peak_device_bytes": None,
        "peak_reserved_bytes": None,
    }


def torch_rank() -> int:
    if not torch_is_tpu():
        return 0
    import torch_xla.runtime as xr

    return int(xr.global_ordinal())


def torch_world_size() -> int:
    if not torch_is_tpu():
        return 1
    import torch_xla.runtime as xr

    return int(xr.world_size())


__all__ = [
    "Platform",
    "data_parallel_job",
    "initialize_jax",
    "torch_device_report",
    "torch_empty_cache",
    "torch_is_tpu",
    "torch_rank",
    "torch_reset_peak_memory",
    "torch_synchronize",
    "torch_world_size",
]
