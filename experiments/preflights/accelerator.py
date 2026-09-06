"""Small accelerator helpers shared by paper preflights."""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib import import_module
from typing import Any, Literal, TypeVar

Platform = Literal["gpu", "tpu"]
Row = TypeVar("Row")


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
    configured = job.model_copy(update={"training": training, "logging": logging})
    return (
        training_only_job(configured, platform=platform)
        if training_only
        else configured
    )


def process_local_rows(rows: Sequence[Row]) -> tuple[Sequence[Row], int, int]:
    """Select this JAX process's contiguous share of one global batch."""

    import jax

    process_count = jax.process_count()
    if len(rows) % process_count:
        raise ValueError(
            f"global row count {len(rows)} does not divide {process_count} processes"
        )
    local_size = len(rows) // process_count
    start = jax.process_index() * local_size
    return rows[start : start + local_size], start, len(rows)


def training_only_job(job: Any, *, platform: Platform) -> Any:
    """Disable lifecycle work while preserving an experiment's sharding plan."""

    from representax.config import ExportConfig

    logging = job.logging.model_copy(
        update={"accelerator": job.logging.accelerator and platform == "gpu"}
    )
    return job.model_copy(
        update={
            "checkpointing": None,
            "evaluation": None,
            "export": ExportConfig(enabled=False),
            "logging": logging,
        }
    )


def torch_is_tpu() -> bool:
    return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


def torch_device() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch_is_tpu():
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return torch.device("cpu")


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


def install_torch_xla_checkpointing() -> None:
    """Route Transformers rematerialization through PyTorch/XLA's implementation."""

    modeling_utils = import_module("transformers.modeling_utils")
    modeling_utils.checkpoint = import_module("torch_xla.utils.checkpoint").checkpoint


def deterministic_tpu_cached_mnr(
    loss_type: type[Any], model: Any, **options: Any
) -> Any:
    """Use cached MNR on TPU when replay does not require RNG restoration."""

    import torch

    active_dropout = {
        name: float(module.p)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Dropout) and float(module.p) != 0.0
    }
    config = getattr(getattr(model[0], "auto_model", None), "config", None)

    def visit(value: Any, path: str = "config") -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                child_path = f"{path}.{name}"
                if "dropout" in name and isinstance(child, (float, int)) and child:
                    active_dropout[child_path] = float(child)
                else:
                    visit(child, child_path)
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    if config is not None:
        visit(config.to_dict())
    if active_dropout:
        raise RuntimeError(
            "TPU cached MNR requires deterministic replay; active dropout: "
            f"{active_dropout}"
        )

    class DeterministicTPUCachedMNR(loss_type):
        def embed_minibatch(
            self,
            sentence_feature: Any,
            begin: int,
            end: int,
            with_grad: bool,
            copy_random_state: bool,
            random_state: Any = None,
        ) -> Any:
            del copy_random_state
            if random_state is not None:
                raise RuntimeError("deterministic TPU replay received RNG state")
            return super().embed_minibatch(
                sentence_feature,
                begin,
                end,
                with_grad,
                False,
                None,
            )

    return DeterministicTPUCachedMNR(model, **options)


def use_fixed_text_padding(model: Any, maximum_length: int) -> None:
    """Configure a Sentence Transformers input module for one static XLA shape."""

    if maximum_length <= 0:
        raise ValueError("maximum text length must be positive")
    module = model[0]
    current = dict(getattr(module, "processing_kwargs", {}))
    text = dict(current.get("text", {}))
    text.update(
        padding="max_length",
        truncation=True,
        max_length=maximum_length,
    )
    module.processing_kwargs = {**current, "text": text}


__all__ = [
    "Platform",
    "data_parallel_job",
    "deterministic_tpu_cached_mnr",
    "initialize_jax",
    "install_torch_xla_checkpointing",
    "process_local_rows",
    "torch_device",
    "torch_device_report",
    "torch_empty_cache",
    "torch_is_tpu",
    "torch_rank",
    "torch_reset_peak_memory",
    "torch_synchronize",
    "torch_world_size",
    "training_only_job",
    "use_fixed_text_padding",
]
