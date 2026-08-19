"""Profile isolated DDP or FSDP ModernVBERT training on one GPU host."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablehlo-output", type=Path)
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), required=True)
    parser.add_argument(
        "--materialization-boundary",
        choices=("model", "layer"),
        default="layer",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--physical-device-ids", default="4,5")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _array_bytes(value: Any) -> int:
    return int(value.size * np.dtype(value.dtype).itemsize)


def _layout_summary(tree: Any) -> dict[str, Any]:
    import equinox as eqx
    import jax

    arrays = [
        (jax.tree_util.keystr(path), value)
        for path, value in jax.tree.flatten_with_path(tree)[0]
        if eqx.is_array(value)
    ]
    per_device: Counter[str] = Counter()
    specs: Counter[str] = Counter()
    for _, value in arrays:
        specs[str(getattr(value.sharding, "spec", None))] += 1
        for shard in value.addressable_shards:
            per_device[str(shard.device)] += _array_bytes(shard.data)
    largest = []
    for path, value in sorted(
        arrays,
        key=lambda item: _array_bytes(item[1]),
        reverse=True,
    )[:8]:
        largest.append(
            {
                "path": path,
                "global_shape": list(value.shape),
                "dtype": str(value.dtype),
                "partition_spec": str(getattr(value.sharding, "spec", None)),
                "global_bytes": _array_bytes(value),
                "addressable_shards": [
                    {
                        "device": str(shard.device),
                        "shape": list(shard.data.shape),
                        "bytes": _array_bytes(shard.data),
                    }
                    for shard in value.addressable_shards
                ],
            }
        )
    return {
        "array_count": len(arrays),
        "logical_global_bytes": sum(_array_bytes(value) for _, value in arrays),
        "addressable_bytes_by_device": dict(sorted(per_device.items())),
        "partition_spec_counts": dict(sorted(specs.items())),
        "largest_arrays": largest,
    }


class _NvmlSampler:
    def __init__(self, physical_device_ids: tuple[int, ...]) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handles = {
            device_id: pynvml.nvmlDeviceGetHandleByIndex(device_id)
            for device_id in physical_device_ids
        }
        self._samples: dict[int, list[dict[str, int]]] = {
            device_id: [] for device_id in physical_device_ids
        }
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._sample,
            name="representax-nvml-sampler",
            daemon=True,
        )

    def _process_bytes(self, handle: Any) -> int:
        try:
            processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except self._pynvml.NVMLError:
            return 0
        return sum(
            int(process.usedGpuMemory)
            for process in processes
            if process.pid == os.getpid()
        )

    def _sample(self) -> None:
        while not self._stop.is_set():
            for device_id, handle in self._handles.items():
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._samples[device_id].append(
                    {
                        "process_bytes": self._process_bytes(handle),
                        "gpu_utilization_percent": int(utilization.gpu),
                        "memory_utilization_percent": int(utilization.memory),
                    }
                )
            self._stop.wait(0.02)

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join()
        self._pynvml.nvmlShutdown()
        return {
            str(device_id): {
                "sample_count": len(samples),
                "peak_process_bytes": max(
                    (sample["process_bytes"] for sample in samples),
                    default=0,
                ),
                "peak_gpu_utilization_percent": max(
                    (sample["gpu_utilization_percent"] for sample in samples),
                    default=0,
                ),
                "active_gpu_samples": sum(
                    sample["gpu_utilization_percent"] > 0 for sample in samples
                ),
            }
            for device_id, samples in self._samples.items()
        }


def _memory_analysis(compiled: Any) -> dict[str, int] | None:
    analysis = compiled.memory_analysis()
    if analysis is None:
        return None
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {
        name: int(getattr(analysis, name))
        for name in names
        if getattr(analysis, name, None) is not None
    }


def main() -> None:
    arguments = _arguments()
    if arguments.global_batch_size % arguments.world_size:
        raise ValueError("global batch size must be divisible by world size")
    if arguments.warmup_steps < 0 or arguments.measured_steps <= 0:
        raise ValueError("warmup must be non-negative and measured steps positive")
    physical_device_ids = tuple(
        int(value) for value in arguments.physical_device_ids.split(",")
    )
    if len(physical_device_ids) != arguments.world_size:
        raise ValueError("physical device count must equal world size")

    import jax
    import jax.numpy as jnp
    import optax

    from representax.models.modernvbert import (
        ModernVBERTTextBatch,
        ModernVBERTTextCheckpointAdapter,
    )
    from representax.tasks.retrieval import MNRTask, retrieval_batch
    from representax.train import (
        GradCache,
        ShardingPlan,
        build_sharded_train_step,
        init_train_state,
    )

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices("gpu")
    if len(devices) != arguments.world_size:
        raise RuntimeError(
            f"expected {arguments.world_size} visible GPUs, received {devices}"
        )
    config = json.loads((arguments.checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    generator = np.random.default_rng(arguments.seed)
    shape = (arguments.global_batch_size, arguments.sequence_length)
    cpu = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu):
        model = ModernVBERTTextCheckpointAdapter().load(
            arguments.checkpoint,
            parameter_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            rematerialization="full",
        )
        optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.0)
        state = init_train_state(model, optimizer)
        input_ids = generator.integers(
            1,
            int(text_config["vocab_size"]),
            size=shape,
            dtype=np.int32,
        )
        document_ids = generator.integers(
            1,
            int(text_config["vocab_size"]),
            size=shape,
            dtype=np.int32,
        )
        attention_mask = np.ones(shape, dtype=np.int32)
        batch = retrieval_batch(
            query=ModernVBERTTextBatch(
                input_ids=jnp.asarray(input_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            document=ModernVBERTTextBatch(
                input_ids=jnp.asarray(document_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            positive_mask=jnp.eye(arguments.global_batch_size, dtype=jnp.bool_),
        )
    mesh = jax.make_mesh(
        (arguments.world_size,),
        ("data",),
        devices=devices,
    )
    if arguments.strategy == "ddp":
        plan = ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
    else:
        plan = ShardingPlan.fsdp(
            state,
            optimizer,
            mesh,
            parameter_axis_name="data",
            data_axis_name="data",
            materialization_boundary=arguments.materialization_boundary,
        )
    step = build_sharded_train_step(
        MNRTask(scale=20.0),
        optimizer,
        plan,
        max_grad_norm=None,
        execution=GradCache(
            query_chunk_size=arguments.chunk_size,
            document_chunk_size=arguments.chunk_size,
            loss_row_chunk_size=arguments.chunk_size,
        ),
        donate_state=False,
    )
    sampler = _NvmlSampler(physical_device_ids)
    sampler.start()
    placed_state = plan.place_state(state)
    placed_batch = plan.place_batch(batch)
    placed_key = jax.device_put(
        jax.random.key(arguments.seed), plan.replicated_sharding
    )
    lowering_started = time.perf_counter()
    lowered = step.lower(placed_state, placed_batch, placed_key)
    stablehlo = lowered.as_text()
    if arguments.stablehlo_output is not None:
        arguments.stablehlo_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.stablehlo_output.write_text(stablehlo)
    lowering_seconds = time.perf_counter() - lowering_started
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started

    current_state = placed_state
    for index in range(arguments.warmup_steps):
        result = compiled(
            current_state,
            placed_batch,
            jax.device_put(
                jax.random.fold_in(jax.random.key(arguments.seed), index),
                plan.replicated_sharding,
            ),
        )
        jax.block_until_ready(result)
        current_state = result.state
    step_seconds = []
    for index in range(arguments.measured_steps):
        key = jax.device_put(
            jax.random.fold_in(
                jax.random.key(arguments.seed),
                arguments.warmup_steps + index,
            ),
            plan.replicated_sharding,
        )
        started = time.perf_counter()
        with jax.profiler.TraceAnnotation(
            "representax_measured_train_step",
            step=index,
            strategy=arguments.strategy,
        ):
            result = compiled(current_state, placed_batch, key)
            jax.block_until_ready(result)
        step_seconds.append(time.perf_counter() - started)
        current_state = result.state
    nvml = sampler.finish()

    artifact = {
        "schema_version": "representax-sharding-profile-v1",
        "strategy": arguments.strategy,
        "materialization_boundary": (
            arguments.materialization_boundary if arguments.strategy == "fsdp" else None
        ),
        "checkpoint": str(arguments.checkpoint),
        "world_size": arguments.world_size,
        "logical_devices": [str(device) for device in devices],
        "physical_device_ids": list(physical_device_ids),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_python_client_allocator": os.environ.get(
            "XLA_PYTHON_CLIENT_ALLOCATOR",
            "default",
        ),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "default",
        ),
        "stablehlo_output": (
            None
            if arguments.stablehlo_output is None
            else str(arguments.stablehlo_output)
        ),
        "global_batch_size": arguments.global_batch_size,
        "local_batch_size": arguments.global_batch_size // arguments.world_size,
        "sequence_length": arguments.sequence_length,
        "chunk_size": arguments.chunk_size,
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "step_seconds": step_seconds,
        "median_step_seconds": statistics.median(step_seconds),
        "examples_per_second": (
            arguments.global_batch_size / statistics.median(step_seconds)
        ),
        "stablehlo_collectives": {
            "all_gather": stablehlo.count('"stablehlo.all_gather"'),
            "reduce_scatter": stablehlo.count('"stablehlo.reduce_scatter"'),
            "all_reduce": stablehlo.count('"stablehlo.all_reduce"'),
        },
        "compiled_memory_analysis": _memory_analysis(compiled),
        "model_layout": _layout_summary(current_state.model),
        "optimizer_layout": _layout_summary(current_state.optimizer_state),
        "batch_layout": _layout_summary(placed_batch),
        "jax_device_memory": {
            str(device): {
                str(name): int(value)
                for name, value in (device.memory_stats() or {}).items()
                if isinstance(value, int)
            }
            for device in devices
        },
        "nvml": nvml,
        "final_loss": float(result.metrics.loss),
        "numeric_finite": bool(result.metrics.numeric_finite),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
