"""Profile isolated DDP or FSDP ModernVBERT training on one GPU host."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint", type=Path)
    model_source.add_argument("--model-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stablehlo-output", type=Path)
    parser.add_argument("--compiled-hlo-output", type=Path)
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), required=True)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--axis-type", choices=("auto", "explicit"), default="auto")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--physical-device-ids", default="4,5")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--donate-state", action="store_true")
    parser.add_argument(
        "--skip-signatures",
        action="store_true",
        help="Skip full-state numerical signatures for capacity-only probes.",
    )
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


def _metric_snapshot(metrics: Any) -> dict[str, Any]:
    import jax

    task = jax.device_get(metrics.task)
    return {
        "loss": float(metrics.loss),
        "task": {str(name): float(value) for name, value in task.items()},
        "gradient_global_norm": float(metrics.gradient_global_norm),
        "clipped_gradient_global_norm": float(metrics.clipped_gradient_global_norm),
        "update_global_norm": float(metrics.update_global_norm),
        "numeric_finite": bool(metrics.numeric_finite),
        "skipped_update": bool(metrics.skipped_update),
    }


def _tree_signature(tree: Any) -> list[dict[str, Any]]:
    """Record compact per-leaf numerical evidence without exporting full state."""

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    signature = []
    for path, value in jax.tree.flatten_with_path(tree)[0]:
        if not eqx.is_array(value):
            continue
        numeric = value.astype(jnp.float32)
        total, squared_total, maximum = jax.device_get(
            (
                jnp.sum(numeric),
                jnp.sum(jnp.square(numeric)),
                jnp.max(jnp.abs(numeric)),
            )
        )
        signature.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sum": float(total),
                "squared_sum": float(squared_total),
                "absolute_maximum": float(maximum),
            }
        )
    return signature


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


def _compiled_collective_count(hlo: str, name: str) -> int:
    """Count synchronous collectives or asynchronous starts, but not completions."""

    return len(re.findall(rf"\b{re.escape(name)}(?:-start)?\(", hlo))


def main(arguments: argparse.Namespace, progress: dict[str, Any]) -> None:
    progress["phase"] = "validation"
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
    from jax.sharding import AxisType

    from representax.config import PrecisionConfig
    from representax.models.modernvbert import (
        ModernVBERTTextBatch,
        ModernVBERTTextCheckpointAdapter,
        ModernVBERTTextConfig,
        ModernVBERTTextEncoder,
    )
    from representax.precision import resolve_precision_policy
    from representax.tasks.retrieval import MNRTask, retrieval_batch
    from representax.train import (
        GradCache,
        ShardingPlan,
        build_train_step,
        init_train_state,
    )

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices("gpu")
    if len(devices) != arguments.world_size:
        raise RuntimeError(
            f"expected {arguments.world_size} visible GPUs, received {devices}"
        )
    generator = np.random.default_rng(arguments.seed)
    shape = (arguments.global_batch_size, arguments.sequence_length)
    cpu = jax.local_devices(backend="cpu")[0]
    progress["phase"] = "initialization"
    precision_config = (
        PrecisionConfig()
        if arguments.precision == "float32"
        else PrecisionConfig.bfloat16_mixed()
    )
    precision = resolve_precision_policy(precision_config)
    model_init_started = time.perf_counter()
    with jax.default_device(cpu):
        if arguments.checkpoint is not None:
            raw_config = json.loads((arguments.checkpoint / "config.json").read_text())
            text_config = ModernVBERTTextConfig.from_hf_config(raw_config)
            model = ModernVBERTTextCheckpointAdapter().load(
                arguments.checkpoint,
                parameter_dtype=precision.parameter_dtype,
                compute_dtype=precision.compute_dtype,
                rematerialization="full",
            )
        else:
            if arguments.model_config is None:  # pragma: no cover - argparse invariant
                raise AssertionError("a model source is required")
            text_config = ModernVBERTTextConfig.model_validate_json(
                arguments.model_config.read_text()
            )
            model = ModernVBERTTextEncoder.init(
                text_config,
                key=jax.random.key(arguments.seed),
                parameter_dtype=precision.parameter_dtype,
                compute_dtype=precision.compute_dtype,
                rematerialization="full",
                model_id=f"representax/{arguments.model_config.stem}",
            )
        optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.0)
        state = init_train_state(model, optimizer, precision=precision)
        input_ids = generator.integers(
            1,
            text_config.vocab_size,
            size=shape,
            dtype=np.int32,
        )
        document_ids = generator.integers(
            1,
            text_config.vocab_size,
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
    model_init_seconds = time.perf_counter() - model_init_started
    parameter_count = sum(
        int(value.size)
        for value in jax.tree.leaves(model)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)
    )
    progress.update(
        phase="planning",
        parameter_count=parameter_count,
        model_init_seconds=model_init_seconds,
    )
    mesh = jax.make_mesh(
        (arguments.world_size,),
        ("data",),
        axis_types=(
            AxisType.Auto if arguments.axis_type == "auto" else AxisType.Explicit,
        ),
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
        )
    step = build_train_step(
        MNRTask(scale=20.0),
        optimizer,
        plan=plan,
        max_grad_norm=None,
        execution=GradCache(
            query_chunk_size=arguments.chunk_size,
            document_chunk_size=arguments.chunk_size,
            loss_row_chunk_size=arguments.chunk_size,
        ),
        donate_state=arguments.donate_state,
        precision=precision,
    )
    sampler = _NvmlSampler(physical_device_ids)
    sampler.start()
    progress["phase"] = "placement"
    placement_started = time.perf_counter()
    placed_state = plan.place_state(state)
    placed_batch = plan.place_batch(batch)
    placed_key = jax.device_put(
        jax.random.key(arguments.seed), plan.replicated_sharding
    )
    jax.block_until_ready((placed_state, placed_batch, placed_key))
    placement_seconds = time.perf_counter() - placement_started
    progress.update(phase="lowering", placement_seconds=placement_seconds)
    lowering_started = time.perf_counter()
    lowered = step.lower(placed_state, placed_batch, placed_key)
    stablehlo = lowered.as_text()
    if arguments.stablehlo_output is not None:
        arguments.stablehlo_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.stablehlo_output.write_text(stablehlo)
    lowering_seconds = time.perf_counter() - lowering_started
    progress.update(phase="compilation", lowering_seconds=lowering_seconds)
    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started
    compiled_hlo = "\n".join(
        module.to_string() for module in compiled.runtime_executable().hlo_modules()
    )
    if arguments.compiled_hlo_output is not None:
        arguments.compiled_hlo_output.parent.mkdir(parents=True, exist_ok=True)
        arguments.compiled_hlo_output.write_text(compiled_hlo)

    current_state = placed_state
    progress.update(phase="warmup", compile_seconds=compile_seconds)
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
        if arguments.donate_state:
            current_state = result.state
    step_seconds = []
    metric_trajectory = []
    progress["phase"] = "measurement"
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
        metric_trajectory.append(_metric_snapshot(result.metrics))
        current_state = result.state
    nvml = sampler.finish()
    median_step_seconds = statistics.median(step_seconds)

    artifact = {
        "schema_version": "representax-sharding-profile-v4",
        "status": "completed",
        "strategy": arguments.strategy,
        "precision": precision_config.model_dump(mode="json"),
        "axis_type": arguments.axis_type,
        "checkpoint": (
            None if arguments.checkpoint is None else str(arguments.checkpoint)
        ),
        "model_config": (
            None if arguments.model_config is None else str(arguments.model_config)
        ),
        "parameter_count": parameter_count,
        "model_init_seconds": model_init_seconds,
        "world_size": arguments.world_size,
        "seed": arguments.seed,
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
        "tf_gpu_allocator": os.environ.get("TF_GPU_ALLOCATOR", "default"),
        "stablehlo_output": (
            None
            if arguments.stablehlo_output is None
            else str(arguments.stablehlo_output)
        ),
        "compiled_hlo_output": (
            None
            if arguments.compiled_hlo_output is None
            else str(arguments.compiled_hlo_output)
        ),
        "global_batch_size": arguments.global_batch_size,
        "local_batch_size": arguments.global_batch_size // arguments.world_size,
        "sequence_length": arguments.sequence_length,
        "chunk_size": arguments.chunk_size,
        "warmup_steps": arguments.warmup_steps,
        "measured_steps": arguments.measured_steps,
        "donate_state": arguments.donate_state,
        "placement_seconds": placement_seconds,
        "lowering_seconds": lowering_seconds,
        "compile_seconds": compile_seconds,
        "step_seconds": step_seconds,
        "metric_trajectory": metric_trajectory,
        "median_step_seconds": median_step_seconds,
        "examples_per_second": arguments.global_batch_size / median_step_seconds,
        "training_tokens_per_second": (
            2
            * arguments.global_batch_size
            * arguments.sequence_length
            / median_step_seconds
        ),
        "stablehlo_collectives": {
            "all_gather": stablehlo.count('"stablehlo.all_gather"'),
            "reduce_scatter": stablehlo.count('"stablehlo.reduce_scatter"'),
            "all_reduce": stablehlo.count('"stablehlo.all_reduce"'),
        },
        "compiled_hlo_collectives": {
            "all_gather": _compiled_collective_count(compiled_hlo, "all-gather"),
            "all_reduce": _compiled_collective_count(compiled_hlo, "all-reduce"),
            "reduce_scatter": _compiled_collective_count(
                compiled_hlo,
                "reduce-scatter",
            ),
            "collective_permute": _compiled_collective_count(
                compiled_hlo,
                "collective-permute",
            ),
            "all_to_all": _compiled_collective_count(compiled_hlo, "all-to-all"),
        },
        "compiled_memory_analysis": _memory_analysis(compiled),
        "model_layout": _layout_summary(current_state.model),
        "optimizer_layout": _layout_summary(current_state.optimizer_state),
        "model_signature": (
            None if arguments.skip_signatures else _tree_signature(current_state.model)
        ),
        "optimizer_signature": (
            None
            if arguments.skip_signatures
            else _tree_signature(current_state.optimizer_state)
        ),
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
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "strategy",
                    "precision",
                    "world_size",
                    "parameter_count",
                    "global_batch_size",
                    "sequence_length",
                    "model_init_seconds",
                    "placement_seconds",
                    "lowering_seconds",
                    "compile_seconds",
                    "median_step_seconds",
                    "examples_per_second",
                    "training_tokens_per_second",
                    "final_loss",
                    "numeric_finite",
                    "compiled_hlo_collectives",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parsed_arguments = _arguments()
    run_progress: dict[str, Any] = {}
    try:
        main(parsed_arguments, run_progress)
    except Exception as error:
        message = str(error)
        category = (
            "oom"
            if any(
                marker in message.lower()
                for marker in ("out of memory", "resource_exhausted", "oom")
            )
            else "error"
        )
        failure = {
            "schema_version": "representax-sharding-profile-v4",
            "status": category,
            "strategy": parsed_arguments.strategy,
            "precision": parsed_arguments.precision,
            "world_size": parsed_arguments.world_size,
            "seed": parsed_arguments.seed,
            "physical_device_ids": parsed_arguments.physical_device_ids,
            "checkpoint": (
                None
                if parsed_arguments.checkpoint is None
                else str(parsed_arguments.checkpoint)
            ),
            "model_config": (
                None
                if parsed_arguments.model_config is None
                else str(parsed_arguments.model_config)
            ),
            "global_batch_size": parsed_arguments.global_batch_size,
            "sequence_length": parsed_arguments.sequence_length,
            "failed_phase": run_progress.get("phase", "argument_parsing"),
            "parameter_count": run_progress.get("parameter_count"),
            "model_init_seconds": run_progress.get("model_init_seconds"),
            "placement_seconds": run_progress.get("placement_seconds"),
            "lowering_seconds": run_progress.get("lowering_seconds"),
            "compile_seconds": run_progress.get("compile_seconds"),
            "xla_python_client_allocator": os.environ.get(
                "XLA_PYTHON_CLIENT_ALLOCATOR",
                "default",
            ),
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "default",
            ),
            "tf_gpu_allocator": os.environ.get("TF_GPU_ALLOCATOR", "default"),
            "error_type": type(error).__name__,
            "error": message[-8_000:],
        }
        parsed_arguments.output.parent.mkdir(parents=True, exist_ok=True)
        parsed_arguments.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise
