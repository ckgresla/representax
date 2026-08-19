"""ModernVBERT one-device versus replicated-data-parallel GradCache probe."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-id",
        default="ModernVBERT/modernvbert-embed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=("ddp", "fsdp"), default="ddp")
    parser.add_argument("--world-size", type=int, choices=(2, 4), required=True)
    parser.add_argument("--process-count", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    parser.add_argument("--coordinator-address")
    parser.add_argument("--local-device-ids")
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _tree_difference(actual: Any, expected: Any) -> dict[str, float]:
    import equinox as eqx
    import jax

    actual_leaves = [leaf for leaf in jax.tree.leaves(actual) if eqx.is_array(leaf)]
    expected_leaves = [leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)]
    if len(actual_leaves) != len(expected_leaves):
        raise ValueError("compared trees contain different array structures")
    maximum = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    squared_actual = 0.0
    dot_product = 0.0
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        actual_value = _host_value(actual_leaf)
        expected_value = _host_value(expected_leaf)
        if not np.issubdtype(actual_value.dtype, np.inexact):
            if not np.array_equal(actual_value, expected_value):
                maximum = float("inf")
            continue
        difference = actual_value.astype(np.float32) - expected_value.astype(np.float32)
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared_error += float(np.sum(np.square(difference), dtype=np.float64))
        squared_actual += float(
            np.sum(np.square(actual_value.astype(np.float32)), dtype=np.float64)
        )
        squared_reference += float(
            np.sum(
                np.square(expected_value.astype(np.float32)),
                dtype=np.float64,
            )
        )
        dot_product += float(
            np.sum(
                actual_value.astype(np.float32) * expected_value.astype(np.float32),
                dtype=np.float64,
            )
        )
    denominator = max((squared_actual * squared_reference) ** 0.5, 1e-30)
    return {
        "maximum_absolute": maximum,
        "relative_l2": (squared_error / max(squared_reference, 1e-30)) ** 0.5,
        "cosine_similarity": dot_product / denominator,
        "error_l2": squared_error**0.5,
        "reference_l2": squared_reference**0.5,
    }


def _host_value(value: Any) -> np.ndarray:
    """Copy an addressable replica, including a cross-process global array."""

    if hasattr(value, "is_fully_addressable") and not value.is_fully_addressable:
        if not value.sharding.is_fully_replicated:
            raise ValueError("only replicated global arrays can be copied locally")
        value = value.addressable_data(0)
    return np.asarray(value)


def _scalar(value: Any) -> float:
    return float(_host_value(value))


def _progress(process_id: int, phase: str) -> None:
    print(f"process {process_id}: {phase}", flush=True)


def main() -> None:
    arguments = _arguments()
    if arguments.global_batch_size % arguments.world_size:
        raise ValueError("global batch size must be divisible by world size")
    if arguments.process_count < 1:
        raise ValueError("process count must be positive")
    if not 0 <= arguments.process_id < arguments.process_count:
        raise ValueError("process id must be in [0, process_count)")
    if arguments.global_batch_size % arguments.process_count:
        raise ValueError("global batch size must be divisible by process count")
    if arguments.strategy == "fsdp" and arguments.process_count != 1:
        raise ValueError("the current FSDP acceptance probe is single-host only")

    import jax

    if arguments.process_count > 1:
        if arguments.coordinator_address is None:
            raise ValueError("multi-process execution requires a coordinator address")
        if arguments.local_device_ids is None:
            raise ValueError("multi-process execution requires local device ids")
        local_device_ids = tuple(
            int(device_id) for device_id in arguments.local_device_ids.split(",")
        )
        jax.distributed.initialize(
            coordinator_address=arguments.coordinator_address,
            num_processes=arguments.process_count,
            process_id=arguments.process_id,
            local_device_ids=local_device_ids,
        )
    _progress(arguments.process_id, "distributed runtime initialized")

    import equinox as eqx
    import jax.numpy as jnp
    import optax

    from representax.models.modernvbert import (
        ModernVBERTTextBatch,
        ModernVBERTTextCheckpointAdapter,
    )
    from representax.tasks.retrieval import (
        MNRTask,
        process_local_retrieval_batch,
        retrieval_batch,
    )
    from representax.train import (
        DataParallel,
        GradCache,
        ShardingPlan,
        build_data_parallel_train_step,
        build_sharded_train_step,
        build_train_step,
        make_train_state,
    )

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices("gpu")
    local_devices = jax.local_devices(backend="gpu")
    if len(devices) != arguments.world_size:
        raise RuntimeError(
            f"expected exactly {arguments.world_size} visible GPUs, got {devices}"
        )
    expected_local_device_count = arguments.world_size // arguments.process_count
    if len(local_devices) != expected_local_device_count:
        raise RuntimeError(
            f"expected {expected_local_device_count} local GPUs, got {local_devices}"
        )
    config = json.loads((arguments.checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    generator = np.random.default_rng(arguments.seed)
    shape = (arguments.global_batch_size, arguments.sequence_length)
    query_ids = generator.integers(
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

    cpu = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu):
        model = ModernVBERTTextCheckpointAdapter().load(
            arguments.checkpoint,
            parameter_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            rematerialization="full",
        )
        optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.0)
        host_state = make_train_state(model, optimizer)
        positive_mask = jnp.eye(arguments.global_batch_size, dtype=jnp.bool_)
        positive_mask = positive_mask.at[0, 0].set(False)
        positive_mask = positive_mask.at[
            0,
            arguments.global_batch_size // 2,
        ].set(True)
        host_batch = retrieval_batch(
            query=ModernVBERTTextBatch(
                input_ids=jnp.asarray(query_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            document=ModernVBERTTextBatch(
                input_ids=jnp.asarray(document_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            positive_mask=positive_mask,
        )
    _progress(arguments.process_id, "checkpoint and host batch loaded")

    execution = GradCache(
        query_chunk_size=arguments.chunk_size,
        document_chunk_size=arguments.chunk_size,
        loss_row_chunk_size=arguments.chunk_size,
    )
    task = MNRTask(scale=20.0)
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        execution=execution,
        donate_state=False,
    )
    reference_state = jax.device_put(host_state, local_devices[0])
    reference_batch = jax.device_put(host_batch, local_devices[0])
    if arguments.process_count == 1:
        mesh = jax.make_mesh(
            (arguments.world_size,),
            ("data",),
            devices=devices,
        )
        if arguments.strategy == "ddp":
            sharding_plan = ShardingPlan.ddp(
                host_state,
                optimizer,
                mesh,
                axis_name="data",
            )
        else:
            sharding_plan = ShardingPlan.fsdp(
                host_state,
                optimizer,
                mesh,
                parameter_axis_name="data",
                data_axis_name="data",
            )
        distributed_step = build_sharded_train_step(
            task,
            optimizer,
            sharding_plan,
            max_grad_norm=None,
            execution=execution,
            donate_state=False,
        )
        distributed_state = sharding_plan.place_state(host_state)
        distributed_batch = sharding_plan.place_batch(host_batch)

        def place_key(value: Any) -> Any:
            return jax.device_put(value, sharding_plan.replicated_sharding)

    else:
        data_parallel = DataParallel.from_devices(devices)
        distributed_step = build_data_parallel_train_step(
            task,
            optimizer,
            data_parallel,
            max_grad_norm=None,
            execution=execution,
            donate_state=False,
        )
        distributed_state = data_parallel.place_replicated(host_state)
        process_batch_size = arguments.global_batch_size // arguments.process_count
        start = arguments.process_id * process_batch_size
        stop = start + process_batch_size

        def local_rows(tree: Any) -> Any:
            return jax.tree.map(
                lambda value: value[start:stop] if eqx.is_array(value) else value,
                tree,
                is_leaf=lambda value: value is None,
            )

        local_batch = process_local_retrieval_batch(
            query=local_rows(host_batch.query),
            document=local_rows(host_batch.document),
            positive_mask=host_batch.positive_mask[start:stop],
            positive_weights=(
                None
                if host_batch.positive_weights is None
                else host_batch.positive_weights[start:stop]
            ),
            query_valid=host_batch.query_valid[start:stop],
            document_valid=host_batch.document_valid[start:stop],
        )
        distributed_batch = data_parallel.place_process_local_batch(local_batch)

        def place_key(value: Any) -> Any:
            return data_parallel.place_replicated(value)

    _progress(arguments.process_id, "reference and distributed inputs placed")
    key = jax.random.key(arguments.seed)

    started = time.perf_counter()
    reference = reference_step(reference_state, reference_batch, key)
    jax.block_until_ready(reference)
    reference_compile_plus_first = time.perf_counter() - started
    _progress(arguments.process_id, "one-device oracle complete")

    started = time.perf_counter()
    distributed = distributed_step(
        distributed_state,
        distributed_batch,
        place_key(key),
    )
    jax.block_until_ready(distributed)
    distributed_compile_plus_first = time.perf_counter() - started
    _progress(arguments.process_id, "distributed first update complete")

    model_difference = _tree_difference(
        distributed.state.model,
        reference.state.model,
    )
    optimizer_difference = _tree_difference(
        distributed.state.optimizer_state,
        reference.state.optimizer_state,
    )
    loss_absolute_difference = abs(
        _scalar(distributed.metrics.loss) - _scalar(reference.metrics.loss)
    )
    gradient_norm_relative_difference = abs(
        _scalar(distributed.metrics.gradient_global_norm)
        - _scalar(reference.metrics.gradient_global_norm)
    ) / max(abs(_scalar(reference.metrics.gradient_global_norm)), 1e-30)
    accepted = (
        loss_absolute_difference <= 5e-5
        and gradient_norm_relative_difference <= 5e-4
        and model_difference["maximum_absolute"] <= 2.5e-5
        and model_difference["relative_l2"] <= 3e-6
        and optimizer_difference["relative_l2"] <= 2e-3
        and optimizer_difference["cosine_similarity"] >= 0.999_999
    )

    state = distributed.state
    for index in range(arguments.warmup_steps):
        result = distributed_step(
            state,
            distributed_batch,
            place_key(jax.random.fold_in(key, index + 1)),
        )
        jax.block_until_ready(result)
        state = result.state
    samples = []
    for index in range(arguments.measured_steps):
        started = time.perf_counter()
        result = distributed_step(
            state,
            distributed_batch,
            place_key(jax.random.fold_in(key, arguments.warmup_steps + index + 1)),
        )
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - started)
        state = result.state

    artifact = {
        "schema_version": "representax-distributed-grad-cache-modernvbert-v1",
        "status": "accepted" if accepted else "rejected",
        "framework_version": _version("representax"),
        "jax_version": jax.__version__,
        "python": platform.python_version(),
        "compilation_cache": {
            "policy": "ambient",
            "cold_start_claimed": False,
        },
        "checkpoint": arguments.checkpoint_id,
        "checkpoint_revision": arguments.checkpoint.name,
        "strategy": arguments.strategy,
        "devices": [str(device) for device in devices],
        "local_devices": [str(device) for device in local_devices],
        "world_size": arguments.world_size,
        "process_count": arguments.process_count,
        "process_id": arguments.process_id,
        "global_batch_size": arguments.global_batch_size,
        "local_batch_size": arguments.global_batch_size // arguments.world_size,
        "process_local_batch_size": (
            arguments.global_batch_size // arguments.process_count
        ),
        "sequence_length": arguments.sequence_length,
        "chunk_size": arguments.chunk_size,
        "cross_process_positive": {
            "query_index": 0,
            "document_index": arguments.global_batch_size // 2,
        },
        "reference": {
            "loss": _scalar(reference.metrics.loss),
            "gradient_global_norm": _scalar(reference.metrics.gradient_global_norm),
            "compile_plus_first_seconds": reference_compile_plus_first,
        },
        "distributed": {
            "loss": _scalar(distributed.metrics.loss),
            "gradient_global_norm": _scalar(distributed.metrics.gradient_global_norm),
            "compile_plus_first_seconds": distributed_compile_plus_first,
            "steady_state_seconds": samples,
            "steady_state_median_seconds": statistics.median(samples),
            "examples_per_second": (
                arguments.global_batch_size / statistics.median(samples)
            ),
            "allocator_peak_bytes_by_device": {
                str(device): int(
                    (device.memory_stats() or {}).get(
                        "peak_bytes_in_use",
                        (device.memory_stats() or {}).get("bytes_in_use", 0),
                    )
                )
                for device in local_devices
            },
        },
        "differences": {
            "loss_absolute": loss_absolute_difference,
            "gradient_norm_relative": gradient_norm_relative_difference,
            "model": model_difference,
            "optimizer_state": optimizer_difference,
        },
        "precision": "float32-highest-no-tf32",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    _progress(arguments.process_id, "artifact published")
    if arguments.process_count > 1:
        from jax.experimental import multihost_utils

        multihost_utils.sync_global_devices("representax-acceptance-complete")
        jax.distributed.shutdown()
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
