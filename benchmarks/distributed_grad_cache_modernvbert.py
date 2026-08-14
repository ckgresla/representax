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
    parser.add_argument("--world-size", type=int, choices=(2, 4), required=True)
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
    expected_leaves = [
        leaf for leaf in jax.tree.leaves(expected) if eqx.is_array(leaf)
    ]
    if len(actual_leaves) != len(expected_leaves):
        raise ValueError("compared trees contain different array structures")
    maximum = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        actual_value = np.asarray(actual_leaf)
        expected_value = np.asarray(expected_leaf)
        if not np.issubdtype(actual_value.dtype, np.inexact):
            if not np.array_equal(actual_value, expected_value):
                maximum = float("inf")
            continue
        difference = actual_value.astype(np.float32) - expected_value.astype(
            np.float32
        )
        maximum = max(maximum, float(np.max(np.abs(difference))))
        squared_error += float(np.sum(np.square(difference), dtype=np.float64))
        squared_reference += float(
            np.sum(
                np.square(expected_value.astype(np.float32)),
                dtype=np.float64,
            )
        )
    return {
        "maximum_absolute": maximum,
        "relative_l2": (squared_error / max(squared_reference, 1e-30)) ** 0.5,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.global_batch_size % arguments.world_size:
        raise ValueError("global batch size must be divisible by world size")

    import jax
    import jax.numpy as jnp
    import optax

    from representax.models.modernvbert import (
        ModernVBERTTextBatch,
        ModernVBERTTextCheckpointAdapter,
    )
    from representax.tasks.retrieval import MNRTask, retrieval_batch
    from representax.train import (
        DataParallel,
        GradCache,
        build_data_parallel_train_step,
        build_train_step,
        make_train_state,
    )

    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices("gpu")
    if len(devices) != arguments.world_size:
        raise RuntimeError(
            f"expected exactly {arguments.world_size} visible GPUs, got {devices}"
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

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = ModernVBERTTextCheckpointAdapter().load(
            arguments.checkpoint,
            parameter_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            rematerialization="full",
        )
        optimizer = optax.adamw(learning_rate=1e-5, weight_decay=0.0)
        host_state = make_train_state(model, optimizer)
        host_batch = retrieval_batch(
            query=ModernVBERTTextBatch(
                input_ids=jnp.asarray(query_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            document=ModernVBERTTextBatch(
                input_ids=jnp.asarray(document_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            positive_mask=jnp.eye(arguments.global_batch_size, dtype=jnp.bool_),
        )

    execution = GradCache(
        query_chunk_size=arguments.chunk_size,
        document_chunk_size=arguments.chunk_size,
        representation_chunk_size=arguments.chunk_size,
    )
    task = MNRTask(scale=20.0)
    reference_step = build_train_step(
        task,
        optimizer,
        max_grad_norm=None,
        execution=execution,
        donate_state=False,
    )
    plan = DataParallel.from_devices(devices)
    distributed_step = build_data_parallel_train_step(
        task,
        optimizer,
        plan,
        max_grad_norm=None,
        execution=execution,
        donate_state=False,
    )
    reference_state = jax.device_put(host_state, devices[0])
    reference_batch = jax.device_put(host_batch, devices[0])
    distributed_state = plan.place_replicated(host_state)
    distributed_batch = plan.place_batch(host_batch)
    key = jax.random.key(arguments.seed)

    started = time.perf_counter()
    reference = reference_step(reference_state, reference_batch, key)
    jax.block_until_ready(reference)
    reference_compile_plus_first = time.perf_counter() - started

    started = time.perf_counter()
    distributed = distributed_step(
        distributed_state,
        distributed_batch,
        plan.place_replicated(key),
    )
    jax.block_until_ready(distributed)
    distributed_compile_plus_first = time.perf_counter() - started

    model_difference = _tree_difference(
        distributed.state.model,
        reference.state.model,
    )
    optimizer_difference = _tree_difference(
        distributed.state.optimizer_state,
        reference.state.optimizer_state,
    )
    loss_absolute_difference = abs(
        float(distributed.metrics.loss) - float(reference.metrics.loss)
    )
    gradient_norm_relative_difference = abs(
        float(distributed.metrics.gradient_global_norm)
        - float(reference.metrics.gradient_global_norm)
    ) / max(abs(float(reference.metrics.gradient_global_norm)), 1e-30)
    accepted = (
        loss_absolute_difference <= 5e-5
        and gradient_norm_relative_difference <= 5e-4
        and model_difference["maximum_absolute"] <= 2e-5
        and optimizer_difference["relative_l2"] <= 5e-4
    )

    state = distributed.state
    for index in range(arguments.warmup_steps):
        result = distributed_step(
            state,
            distributed_batch,
            plan.place_replicated(jax.random.fold_in(key, index + 1)),
        )
        jax.block_until_ready(result)
        state = result.state
    samples = []
    for index in range(arguments.measured_steps):
        started = time.perf_counter()
        result = distributed_step(
            state,
            distributed_batch,
            plan.place_replicated(
                jax.random.fold_in(key, arguments.warmup_steps + index + 1)
            ),
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
        "checkpoint": arguments.checkpoint_id,
        "checkpoint_revision": arguments.checkpoint.name,
        "devices": [str(device) for device in devices],
        "world_size": arguments.world_size,
        "global_batch_size": arguments.global_batch_size,
        "local_batch_size": arguments.global_batch_size // arguments.world_size,
        "sequence_length": arguments.sequence_length,
        "chunk_size": arguments.chunk_size,
        "reference": {
            "loss": float(reference.metrics.loss),
            "gradient_global_norm": float(reference.metrics.gradient_global_norm),
            "compile_plus_first_seconds": reference_compile_plus_first,
        },
        "distributed": {
            "loss": float(distributed.metrics.loss),
            "gradient_global_norm": float(distributed.metrics.gradient_global_norm),
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
                for device in devices
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
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
