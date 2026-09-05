"""Exercise one Cloud TPU slice with matched JAX and PyTorch/XLA programs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

INPUT_DIMENSION = 1024
OUTPUT_DIMENSION = 256
GLOBAL_BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
STEPS = 20


def _emit(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _safe_attribute(value: Any, name: str) -> Any:
    try:
        return getattr(value, name)
    except (AttributeError, RuntimeError):
        return None


def _rows(start: int, size: int) -> tuple[Any, Any, Any]:
    import numpy as np

    row = np.arange(start, start + size, dtype=np.float32)[:, None]
    axis = np.arange(INPUT_DIMENSION, dtype=np.float32)[None, :]
    left = np.sin(row * 0.013 + axis * 0.017, dtype=np.float32)
    right = np.cos(row * 0.019 + axis * 0.011, dtype=np.float32)
    target = (0.5 * np.sin(row[:, 0] * 0.007) + 0.25).astype(np.float32)
    return left, right, target


def _initial_parameters() -> tuple[Any, Any]:
    import numpy as np

    row = np.arange(INPUT_DIMENSION, dtype=np.float32)[:, None]
    column = np.arange(OUTPUT_DIMENSION, dtype=np.float32)[None, :]
    weight = np.sin(row * 0.003 + column * 0.005, dtype=np.float32)
    weight /= math.sqrt(INPUT_DIMENSION)
    return weight, np.zeros((OUTPUT_DIMENSION,), dtype=np.float32)


def _jax_initialize(distributed: bool) -> Any:
    import jax

    if distributed:
        jax.distributed.initialize()
    return jax


def jax_topology(
    *, distributed: bool, scope: str, payload_mib: int, iterations: int
) -> None:
    jax = _jax_initialize(distributed)
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()
    local_devices = jax.local_devices()
    topology = [
        {
            "id": device.id,
            "process_index": device.process_index,
            "coords": _safe_attribute(device, "coords"),
            "core_on_chip": _safe_attribute(device, "core_on_chip"),
            "device_kind": device.device_kind,
        }
        for device in devices
    ]
    if jax.process_index() == 0:
        _emit(
            "topology",
            distributed=distributed,
            process_count=jax.process_count(),
            global_device_count=len(devices),
            local_device_count=len(local_devices),
            devices=topology,
        )

    mesh_devices = devices if distributed else local_devices
    mesh = Mesh(np.asarray(mesh_devices), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    bytes_per_device = payload_mib * 1024 * 1024
    elements_per_device = bytes_per_device // np.dtype(np.float32).itemsize
    local = np.ones(
        (len(local_devices) * elements_per_device,),
        dtype=np.float32,
    )
    value = jax.make_array_from_process_local_data(
        sharding,
        local,
        global_shape=(len(mesh_devices) * elements_per_device,),
    )

    process_groups = [
        [
            index
            for index, device in enumerate(mesh_devices)
            if device.process_index == process_index
        ]
        for process_index in sorted({device.process_index for device in mesh_devices})
    ]
    scopes = ("local", "global") if scope == "both" else (scope,)
    def make_repeated_all_reduce(groups: list[list[int]] | None) -> Any:
        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P(),
            check_vma=False,
        )
        def repeated_all_reduce(shard: Any) -> Any:
            def reduce_once(_: int, current: Any) -> Any:
                return jax.lax.psum(
                    current, "data", axis_index_groups=groups
                )

            return jax.lax.fori_loop(0, iterations, reduce_once, shard)

        return jax.jit(
            repeated_all_reduce,
            in_shardings=sharding,
            out_shardings=replicated,
        )

    for measured_scope in scopes:
        groups = process_groups if measured_scope == "local" else None
        repeated_all_reduce = make_repeated_all_reduce(groups)
        repeated_all_reduce(value).block_until_ready()
        durations = []
        for _ in range(5):
            started = time.perf_counter()
            repeated_all_reduce(value).block_until_ready()
            durations.append(time.perf_counter() - started)
        if jax.process_index() == 0:
            median = statistics.median(durations) / iterations
            participants = (
                len(process_groups[0])
                if measured_scope == "local"
                else len(mesh_devices)
            )
            _emit(
                "all_reduce",
                distributed=distributed,
                scope=measured_scope,
                participants=participants,
                payload_bytes_per_device=bytes_per_device,
                median_seconds=median,
                measurements=5,
                collectives_per_measurement=iterations,
                payload_gigabytes_per_second_per_device=(
                    bytes_per_device / median / 1e9
                ),
            )


def jax_dense(*, steps: int, global_batch_size: int) -> None:
    jax = _jax_initialize(True)
    import jax.numpy as jnp
    import numpy as np
    import optax
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    if global_batch_size % jax.device_count():
        raise ValueError("global batch must be divisible by the global device count")
    local_size = global_batch_size * jax.local_device_count() // jax.device_count()
    start = jax.process_index() * local_size
    left, right, target = _rows(start, local_size)
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    batch_sharding = NamedSharding(mesh, P("data", None))
    target_sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())
    left_array = jax.make_array_from_process_local_data(
        batch_sharding,
        left,
        global_shape=(global_batch_size, INPUT_DIMENSION),
    )
    right_array = jax.make_array_from_process_local_data(
        batch_sharding,
        right,
        global_shape=(global_batch_size, INPUT_DIMENSION),
    )
    target_array = jax.make_array_from_process_local_data(
        target_sharding,
        target,
        global_shape=(global_batch_size,),
    )
    weight, bias = _initial_parameters()
    parameters = (
        jax.device_put(jnp.asarray(weight), replicated),
        jax.device_put(jnp.asarray(bias), replicated),
    )
    optimizer = optax.adamw(LEARNING_RATE, weight_decay=0.0)
    optimizer_state = jax.device_put(optimizer.init(parameters), replicated)

    def objective(
        model: Any,
        left_batch: Any,
        right_batch: Any,
        targets: Any,
    ) -> Any:
        matrix, offset = model
        left_embedding = left_batch @ matrix + offset
        right_embedding = right_batch @ matrix + offset
        left_embedding /= jnp.maximum(
            jnp.linalg.norm(left_embedding, axis=-1, keepdims=True), 1e-12
        )
        right_embedding /= jnp.maximum(
            jnp.linalg.norm(right_embedding, axis=-1, keepdims=True), 1e-12
        )
        prediction = jnp.sum(left_embedding * right_embedding, axis=-1)
        return jnp.mean(jnp.square(prediction - targets))

    @partial(
        jax.jit,
        in_shardings=(
            replicated,
            replicated,
            batch_sharding,
            batch_sharding,
            target_sharding,
        ),
    )
    def train_step(
        model: Any,
        state: Any,
        left_batch: Any,
        right_batch: Any,
        targets: Any,
    ) -> tuple[Any, Any, Any]:
        loss, gradients = jax.value_and_grad(objective)(
            model, left_batch, right_batch, targets
        )
        updates, state = optimizer.update(gradients, state, model)
        return optax.apply_updates(model, updates), state, loss

    durations = []
    losses = []
    compilation_seconds = None
    with jax.default_matmul_precision("highest"):
        for step in range(steps):
            started = time.perf_counter()
            parameters, optimizer_state, loss = train_step(
                parameters,
                optimizer_state,
                left_array,
                right_array,
                target_array,
            )
            loss = float(loss.block_until_ready())
            elapsed = time.perf_counter() - started
            losses.append(loss)
            if step == 0:
                compilation_seconds = elapsed
            else:
                durations.append(elapsed)
            if jax.process_index() == 0:
                _emit(
                    "training_step",
                    framework="jax",
                    step=step + 1,
                    loss=loss,
                    seconds=elapsed,
                )
    if jax.process_index() == 0:
        median = statistics.median(durations)
        _emit(
            "summary",
            framework="jax",
            process_count=jax.process_count(),
            device_count=jax.device_count(),
            steps=steps,
            global_batch_size=global_batch_size,
            first_loss=losses[0],
            final_loss=losses[-1],
            compilation_and_first_step_seconds=compilation_seconds,
            median_step_seconds=median,
            examples_per_second=global_batch_size / median,
        )


def representax_dense(
    *, output: Path, steps: int, global_batch_size: int
) -> None:
    jax = _jax_initialize(True)

    from experiments.preflights.tpu import (
        MAPPER,
        ToySource,
        Variant,
        _job,
        _record,
        identity,
    )
    from representax.config import ExportConfig
    from representax.train import run_job

    if global_batch_size % jax.device_count():
        raise ValueError("global batch must be divisible by the global device count")
    variant = Variant(
        "retrieval-grad-cache-ddp",
        "retrieval",
        grad_cache="rematerialized",
        sharding="ddp",
    )
    job = _job(
        variant,
        device_count=jax.device_count(),
        steps=steps,
        global_batch_size=global_batch_size,
    ).model_copy(
        update={
            "checkpointing": None,
            "evaluation": None,
            "export": ExportConfig(enabled=False),
        }
    )
    run_directory = output.expanduser().resolve() / f"process-{jax.process_index()}"
    records = ToySource(
        tuple(_record(index) for index in range(global_batch_size * steps))
    )

    def resolve_records(_: Any) -> ToySource:
        return records

    result = run_job(
        job,
        run_directory,
        resolvers={"memory": resolve_records},
        mappers={MAPPER: identity},
    )
    if jax.process_index() != 0:
        return
    rows = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    training = [row for row in rows if row["event"] == "training_step"]
    warm_rates = [
        float(row["metrics"]["perf/examples_per_second"])
        for row in training
        if "perf/examples_per_second" in row["metrics"]
    ]
    _emit(
        "summary",
        framework="representax",
        process_count=jax.process_count(),
        device_count=jax.device_count(),
        steps=result.completed_iterations,
        global_batch_size=global_batch_size,
        first_loss=float(training[0]["metrics"]["train/loss"]),
        final_loss=float(training[-1]["metrics"]["train/loss"]),
        median_examples_per_second=statistics.median(warm_rates),
        run_directory=str(run_directory),
    )


def _torch_worker(index: int, steps: int, global_batch_size: int) -> None:
    del index
    import torch
    import torch.nn.functional as functional
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    device = torch_xla.device()
    world_size = xr.world_size()
    rank = xr.global_ordinal()
    if global_batch_size % world_size:
        raise ValueError("global batch must be divisible by the global device count")
    local_size = global_batch_size // world_size
    left, right, target = _rows(rank * local_size, local_size)
    weight, bias = _initial_parameters()
    model = torch.nn.Linear(INPUT_DIMENSION, OUTPUT_DIMENSION).to(device)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(weight.T).to(device))
        model.bias.copy_(torch.from_numpy(bias).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    left_tensor = torch.from_numpy(left).to(device)
    right_tensor = torch.from_numpy(right).to(device)
    target_tensor = torch.from_numpy(target).to(device)
    durations = []
    losses = []
    compilation_seconds = None
    for step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        left_embedding = functional.normalize(model(left_tensor), dim=-1)
        right_embedding = functional.normalize(model(right_tensor), dim=-1)
        prediction = torch.sum(left_embedding * right_embedding, dim=-1)
        loss = torch.mean(torch.square(prediction - target_tensor))
        loss.backward()
        xm.all_reduce(
            xm.REDUCE_SUM,
            [parameter.grad for parameter in model.parameters()],
            scale=1.0 / world_size,
        )
        optimizer.step()
        reduced_loss = xm.all_reduce(
            xm.REDUCE_SUM,
            loss.detach(),
            scale=1.0 / world_size,
        )
        torch_xla.sync(wait=True)
        reduced_loss = float(reduced_loss.cpu())
        elapsed = time.perf_counter() - started
        losses.append(reduced_loss)
        if step == 0:
            compilation_seconds = elapsed
        else:
            durations.append(elapsed)
        if rank == 0:
            _emit(
                "training_step",
                framework="torch-xla",
                step=step + 1,
                loss=reduced_loss,
                seconds=elapsed,
            )
    if rank == 0:
        median = statistics.median(durations)
        _emit(
            "summary",
            framework="torch-xla",
            process_count=xr.process_count(),
            device_count=world_size,
            steps=steps,
            global_batch_size=global_batch_size,
            first_loss=losses[0],
            final_loss=losses[-1],
            compilation_and_first_step_seconds=compilation_seconds,
            median_step_seconds=median,
            examples_per_second=global_batch_size / median,
        )


def torch_dense(*, steps: int, global_batch_size: int) -> None:
    import torch_xla

    torch_xla.launch(_torch_worker, args=(steps, global_batch_size))


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    topology = commands.add_parser("topology")
    topology.add_argument("--distributed", action="store_true")
    topology.add_argument(
        "--scope", choices=("local", "global", "both"), default="both"
    )
    topology.add_argument("--payload-mib", type=int, default=16)
    topology.add_argument("--iterations", type=int, default=20)
    for name in ("jax-dense", "torch-dense"):
        dense = commands.add_parser(name)
        dense.add_argument("--steps", type=int, default=STEPS)
        dense.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    representax = commands.add_parser("representax-dense")
    representax.add_argument("--output", type=Path, required=True)
    representax.add_argument("--steps", type=int, default=STEPS)
    representax.add_argument("--global-batch-size", type=int, default=256)
    parsed = parser.parse_args(arguments)
    if parsed.command == "topology":
        jax_topology(
            distributed=parsed.distributed,
            scope=parsed.scope,
            payload_mib=parsed.payload_mib,
            iterations=parsed.iterations,
        )
    elif parsed.command == "jax-dense":
        jax_dense(steps=parsed.steps, global_batch_size=parsed.global_batch_size)
    elif parsed.command == "representax-dense":
        representax_dense(
            output=parsed.output,
            steps=parsed.steps,
            global_batch_size=parsed.global_batch_size,
        )
    else:
        torch_dense(steps=parsed.steps, global_batch_size=parsed.global_batch_size)


if __name__ == "__main__":
    main()
