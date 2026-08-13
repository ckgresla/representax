"""Topology-neutral single-device training orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from representax.planning import ScientificSpec

from .logging import EventSink, RunLogger, TrainingStepRecord
from .state import StepResult, TrainState
from .step import TrainStep


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Mechanics of the host loop, separate from scientific semantics."""

    console_every: int = 1
    start_iteration: int = 0

    def __post_init__(self) -> None:
        if self.console_every <= 0:
            raise ValueError("console_every must be positive")
        if self.start_iteration < 0:
            raise ValueError("start_iteration must be non-negative")


@dataclass(frozen=True)
class TrainingRunResult:
    """Final state and durable location of a completed training loop."""

    state: TrainState
    completed_iterations: int
    run_directory: Path


def _float(value: Any) -> float:
    return float(jax.device_get(value))


def _int(value: Any) -> int:
    return int(jax.device_get(value))


def _bool(value: Any) -> bool:
    return bool(jax.device_get(value))


def _compilation_signature(tree: Any) -> str:
    leaves, structure = jax.tree.flatten(tree)
    description = {
        "structure": str(structure),
        "leaves": [
            {
                "type": type(leaf).__qualname__,
                "shape": tuple(getattr(leaf, "shape", ())),
                "dtype": str(getattr(leaf, "dtype", "")),
            }
            for leaf in leaves
        ],
    }
    payload = json.dumps(description, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _close_batches(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _record(
    *,
    iteration: int,
    result: StepResult,
    science: ScientificSpec,
    data_wait_seconds: float,
    placement_seconds: float,
    step_seconds: float,
    first_use: bool,
) -> TrainingStepRecord:
    compiled_step_seconds = None if first_use else step_seconds
    compilation_and_first_step_seconds = step_seconds if first_use else None
    end_to_end_seconds = data_wait_seconds + placement_seconds + step_seconds
    return TrainingStepRecord(
        iteration=iteration,
        optimizer_step=_int(result.state.step),
        loss=_float(result.metrics.loss),
        task={
            name: jax.device_get(value) for name, value in result.metrics.task.items()
        },
        gradient_global_norm=_float(result.metrics.gradient_global_norm),
        clipped_gradient_global_norm=_float(
            result.metrics.clipped_gradient_global_norm
        ),
        update_global_norm=_float(result.metrics.update_global_norm),
        numeric_finite=_bool(result.metrics.numeric_finite),
        skipped_update=_bool(result.metrics.skipped_update),
        data_wait_seconds=data_wait_seconds,
        placement_seconds=placement_seconds,
        compiled_step_seconds=compiled_step_seconds,
        compilation_and_first_step_seconds=compilation_and_first_step_seconds,
        end_to_end_seconds=end_to_end_seconds,
        examples_per_second=(
            None
            if compiled_step_seconds is None
            else science.global_batch_size / compiled_step_seconds
        ),
        end_to_end_examples_per_second=(
            science.global_batch_size / end_to_end_seconds
        ),
    )


def run_training(
    *,
    state: TrainState,
    step: TrainStep,
    batches: Iterable[Any],
    science: ScientificSpec,
    run_directory: str | Path,
    config: TrainingLoopConfig | None = None,
    sinks: tuple[EventSink, ...] = (),
    place_batch: Callable[[Any], Any] = jax.device_put,
) -> TrainingRunResult:
    """Run model-ready batches through one compiled single-device update.

    ``batches`` may be a Grain ``IterDataset`` or any iterator with the same
    model-ready batch contract. The loop owns and closes the iterator it creates.
    Scientific step count, batch size, and seed come from ``science``; host-loop
    logging cadence remains a separate execution concern.
    """

    config = TrainingLoopConfig() if config is None else config
    if config.start_iteration >= science.max_steps:
        raise ValueError("start_iteration must precede science.max_steps")
    source_batch_size = getattr(batches, "global_batch_size", None)
    if (
        source_batch_size is not None
        and source_batch_size != science.global_batch_size
    ):
        raise ValueError(
            "batch source global_batch_size differs from the scientific "
            f"specification: {source_batch_size} != {science.global_batch_size}"
        )
    run_path = Path(run_directory).expanduser().resolve()
    logger = RunLogger(
        run_path,
        manifest={
            "task": science.task,
            "global_batch_size": science.global_batch_size,
            "max_steps": science.max_steps,
            "seed": science.seed,
            "start_iteration": config.start_iteration,
        },
        sinks=sinks,
    )
    iterator = None
    iterator_closed = False
    current = state
    completed = config.start_iteration
    base_key = jax.random.key(science.seed)
    seen_signatures: set[str] = set()
    try:
        iterator = iter(batches)
        logger.event(
            "training_started",
            iteration=config.start_iteration,
            end_iteration=science.max_steps,
        )
        for iteration_index in range(config.start_iteration, science.max_steps):
            wait_started = time.perf_counter()
            try:
                host_batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "batch source exhausted before science.max_steps"
                ) from error
            data_wait_seconds = time.perf_counter() - wait_started

            placement_started = time.perf_counter()
            batch = place_batch(host_batch)
            jax.block_until_ready(batch)
            placement_seconds = time.perf_counter() - placement_started

            signature = _compilation_signature(batch)
            first_use = signature not in seen_signatures
            if first_use:
                logger.event(
                    "executable_first_use_started",
                    iteration=iteration_index,
                    signature=signature,
                )
            step_key = jax.random.fold_in(base_key, iteration_index)
            step_started = time.perf_counter()
            update = step(current, batch, step_key)
            jax.block_until_ready(update)
            step_seconds = time.perf_counter() - step_started
            if first_use:
                seen_signatures.add(signature)
                logger.event(
                    "executable_first_use_finished",
                    iteration=iteration_index,
                    signature=signature,
                    duration_seconds=step_seconds,
                    includes_execution=True,
                )

            completed = iteration_index + 1
            current = update.state
            record = _record(
                iteration=completed,
                result=update,
                science=science,
                data_wait_seconds=data_wait_seconds,
                placement_seconds=placement_seconds,
                step_seconds=step_seconds,
                first_use=first_use,
            )
            logger.metrics(record)
            if record.skipped_update:
                logger.event("nonfinite_update_skipped", iteration=completed)
            if completed % config.console_every == 0:
                logger.console_metrics(record)

        _close_batches(iterator)
        iterator_closed = True
        logger.event(
            "training_finished",
            iteration=completed,
            optimizer_step=_int(current.step),
        )
        logger.finish(
            "completed",
            completed_iterations=completed,
            optimizer_steps=_int(current.step),
        )
        return TrainingRunResult(
            state=current,
            completed_iterations=completed,
            run_directory=run_path,
        )
    except BaseException as error:
        if iterator is not None and not iterator_closed:
            try:
                _close_batches(iterator)
                iterator_closed = True
            except BaseException as close_error:
                logger.event(
                    "batch_source_close_failed",
                    iteration=completed,
                    error_type=type(close_error).__name__,
                    error=str(close_error),
                )
        logger.event(
            "training_failed",
            iteration=completed,
            error_type=type(error).__name__,
            error=str(error),
        )
        logger.finish(
            "failed",
            completed_iterations=completed,
            optimizer_steps=_int(current.step),
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        logger.close()


__all__ = ["TrainingLoopConfig", "TrainingRunResult", "run_training"]
