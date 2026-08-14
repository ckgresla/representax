"""Topology-neutral single-device training orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax

from representax.planning import ScientificSpec

from .checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    science_fingerprint,
    training_checkpointables,
)
from .logging import EventSink, RunLogger, TrainingStepRecord
from .state import StepResult, TrainState
from .step import TrainStep


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Mechanics of the host loop, separate from scientific semantics."""

    console_every: int = 1

    def __post_init__(self) -> None:
        if self.console_every <= 0:
            raise ValueError("console_every must be positive")


@dataclass(frozen=True)
class TrainingRunResult:
    """Final state and durable location of a completed training loop."""

    state: TrainState
    completed_iterations: int
    run_directory: Path
    resumed: bool


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


def _get_iterator_state(iterator: Any) -> Mapping[str, Any]:
    get_state = getattr(iterator, "get_state", None)
    if get_state is None:
        raise TypeError(
            "checkpointed training requires an iterator with get_state/set_state"
        )
    state = get_state()
    if not isinstance(state, Mapping):
        raise TypeError("iterator get_state() must return a mapping")
    return state


def _set_iterator_state(iterator: Any, state: Mapping[str, Any]) -> None:
    set_state = getattr(iterator, "set_state", None)
    if set_state is None:
        raise TypeError(
            "checkpointed training requires an iterator with get_state/set_state"
        )
    set_state(state)


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
        end_to_end_examples_per_second=(science.global_batch_size / end_to_end_seconds),
    )


def run_training(
    *,
    state: TrainState,
    step: TrainStep,
    batches: Iterable[Any],
    science: ScientificSpec,
    run_directory: str | Path,
    config: TrainingLoopConfig | None = None,
    checkpoint: CheckpointConfig | None = None,
    resume: bool = False,
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
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint configuration")
    if checkpoint is not None and any(
        iteration > science.max_steps for iteration in checkpoint.additional_iterations
    ):
        raise ValueError(
            "additional checkpoint iterations cannot exceed science.max_steps"
        )
    source_batch_size = getattr(batches, "global_batch_size", None)
    if source_batch_size is not None and source_batch_size != science.global_batch_size:
        raise ValueError(
            "batch source global_batch_size differs from the scientific "
            f"specification: {source_batch_size} != {science.global_batch_size}"
        )
    run_path = Path(run_directory).expanduser().resolve()
    fingerprint = science_fingerprint(science)
    manifest = {
        "task": science.task,
        "global_batch_size": science.global_batch_size,
        "max_steps": science.max_steps,
        "seed": science.seed,
        "science_fingerprint": fingerprint,
        "checkpoint": None if checkpoint is None else asdict(checkpoint),
    }
    logger = None
    checkpoint_manager = None
    iterator = None
    iterator_closed = False
    current = state
    completed = 0
    base_key = jax.random.key(science.seed)
    seen_signatures: set[str] = set()
    try:
        restored = None
        if resume:
            checkpoint_manager = CheckpointManager(
                run_path,
                science_fingerprint=fingerprint,
                keep=checkpoint.keep,
                asynchronous=checkpoint.asynchronous,
            )
            restore_started = time.perf_counter()
            restored = checkpoint_manager.restore_training_state(state)
            restore_seconds = time.perf_counter() - restore_started
            current = restored.state
            completed = restored.iteration
            base_key = restored.rng
            if completed >= science.max_steps:
                raise ValueError(
                    "checkpoint already reached or exceeded science.max_steps"
                )
            logger = RunLogger(
                run_path,
                manifest=manifest,
                sinks=sinks,
                resume_cursor=restored.logging_cursor,
            )
            checkpoint_manager.set_event_callback(logger.event)
            logger.event(
                "checkpoint_restored",
                iteration=completed,
                checkpoint_path=str(restored.record.path),
                duration_seconds=restore_seconds,
            )
            logger.event(
                "training_resumed",
                iteration=completed,
                optimizer_step=_int(current.step),
                checkpoint_path=str(restored.record.path),
                checkpoint_fingerprint=restored.record.checkpoint_fingerprint,
            )
        else:
            logger = RunLogger(run_path, manifest=manifest, sinks=sinks)
            if checkpoint is not None:
                checkpoint_manager = CheckpointManager(
                    run_path,
                    science_fingerprint=fingerprint,
                    keep=checkpoint.keep,
                    asynchronous=checkpoint.asynchronous,
                    event=logger.event,
                )
        iterator = iter(batches)
        if checkpoint_manager is not None:
            _get_iterator_state(iterator)
        if restored is not None:
            _set_iterator_state(iterator, restored.data_state)
        if not resume:
            logger.event(
                "training_started",
                iteration=0,
                end_iteration=science.max_steps,
            )
        for iteration_index in range(completed, science.max_steps):
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

            final = completed == science.max_steps
            if checkpoint_manager is not None and checkpoint.should_save(
                completed, final=final
            ):
                checkpoint_manager.save(
                    completed,
                    training_checkpointables(
                        state=current,
                        iteration=completed,
                        rng=base_key,
                        data_state=_get_iterator_state(iterator),
                        logging_cursor=logger.cursor(),
                    ),
                    metrics={
                        "loss": record.loss,
                        "optimizer_step": record.optimizer_step,
                    },
                )

        _close_batches(iterator)
        iterator_closed = True
        if checkpoint_manager is not None:
            checkpoint_manager.close()
            checkpoint_manager = None
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
            resumed=resume,
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
        if checkpoint_manager is not None:
            try:
                checkpoint_manager.close()
                checkpoint_manager = None
            except BaseException as close_error:
                if logger is not None:
                    logger.event(
                        "checkpoint_close_failed",
                        iteration=completed,
                        error_type=type(close_error).__name__,
                        error=str(close_error),
                    )
        if logger is not None:
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
        if logger is not None:
            logger.close()


__all__ = ["TrainingLoopConfig", "TrainingRunResult", "run_training"]
