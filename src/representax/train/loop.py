"""Topology-neutral single-device training orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax

from representax.config import JobConfig, ParameterRole

from .checkpoint import (
    CheckpointManager,
    scientific_fingerprint,
    training_checkpointables,
)
from .evaluation import EvaluationRunner
from .logging import MetricRecord, Reporter, RunLogger
from .state import StepResult, TrainState
from .step import TrainStep


@dataclass(frozen=True)
class TrainingRunResult:
    """Final state and durable location of a completed training loop."""

    state: TrainState
    completed_iterations: int
    run_directory: Path
    resumed: bool
    selected_model: Any
    best_iteration: int | None = None
    best_metrics: Mapping[str, Any] | None = None
    inference_bundle: Path | None = None


def _int(value: Any) -> int:
    return int(jax.device_get(value))


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


def _training_metric_record(
    *,
    iteration: int,
    result: StepResult,
    data_wait_seconds: float,
    placement_enqueue_seconds: float,
    step_dispatch_seconds: float,
    compilation_and_first_step_seconds: float | None,
) -> MetricRecord:
    values: dict[str, Any] = {
        "train/loss": result.metrics.loss,
        "train/gradient_global_norm": result.metrics.gradient_global_norm,
        "train/clipped_gradient_global_norm": (
            result.metrics.clipped_gradient_global_norm
        ),
        "train/update_global_norm": result.metrics.update_global_norm,
        "train/numeric_finite": result.metrics.numeric_finite,
        "train/skipped_update": result.metrics.skipped_update,
        "perf/data_wait_seconds": data_wait_seconds,
        "perf/placement_enqueue_seconds": placement_enqueue_seconds,
        "perf/step_dispatch_seconds": step_dispatch_seconds,
    }
    for name, value in result.metrics.task.items():
        metric_name = name if name.startswith("train/") else f"train/{name}"
        if metric_name in values:
            raise ValueError(f"task metric name collides with {metric_name!r}")
        values[metric_name] = value
    if compilation_and_first_step_seconds is not None:
        values["perf/compilation_and_first_step_seconds"] = (
            compilation_and_first_step_seconds
        )
    return MetricRecord(
        iteration=iteration,
        values=values,
    )


def run_training(
    *,
    state: TrainState,
    step: TrainStep,
    batches: Iterable[Any],
    job: JobConfig,
    run_directory: str | Path,
    resume: bool = False,
    reporters: tuple[Reporter, ...] = (),
    place_batch: Callable[[Any], Any] = jax.device_put,
    evaluation_runners: tuple[EvaluationRunner, ...] = (),
    evaluation_batches: Callable[[], Iterable[Any]] | None = None,
) -> TrainingRunResult:
    """Run model-ready batches through one compiled single-device update.

    ``batches`` may be a Grain ``IterDataset`` or any iterator with the same
    model-ready batch contract. The loop owns and closes the iterator it creates.
    Step count, scientific batch size, execution choices, logging, and
    checkpointing come from the validated ``job`` configuration.
    """

    training = job.training
    logging = job.logging
    checkpoint = job.checkpointing
    evaluation = job.evaluation
    if evaluation is None:
        if evaluation_runners or evaluation_batches is not None:
            raise ValueError("evaluation runtime requires job.evaluation")
    else:
        if evaluation_batches is None or not evaluation_runners:
            raise ValueError(
                "job.evaluation requires evaluation runners and a batch factory"
            )
        configured_names = tuple(item.name for item in evaluation.evaluators)
        runtime_names = tuple(runner.name for runner in evaluation_runners)
        if runtime_names != configured_names:
            raise ValueError(
                "evaluation runner names differ from configuration: "
                f"{runtime_names!r} != {configured_names!r}"
            )
    if resume and checkpoint is None:
        raise ValueError("resume requires checkpoint configuration")
    source_batch_size = getattr(batches, "global_batch_size", None)
    if (
        source_batch_size is not None
        and source_batch_size != training.global_batch_size
    ):
        raise ValueError(
            "batch source global_batch_size differs from training config: "
            f"{source_batch_size} != {training.global_batch_size}"
        )
    data_contract = getattr(batches, "data_contract", None)
    data_fingerprint = getattr(batches, "data_fingerprint", None)
    if data_contract is not None and not isinstance(data_contract, Mapping):
        raise TypeError("batch source data_contract must be a mapping")
    if data_fingerprint is not None and not isinstance(data_fingerprint, str):
        raise TypeError("batch source data_fingerprint must be a string")
    if checkpoint is not None and (not data_contract or not data_fingerprint):
        raise TypeError(
            "checkpointed training requires a batch source with a stable "
            "data_contract and data_fingerprint"
        )
    run_path = Path(run_directory).expanduser().resolve()
    fingerprint = scientific_fingerprint(job)
    initial_optimizer_step = _int(state.step)
    manifest = {
        "config": job.model_dump(mode="json"),
        "scientific": job.parameters(ParameterRole.SCIENTIFIC),
        "scientific_fingerprint": fingerprint,
        "execution": job.parameters(ParameterRole.EXECUTION),
        "data_contract": data_contract,
        "data_fingerprint": data_fingerprint,
        "initial_optimizer_step": initial_optimizer_step,
    }
    logger = None
    checkpoint_manager = None
    iterator = None
    iterator_closed = False
    current = state
    completed = 0
    base_key = jax.random.key(training.seed)
    seen_signatures: set[str] = set()
    pending_metric: MetricRecord | None = None
    run_failed = False
    best_iteration: int | None = None
    best_metrics: Mapping[str, Any] | None = None
    try:
        restored = None
        if resume:
            if checkpoint is None:  # pragma: no cover - guarded before setup
                raise AssertionError("resume requires checkpoint configuration")
            if not isinstance(
                data_fingerprint, str
            ):  # pragma: no cover - guarded above
                raise AssertionError("checkpointed training requires data fingerprint")
            checkpoint_manager = CheckpointManager(
                run_path,
                scientific_fingerprint=fingerprint,
                data_fingerprint=data_fingerprint,
                keep=checkpoint.keep,
                best_metric=(
                    evaluation.primary_metric
                    if evaluation is not None and evaluation.save_best
                    else None
                ),
                best_mode=(
                    evaluation.primary_metric_mode if evaluation is not None else "min"
                ),
                keep_best=(evaluation.keep_best if evaluation is not None else 1),
                asynchronous=checkpoint.asynchronous,
            )
            restore_started = time.perf_counter()
            restored = checkpoint_manager.restore_training_state(state)
            restore_seconds = time.perf_counter() - restore_started
            current = restored.state
            completed = restored.iteration
            base_key = restored.rng
            if completed >= training.max_steps:
                raise ValueError(
                    "checkpoint already reached or exceeded training.max_steps"
                )
            logger = RunLogger(
                run_path,
                manifest=manifest,
                reporters=reporters,
                resume_cursor=restored.logging_cursor,
                queue_size=logging.reporter_queue_size,
                initial_optimizer_step=_int(current.step),
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
            logger = RunLogger(
                run_path,
                manifest=manifest,
                reporters=reporters,
                queue_size=logging.reporter_queue_size,
                initial_optimizer_step=initial_optimizer_step,
            )
            if checkpoint is not None:
                if not isinstance(
                    data_fingerprint, str
                ):  # pragma: no cover - guarded above
                    raise AssertionError(
                        "checkpointed training requires data fingerprint"
                    )
                checkpoint_manager = CheckpointManager(
                    run_path,
                    scientific_fingerprint=fingerprint,
                    data_fingerprint=data_fingerprint,
                    keep=checkpoint.keep,
                    best_metric=(
                        evaluation.primary_metric
                        if evaluation is not None and evaluation.save_best
                        else None
                    ),
                    best_mode=(
                        evaluation.primary_metric_mode
                        if evaluation is not None
                        else "min"
                    ),
                    keep_best=(evaluation.keep_best if evaluation is not None else 1),
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
                end_iteration=training.max_steps,
            )

        def run_evaluation(iteration: int) -> dict[str, float]:
            if evaluation is None or evaluation_batches is None:
                raise AssertionError("evaluation runtime is not configured")
            logger.event("evaluation_started", iteration=iteration)
            started = time.perf_counter()
            metrics: dict[str, float] = {}
            compilation_seconds = 0.0
            evaluated_examples = 0
            evaluated_batches = 0
            for runner in evaluation_runners:
                result = runner.run(
                    current.model,
                    evaluation_batches(),
                    iteration=iteration,
                    key=None,
                    max_batches=evaluation.max_batches,
                    place_batch=place_batch,
                )
                overlap = set(metrics) & set(result.metrics)
                if overlap:
                    raise ValueError(
                        f"evaluation metric names collide: {sorted(overlap)}"
                    )
                metrics.update(result.metrics)
                compilation_seconds += result.compilation_seconds
                evaluated_examples += result.examples
                evaluated_batches += result.batches
            if evaluation.primary_metric not in metrics:
                raise ValueError(
                    "primary evaluation metric was not produced: "
                    f"{evaluation.primary_metric!r}"
                )
            duration_seconds = time.perf_counter() - started
            logger.metrics(
                MetricRecord(
                    iteration=iteration,
                    event="evaluation",
                    values={
                        **metrics,
                        "perf/evaluation_seconds": duration_seconds,
                        "perf/evaluation_compilation_seconds": compilation_seconds,
                    },
                ),
                console=True,
            )
            logger.event(
                "evaluation_finished",
                iteration=iteration,
                batches=evaluated_batches,
                examples=evaluated_examples,
                duration_seconds=duration_seconds,
                primary_metric=evaluation.primary_metric,
                primary_value=metrics[evaluation.primary_metric],
            )
            return metrics

        def save_checkpoint(
            iteration: int,
            metrics: Mapping[str, Any] | None = None,
        ) -> None:
            if checkpoint_manager is None:
                raise AssertionError("checkpoint manager is not configured")
            checkpoint_manager.save(
                iteration,
                training_checkpointables(
                    state=current,
                    iteration=iteration,
                    rng=base_key,
                    data_state=_get_iterator_state(iterator),
                    logging_cursor=logger.cursor(),
                ),
                metrics=metrics,
            )

        if not resume and evaluation is not None and evaluation.on_start:
            validation_metrics = run_evaluation(0)
            if evaluation.save_best:
                save_checkpoint(0, validation_metrics)

        for iteration_index in range(completed, training.max_steps):
            wait_started = time.perf_counter()
            try:
                host_batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "batch source exhausted before training.max_steps"
                ) from error
            data_wait_seconds = time.perf_counter() - wait_started

            placement_started = time.perf_counter()
            batch = place_batch(host_batch)
            placement_enqueue_seconds = time.perf_counter() - placement_started

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
            step_dispatch_seconds = time.perf_counter() - step_started
            compilation_and_first_step_seconds = None
            if first_use:
                jax.block_until_ready(update)
                compilation_and_first_step_seconds = time.perf_counter() - step_started
                seen_signatures.add(signature)
                logger.event(
                    "executable_first_use_finished",
                    iteration=iteration_index,
                    signature=signature,
                    duration_seconds=compilation_and_first_step_seconds,
                    includes_execution=True,
                )

            completed = iteration_index + 1
            current = update.state
            record = _training_metric_record(
                iteration=completed,
                result=update,
                data_wait_seconds=data_wait_seconds,
                placement_enqueue_seconds=placement_enqueue_seconds,
                step_dispatch_seconds=step_dispatch_seconds,
                compilation_and_first_step_seconds=(compilation_and_first_step_seconds),
            )
            if pending_metric is not None:
                logger.metrics(
                    pending_metric,
                    console=pending_metric.iteration % logging.console_every == 0,
                )
            pending_metric = record

            final = completed == training.max_steps
            should_evaluate = evaluation is not None and (
                (
                    evaluation.every_steps is not None
                    and completed % evaluation.every_steps == 0
                )
                or (final and evaluation.on_end)
            )
            validation_metrics = None
            if should_evaluate:
                logger.metrics(
                    pending_metric,
                    console=pending_metric.iteration % logging.console_every == 0,
                )
                pending_metric = None
                validation_metrics = run_evaluation(completed)
            should_checkpoint = (
                checkpoint_manager is not None
                and checkpoint is not None
                and (
                    checkpoint.should_save(completed, final=final)
                    or (
                        validation_metrics is not None
                        and evaluation is not None
                        and evaluation.save_best
                    )
                )
            )
            if should_checkpoint:
                if pending_metric is not None:
                    logger.metrics(
                        pending_metric,
                        console=(pending_metric.iteration % logging.console_every == 0),
                    )
                    pending_metric = None
                save_checkpoint(completed, validation_metrics)

        if pending_metric is not None:
            logger.metrics(
                pending_metric,
                console=pending_metric.iteration % logging.console_every == 0,
            )
            pending_metric = None
        _close_batches(iterator)
        iterator_closed = True
        selected_model = current.model
        if checkpoint_manager is not None:
            if evaluation is not None and evaluation.save_best:
                best_record = checkpoint_manager.best()
                best_iteration = best_record.iteration
                manifest_metrics = best_record.manifest.get("metrics")
                if isinstance(manifest_metrics, Mapping):
                    best_metrics = manifest_metrics
                if job.export.selection == "best":
                    restored_model, _ = checkpoint_manager.restore(
                        {"model": current.model},
                        iteration=best_iteration,
                    )
                    selected_model = restored_model["model"]
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
            selected_model=selected_model,
            best_iteration=best_iteration,
            best_metrics=best_metrics,
        )
    except BaseException as error:
        run_failed = True
        if logger is not None and pending_metric is not None:
            with suppress(BaseException):
                logger.metrics(
                    pending_metric,
                    console=pending_metric.iteration % logging.console_every == 0,
                )
            pending_metric = None
        if iterator is not None and not iterator_closed:
            try:
                _close_batches(iterator)
                iterator_closed = True
            except BaseException as close_error:
                if logger is not None:
                    with suppress(BaseException):
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
                    with suppress(BaseException):
                        logger.event(
                            "checkpoint_close_failed",
                            iteration=completed,
                            error_type=type(close_error).__name__,
                            error=str(close_error),
                        )
        if logger is not None:
            try:
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
            except BaseException:
                pass
        raise
    finally:
        if logger is not None:
            try:
                logger.close()
            except BaseException:
                if not run_failed:
                    raise


__all__ = ["TrainingRunResult", "run_training"]
