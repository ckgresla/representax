"""Reusable compiled evaluation for offline and in-training validation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax

from representax.core import Task
from representax.evaluation import Evaluator, LossEvaluator
from representax.precision import (
    FP32_POLICY,
    PrecisionPolicy,
    precision_context,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Host-resident aggregate metrics for one complete evaluation pass."""

    iteration: int
    batches: int
    examples: int
    duration_seconds: float
    compilation_seconds: float
    metrics: Mapping[str, float]


def _batch_size(batch: Any) -> int:
    arrays = [leaf for leaf in jax.tree.leaves(batch) if eqx.is_array(leaf)]
    if not arrays or arrays[0].ndim == 0:
        raise TypeError("evaluation batches require a leading example dimension")
    size = arrays[0].shape[0]
    if any(leaf.ndim == 0 or leaf.shape[0] != size for leaf in arrays[1:]):
        raise ValueError("evaluation batch leaves must share a leading size")
    return size


def _compilation_signature(tree: Any) -> str:
    leaves, structure = jax.tree.flatten(tree)
    document = {
        "structure": str(structure),
        "leaves": [
            {
                "shape": tuple(getattr(leaf, "shape", ())),
                "dtype": str(getattr(leaf, "dtype", "")),
            }
            for leaf in leaves
        ],
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


class EvaluationRunner:
    """Run one typed evaluator and reuse its JAX executable across invocations."""

    def __init__(
        self,
        evaluator: Evaluator[Any, Any],
        *,
        precision: PrecisionPolicy = FP32_POLICY,
    ) -> None:
        self.name = evaluator.name
        children = getattr(evaluator, "evaluators", None)
        self.names = (
            tuple(child.name for child in children)
            if children is not None
            else (evaluator.name,)
        )
        self._seen_signatures: set[str] = set()
        self._evaluator = evaluator

        @eqx.filter_jit
        def compiled(model: eqx.Module, batch: Any, key: Any) -> Any:
            with precision_context(precision):
                return evaluator.evaluate_batch(model, batch, key=key)

        self._compiled = compiled

    def run(
        self,
        model: eqx.Module,
        batches: Iterable[Any],
        *,
        iteration: int = 0,
        key: Any = None,
        max_batches: int | None = None,
        place_batch: Callable[[Any], Any] = jax.device_put,
    ) -> EvaluationResult:
        """Evaluate an iterable using weighted, host-resident metric means."""

        if iteration < 0:
            raise ValueError("evaluation iteration must be non-negative")
        if max_batches is not None and max_batches <= 0:
            raise ValueError("max_batches must be positive or None")
        started = time.perf_counter()
        compilation_seconds = 0.0
        accumulator = self._evaluator.initialize()
        examples = 0
        batch_count = 0
        pending_output: Any = None

        def consume(output: Any) -> None:
            nonlocal accumulator
            arrays = [
                leaf for leaf in jax.tree.leaves(output) if isinstance(leaf, jax.Array)
            ]
            if any(not leaf.is_fully_addressable for leaf in arrays):
                raise NotImplementedError(
                    "multi-host evaluator reduction is not implemented; run this "
                    "evaluation on one fully addressable mesh"
                )
            accumulator = self._evaluator.accumulate(
                accumulator,
                jax.device_get(output),
            )

        iterator = iter(batches)
        try:
            for batch_index, host_batch in enumerate(iterator):
                if max_batches is not None and batch_index >= max_batches:
                    break
                batch = place_batch(host_batch)
                size = _batch_size(batch)
                signature = _compilation_signature(batch)
                first_use = signature not in self._seen_signatures
                batch_key = (
                    None if key is None else jax.random.fold_in(key, batch_index)
                )
                dispatch_started = time.perf_counter()
                output = self._compiled(model, batch, batch_key)
                if first_use:
                    if pending_output is not None:
                        consume(pending_output)
                        pending_output = None
                    consume(output)
                    compilation_seconds += time.perf_counter() - dispatch_started
                    self._seen_signatures.add(signature)
                else:
                    if pending_output is not None:
                        consume(pending_output)
                    pending_output = output
                examples += size
                batch_count += 1
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if pending_output is not None:
            consume(pending_output)
        if batch_count == 0:
            raise ValueError("evaluation batch source produced no batches")
        return EvaluationResult(
            iteration=iteration,
            batches=batch_count,
            examples=examples,
            duration_seconds=time.perf_counter() - started,
            compilation_seconds=compilation_seconds,
            metrics=self._evaluator.finalize(accumulator),
        )


def evaluate(
    model: eqx.Module,
    task: Task[Any],
    batches: Iterable[Any],
    *,
    precision: PrecisionPolicy = FP32_POLICY,
    **kwargs: Any,
) -> EvaluationResult:
    """Run the same loss evaluator used by the training lifecycle offline."""

    return EvaluationRunner(LossEvaluator(task), precision=precision).run(
        model,
        batches,
        **kwargs,
    )


__all__ = ["EvaluationResult", "EvaluationRunner", "evaluate"]
