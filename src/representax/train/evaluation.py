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
import jax.numpy as jnp

from representax.core import Task, evaluate_loss


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


def _metric_names(name: str, loss: Any) -> dict[str, Any]:
    prefix = "valid" if name == "loss" else f"valid/{name}"
    return {f"{prefix}/loss": loss}


class EvaluationRunner:
    """Own one task evaluator and reuse its JAX executable across invocations."""

    def __init__(self, task: Task[Any], *, name: str = "loss") -> None:
        if not name:
            raise ValueError("evaluation name must be non-empty")
        self.name = name
        self._seen_signatures: set[str] = set()
        metric_weight = getattr(task, "accumulation_weight", None)

        @eqx.filter_jit
        def compiled(model: eqx.Module, batch: Any, key: Any) -> Any:
            output = evaluate_loss(task, model, batch, key=key)
            weight = (
                metric_weight(batch)
                if callable(metric_weight)
                else jnp.asarray(_batch_size(batch), dtype=jnp.float32)
            )
            return output, weight

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
        totals: dict[str, float] = {}
        total_weight = 0.0
        examples = 0
        batch_count = 0
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
                    None
                    if key is None
                    else jax.random.fold_in(key, jnp.asarray(batch_index))
                )
                dispatch_started = time.perf_counter()
                output, weight = jax.device_get(self._compiled(model, batch, batch_key))
                if first_use:
                    compilation_seconds += time.perf_counter() - dispatch_started
                    self._seen_signatures.add(signature)
                batch_metrics = _metric_names(self.name, output.loss)
                for metric_name, value in batch_metrics.items():
                    scalar = float(value)
                    totals[metric_name] = totals.get(metric_name, 0.0) + scalar * float(
                        weight
                    )
                total_weight += float(weight)
                examples += size
                batch_count += 1
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if batch_count == 0:
            raise ValueError("evaluation batch source produced no batches")
        return EvaluationResult(
            iteration=iteration,
            batches=batch_count,
            examples=examples,
            duration_seconds=time.perf_counter() - started,
            compilation_seconds=compilation_seconds,
            metrics={
                name: total / max(total_weight, 1.0) for name, total in totals.items()
            },
        )


def evaluate(
    model: eqx.Module,
    task: Task[Any],
    batches: Iterable[Any],
    **kwargs: Any,
) -> EvaluationResult:
    """Run the same loss evaluator used by the training lifecycle offline."""

    return EvaluationRunner(task).run(model, batches, **kwargs)


__all__ = ["EvaluationResult", "EvaluationRunner", "evaluate"]
