"""Loss-evaluation schedules used inside the generic optimizer step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import equinox as eqx
import jax

from representax.core import LossOutput, Task, evaluate_loss


@dataclass(frozen=True)
class ExecutionContext:
    """Named collective axes available while evaluating a loss."""

    data_axis_name: str | None = None


_LOCAL_EXECUTION_CONTEXT = ExecutionContext()


class LossExecution(Protocol):
    """Static strategy for evaluating a task loss before one optimizer update."""

    def validate(self, task: Task[Any]) -> None: ...

    def evaluate(
        self,
        task: Task[Any],
        model: eqx.Module,
        batch: Any,
        *,
        key: jax.Array | None,
        context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    ) -> LossOutput: ...


@dataclass(frozen=True)
class Direct:
    """Differentiate the task's ordinary full-batch computation."""

    def validate(self, task: Task[Any]) -> None:
        del task

    def evaluate(
        self,
        task: Task[Any],
        model: eqx.Module,
        batch: Any,
        *,
        key: jax.Array | None,
        context: ExecutionContext = _LOCAL_EXECUTION_CONTEXT,
    ) -> LossOutput:
        if context.data_axis_name is not None:
            raise TypeError("distributed Direct execution is not implemented")
        return evaluate_loss(task, model, batch, key=key)
