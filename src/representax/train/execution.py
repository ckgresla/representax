"""Loss-evaluation schedules used inside the generic optimizer step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import equinox as eqx
import jax

from representax.core import LossOutput, Task, evaluate_loss


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
    ) -> LossOutput:
        return evaluate_loss(task, model, batch, key=key)
