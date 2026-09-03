"""Shared optimizer-step timing for PyTorch reference trainers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterable, Sequence
from typing import Any


class CudaStepTimer:
    def __init__(self) -> None:
        self._started: float | None = None
        self.rows: list[tuple[int, float]] = []

    def callback(self) -> Any:
        import torch
        from transformers import TrainerCallback

        owner = self

        class Callback(TrainerCallback):
            def on_step_begin(
                self, _args: Any, _state: Any, control: Any, **_: Any
            ) -> Any:
                torch.cuda.synchronize()
                owner._started = time.perf_counter()
                return control

            def on_step_end(
                self, _args: Any, state: Any, control: Any, **_: Any
            ) -> Any:
                torch.cuda.synchronize()
                if owner._started is None:
                    raise RuntimeError("optimizer-step timer ended without starting")
                owner.rows.append(
                    (int(state.global_step), time.perf_counter() - owner._started)
                )
                owner._started = None
                return control

        return Callback()


def warm_step_summary(
    rows: Sequence[tuple[int, float]],
    *,
    batch_size: int,
    excluded_steps: Iterable[int] = (1,),
) -> dict[str, float | int]:
    excluded = frozenset(excluded_steps)
    durations = [
        duration
        for step, duration in rows
        if step not in excluded and duration > 0
    ]
    if not durations:
        raise ValueError("reference run emitted no warmed optimizer-step durations")
    return {
        "measured_steps": len(durations),
        "median_step_seconds": statistics.median(durations),
        "examples_per_second": batch_size * len(durations) / sum(durations),
    }


__all__ = ["CudaStepTimer", "warm_step_summary"]
