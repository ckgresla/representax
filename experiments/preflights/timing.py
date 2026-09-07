"""Shared optimizer-step timing for PyTorch reference trainers."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import IO, Any

from experiments.preflights.accelerator import torch_synchronize


class CudaStepTimer:
    def __init__(self, output: Path | None = None) -> None:
        self._started: float | None = None
        self.rows: list[tuple[int, float]] = []
        self._stream: IO[str] | None = None
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            self._stream = output.open("x", encoding="utf-8", buffering=1)

    def record(self, step: int, duration: float) -> None:
        self.rows.append((step, duration))
        if self._stream is not None:
            self._stream.write(
                json.dumps(
                    {"step": step, "duration_seconds": duration},
                    sort_keys=True,
                )
                + "\n"
            )

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def restart(self) -> None:
        """Start the next interval after untimed evaluation or checkpoint work."""

        torch_synchronize()
        self._started = time.perf_counter()

    def callback(self, *, stop_after: int | None = None) -> Any:
        from transformers import TrainerCallback

        owner = self

        class Callback(TrainerCallback):
            def on_train_begin(
                self, _args: Any, _state: Any, control: Any, **_: Any
            ) -> Any:
                torch_synchronize()
                owner._started = time.perf_counter()
                return control

            def on_step_end(
                self, _args: Any, state: Any, control: Any, **_: Any
            ) -> Any:
                torch_synchronize()
                if owner._started is None:
                    raise RuntimeError("optimizer-step timer ended without starting")
                completed_at = time.perf_counter()
                owner.record(
                    int(state.global_step), completed_at - owner._started
                )
                owner._started = completed_at
                if stop_after is not None and int(state.global_step) >= stop_after:
                    control.should_training_stop = True
                return control

            def on_evaluate(
                self, _args: Any, _state: Any, control: Any, **_: Any
            ) -> Any:
                owner.restart()
                return control

            def on_save(
                self, _args: Any, _state: Any, control: Any, **_: Any
            ) -> Any:
                owner.restart()
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
