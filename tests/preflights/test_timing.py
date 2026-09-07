"""Reference optimizer-step timing contracts."""

import json
from pathlib import Path

import pytest
from experiments.preflights.timing import CudaStepTimer, warm_step_summary


def test_step_timer_persists_each_completed_step(tmp_path: Path) -> None:
    output = tmp_path / "steps.jsonl"
    timer = CudaStepTimer(output)

    timer.record(1, 1.25)
    timer.record(2, 0.5)
    timer.close()

    assert timer.rows == [(1, 1.25), (2, 0.5)]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [
        {"duration_seconds": 1.25, "step": 1},
        {"duration_seconds": 0.5, "step": 2},
    ]


def test_step_timer_callback_uses_completion_boundaries(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    clock = iter((10.0, 13.0, 17.5, 18.0))
    monkeypatch.setattr("experiments.preflights.timing.time.perf_counter", lambda: next(clock))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    timer = CudaStepTimer()
    callback = timer.callback(stop_after=2)
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(None, None, control)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)
    callback.on_step_end(None, SimpleNamespace(global_step=2), control)

    assert timer.rows == [(1, 3.0), (2, 4.5)]
    assert control.should_training_stop


def test_step_timer_can_restart_after_untimed_work(monkeypatch) -> None:
    from types import SimpleNamespace

    import torch

    clock = iter((10.0, 13.0, 20.0, 24.0))
    monkeypatch.setattr(
        "experiments.preflights.timing.time.perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    timer = CudaStepTimer()
    callback = timer.callback()
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(None, None, control)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)
    timer.restart()
    callback.on_step_end(None, SimpleNamespace(global_step=2), control)

    assert timer.rows == [(1, 3.0), (2, 4.0)]


@pytest.mark.parametrize("event", ("on_evaluate", "on_save"))
def test_step_timer_restarts_after_trainer_overhead(monkeypatch, event: str) -> None:
    from types import SimpleNamespace

    import torch

    clock = iter((10.0, 13.0, 20.0, 24.0))
    monkeypatch.setattr(
        "experiments.preflights.timing.time.perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    timer = CudaStepTimer()
    callback = timer.callback()
    control = SimpleNamespace(should_training_stop=False)
    callback.on_train_begin(None, None, control)
    callback.on_step_end(None, SimpleNamespace(global_step=1), control)
    getattr(callback, event)(None, None, control)
    callback.on_step_end(None, SimpleNamespace(global_step=2), control)

    assert timer.rows == [(1, 3.0), (2, 4.0)]


def test_warm_step_summary_excludes_each_cold_restart() -> None:
    summary = warm_step_summary(
        ((1, 9.0), (2, 2.0), (3, 4.0), (4, 8.0), (5, 2.0)),
        batch_size=12,
        excluded_steps=(1, 4),
    )

    assert summary == {
        "measured_steps": 3,
        "median_step_seconds": 2.0,
        "examples_per_second": pytest.approx(4.5),
    }


def test_warm_step_summary_rejects_an_empty_measurement() -> None:
    with pytest.raises(ValueError, match="no warmed optimizer-step durations"):
        warm_step_summary(((1, 1.0),), batch_size=8)
