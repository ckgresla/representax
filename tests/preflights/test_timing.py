"""Reference optimizer-step timing contracts."""

import pytest
from experiments.preflights.timing import warm_step_summary


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
