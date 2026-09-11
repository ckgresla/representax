"""Timeline accounting must not double-count overlapping GPU activity."""

import importlib.util
from pathlib import Path

import pytest


def partition_time(*args):
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/15-modernbert-mlm-scaling/analyze_trace.py"
    )
    spec = importlib.util.spec_from_file_location("mlm_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.partition_time(*args)


def test_timeline_unions_same_kind_and_separates_cross_kind_overlap():
    second = 10**9
    intervals = [
        (1, 4, "compute"),
        (3, 6, "compute"),
        (5, 8, "collective"),
        (7, 9, "copy"),
    ]
    result = partition_time(
        0,
        10 * second,
        [(start * second, end * second, kind) for start, end, kind in intervals],
    )
    assert result == pytest.approx(
        {
            "idle": 2,
            "compute": 4,
            "collective+compute": 1,
            "collective": 1,
            "collective+copy": 1,
            "copy": 1,
        }
    )
    assert sum(result.values()) == pytest.approx(10)


def test_timeline_clips_to_optimizer_interval():
    assert partition_time(
        2 * 10**9,
        4 * 10**9,
        [(0, 3 * 10**9, "compute"), (5 * 10**9, 6 * 10**9, "copy")],
    ) == pytest.approx({"compute": 1, "idle": 1})
