"""Repository-only dense-retrieval evidence contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from benchmarks.dense_retrieval import _aggregate, _three_run_interval


def _summary(seed: int, speedup: float) -> dict:
    return {
        "schema_version": "representax-dense-retrieval-comparison-v1",
        "contract": {"model": "tiny", "batch_size": 8, "seed": seed},
        "representax": {
            "sustained_examples_per_second_after_first_step": 100.0 * speedup,
        },
        "sentence_transformers": {"amortized_examples_per_second": 100.0},
        "comparison": {
            "initial_metric_parity": True,
            "sustained_training_speedup": speedup,
            "final_ndcg@10_difference": seed / 1_000.0,
            "compilation_break_even_steps": 200.0 + seed,
        },
    }


def _write_summary(path: Path, seed: int, speedup: float) -> Path:
    path.write_text(json.dumps(_summary(seed, speedup)))
    return path


def test_three_run_interval_reports_student_t_confidence_bounds():
    interval = _three_run_interval([1.0, 2.0, 3.0])

    assert interval["mean"] == 2.0
    assert interval["confidence_level"] == 0.95
    assert interval["confidence_interval"] == pytest.approx(
        [-0.484137711719546, 4.484137711719546]
    )


def test_dense_retrieval_aggregate_sorts_and_fingerprints_three_seeds(tmp_path):
    summaries = [
        _write_summary(tmp_path / f"seed-{seed}.json", seed, speedup)
        for seed, speedup in ((73, 1.3), (17, 1.1), (42, 1.2))
    ]
    output = tmp_path / "aggregate.json"

    _aggregate(argparse.Namespace(summary=summaries, output=output))

    aggregate = json.loads(output.read_text())
    assert aggregate["seeds"] == [17, 42, 73]
    assert aggregate["contract"] == {"model": "tiny", "batch_size": 8}
    assert aggregate["metrics"]["sustained_training_speedup"]["mean"] == 1.2
    assert all(item["sha256"].startswith("sha256:") for item in aggregate["inputs"])


def test_dense_retrieval_aggregate_rejects_contract_drift(tmp_path):
    summaries = [
        _write_summary(tmp_path / f"seed-{seed}.json", seed, 1.1)
        for seed in (17, 42, 73)
    ]
    changed = json.loads(summaries[-1].read_text())
    changed["contract"]["batch_size"] = 16
    summaries[-1].write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="contracts differ"):
        _aggregate(
            argparse.Namespace(summary=summaries, output=tmp_path / "aggregate.json")
        )
