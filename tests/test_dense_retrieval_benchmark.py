"""Repository-only dense-retrieval evidence contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from benchmarks.dense_retrieval import (
    MODEL_SPECS,
    _aggregate,
    _reference_steady_state_summary,
    _representax_worker_flags,
    _shared_worker_flags,
    _three_run_interval,
    _time_to_quality_summary,
    _training_compile_summary,
    _training_steady_state_summary,
)


def _summary(seed: int, speedup: float) -> dict:
    return {
        "schema_version": "representax-dense-retrieval-comparison-v1",
        "contract": {"model": "tiny", "batch_size": 8, "seed": seed},
        "representax": {
            "steady_state_examples_per_second": 100.0 * speedup,
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


def test_pair_worker_flags_preserve_precision_and_telemetry():
    arguments = argparse.Namespace(mixed_precision=True, telemetry=True)

    assert _shared_worker_flags(arguments) == ["--mixed-precision", "--telemetry"]


def test_pair_worker_flags_enable_shared_export_and_offline_evaluation():
    arguments = argparse.Namespace(
        mixed_precision=True,
        telemetry=False,
        export=True,
    )

    assert _shared_worker_flags(arguments) == ["--mixed-precision", "--export"]


def test_pair_worker_flags_preserve_representax_sequence_buckets():
    arguments = argparse.Namespace(sequence_length_bucket=[16, 32, 128])

    assert _representax_worker_flags(arguments) == [
        "--sequence-length-bucket",
        "16",
        "--sequence-length-bucket",
        "32",
        "--sequence-length-bucket",
        "128",
    ]


def test_mpnet_mixed_precision_uses_a_metric_level_parity_tolerance():
    assert MODEL_SPECS["mpnet"].initial_metric_tolerance == 5e-3


def test_compile_summary_counts_every_executable_first_use(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "metrics": {
                "perf/compilation_and_first_step_seconds": duration,
            }
        }
        for duration in (3.0, 5.0, 7.0)
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _training_compile_summary(run) == (15.0, 3)


def test_steady_state_summary_uses_completed_warm_steps(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "event": "training_step",
            "iteration": 1,
            "metrics": {
                "perf/compilation_and_first_step_seconds": 36.8,
                "perf/examples": 2048,
            },
        },
        {
            "event": "training_step",
            "iteration": 2,
            "metrics": {
                "perf/step_seconds": 8.4,
                "perf/examples": 2048,
            },
        },
        {
            "event": "evaluation",
            "iteration": 2,
            "metrics": {
                "perf/step_seconds": 2.0,
                "perf/examples": 100,
            },
        },
        {
            "event": "training_step",
            "iteration": 3,
            "metrics": {
                "perf/step_seconds": 8.2,
                "perf/examples": 2048,
            },
        },
        {
            "event": "training_step",
            "iteration": 4,
            "metrics": {
                "perf/compilation_and_first_step_seconds": 8.3,
                "perf/step_seconds": 8.3,
                "perf/examples": 2048,
            },
        },
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _training_steady_state_summary(run) == pytest.approx((16.6, 2, 4096))


def test_reference_steady_state_summary_excludes_framework_warmup(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {
            "event": "training_step",
            "iteration": iteration,
            "metrics": {
                "perf/step_seconds": duration,
                "perf/examples": 32,
            },
        }
        for iteration, duration in ((1, 0.6), (2, 0.2), (3, 0.21))
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _reference_steady_state_summary(run) == pytest.approx((0.41, 2, 64))


def test_reference_steady_state_summary_excludes_delayed_compilation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    durations = (1.5, 0.7, 0.6, 0.5, 0.2, 0.21, 0.19, 0.2, 0.2)
    rows = [
        {
            "event": "training_step",
            "iteration": iteration,
            "metrics": {
                "perf/step_seconds": duration,
                "perf/examples": 32,
            },
        }
        for iteration, duration in enumerate(durations, start=1)
    ]
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _reference_steady_state_summary(run) == pytest.approx((0.8, 4, 128))


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


def _curve_point(updates: int, quality: float, seconds: float) -> dict:
    metric = "valid/NanoMSMARCO/cosine_ndcg@10"
    return {
        "updates": updates,
        "metrics": {metric: quality},
        "final_train_loss": None if updates == 0 else 1.0 / updates,
        "evaluation_seconds": 1.0,
        "training_elapsed_seconds": seconds,
    }


def test_time_to_quality_sorts_checkpoints_and_reports_first_crossing():
    result = _time_to_quality_summary(
        {
            "schema_version": "representax-dense-retrieval-comparison-v1",
            "contract": {"model": "tiny", "steps": 100},
            "representax": {
                "evaluation_history": [
                    _curve_point(0, 0.2, 0.0),
                    _curve_point(50, 0.35, 25.0),
                    _curve_point(100, 0.42, 50.0),
                ]
            },
            "sentence_transformers": {
                "evaluation_history": [
                    _curve_point(0, 0.2, 0.0),
                    _curve_point(50, 0.41, 37.5),
                    _curve_point(100, 0.39, 75.0),
                ]
            },
        },
        quality_target=0.4,
    )

    assert [point["updates"] for point in result["points"]] == [0, 50, 100]
    assert result["first_observed_crossing"]["representax"] == {
        "updates": 100,
        "training_seconds": 50.0,
        "ndcg@10": 0.42,
    }
    assert result["first_observed_crossing"]["sentence_transformers"] == {
        "updates": 50,
        "training_seconds": 37.5,
        "ndcg@10": 0.41,
    }


def test_tracked_modernvbert_dense_acceptance_meets_its_declared_gates():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "benchmarks/results/dense-retrieval-modernvbert-acceptance-20260818"
        / "summary.json"
    )
    evidence = json.loads(path.read_text())
    update = evidence["one_step_update_parity"]
    curve = evidence["continuous_time_to_quality"]
    capacity = evidence["sequence_512_capacity"]

    assert update["tensor_count"] == 134
    assert update["loss_absolute_difference"] <= 2e-6
    assert update["global_gradient_relative_difference"] <= 0.003
    assert update["global_gradient_cosine"] >= 0.99999
    assert update["global_update_relative_difference"] <= 0.015
    assert update["global_update_cosine"] >= 0.9999
    assert curve["sustained_training_speedup"] >= 1.2
    assert curve["amortized_training_speedup"] >= 1.0
    assert curve["first_quality_crossing"]["representax"]["updates"] == 25
    assert capacity["direct"]["largest_passing_batch"] == 32
    assert capacity["direct"]["first_failing_batch"] == 64
    assert capacity["grad_cache_global_batch_128"]["largest_passing_chunk"] == 64
    assert capacity["grad_cache_global_batch_128"]["first_failing_chunk"] == 128
