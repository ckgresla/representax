"""Matched GradCache performance contracts and reproducible GPU acceptance."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from benchmarks.grad_cache_matrix import (
    DEFAULT_BATCHES,
    MatrixContract,
    compare_reports,
    run_matrix,
    summarize_directory,
)


def _report(*, runtime: str, version: str, seconds: float) -> dict[str, object]:
    upstream = runtime == "sentence-transformers"
    precision = {
        "parameters": "float32",
        "compute": "float32",
        "objective": "float32",
        "float32_matmul": "highest",
    }
    if upstream:
        precision.update(
            {
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            }
        )
    return {
        "schema_version": "representax-grad-cache-modernvbert-probe-v1",
        "status": "completed",
        "oom": False,
        "runtime": runtime,
        "framework_version": version,
        "transformers_version": "5.3.0",
        "batch_size": 32,
        "sequence_length": 512,
        "chunk_size": 2,
        "checkpoint_revision": "revision",
        "seed": 7,
        "workload_fingerprints": {"input_sha256": "same", "scientific": "same"},
        "precision_policy": precision,
        "losses": [3.0, 3.0],
        "gradient_global_norm": 2.0,
        "steady_state_median_seconds": seconds,
        "examples_per_second": 32 / seconds,
        "compile_plus_first_seconds": 5.0 if upstream else 20.0,
        "allocator_peak_device_bytes": 4 * 1024**3,
        "process_peak_device_bytes": 5 * 1024**3,
    }


def test_grad_cache_matrix_pair_requires_one_matched_scientific_contract():
    contract = MatrixContract(batches=(32,))
    native = _report(runtime="grad-cache", version="unreleased", seconds=1.0)
    upstream = _report(
        runtime="sentence-transformers",
        version="5.6.1",
        seconds=1.25,
    )
    result = compare_reports(native, upstream, contract=contract)
    assert result["speedup"] == pytest.approx(1.25)
    assert result["performance_status"] == "pass"

    mismatched = dict(upstream)
    mismatched["seed"] = 8
    with pytest.raises(AssertionError, match="seed"):
        compare_reports(native, mismatched, contract=contract)


@pytest.mark.performance
def test_recorded_modernvbert_grad_cache_matrix_beats_pinned_oracle():
    root = Path(__file__).parents[2]
    directory = root / "benchmarks/results/grad-cache-st56-20260817"
    if not directory.is_dir():
        pytest.skip("tracked GradCache 5.6.1 performance evidence is not installed")
    summary = summarize_directory(directory)
    assert summary["all_performance_points_pass"] is True
    assert tuple(point["batch_size"] for point in summary["points"]) == DEFAULT_BATCHES


@pytest.mark.performance
def test_live_modernvbert_grad_cache_matrix_beats_pinned_oracle(tmp_path):
    checkpoint = os.environ.get("REPRESENTAX_MODERNVBERT_CHECKPOINT")
    gpu_pair = os.environ.get("REPRESENTAX_GRAD_CACHE_PERFORMANCE_GPUS")
    if checkpoint is None or gpu_pair is None:
        pytest.skip(
            "set REPRESENTAX_MODERNVBERT_CHECKPOINT and "
            "REPRESENTAX_GRAD_CACHE_PERFORMANCE_GPUS"
        )
    try:
        native_gpu, upstream_gpu = map(int, gpu_pair.split(","))
    except ValueError as error:
        raise ValueError(
            "REPRESENTAX_GRAD_CACHE_PERFORMANCE_GPUS must be 'native,upstream'"
        ) from error
    summary = run_matrix(
        checkpoint=Path(checkpoint),
        output_directory=tmp_path,
        native_python=Path(sys.executable),
        upstream_python=Path(sys.executable),
        native_gpu=native_gpu,
        upstream_gpu=upstream_gpu,
    )
    assert summary["all_performance_points_pass"] is True
