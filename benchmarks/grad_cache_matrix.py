"""Run and validate the matched ModernVBERT GradCache performance matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ORACLE_VERSION = "5.6.1"
TRANSFORMERS_VERSION = "5.3.0"
PROBE_SCHEMA = "representax-grad-cache-modernvbert-probe-v1"
SUMMARY_SCHEMA = "representax-grad-cache-modernvbert-matrix-v1"
DEFAULT_BATCHES = (32, 128, 512, 1024)
_GIB = 1024**3


@dataclass(frozen=True)
class MatrixContract:
    """Scientific and acceptance contract shared by every matrix point."""

    batches: tuple[int, ...] = DEFAULT_BATCHES
    sequence_length: int = 512
    chunk_size: int = 2
    sentence_transformers_version: str = ORACLE_VERSION
    maximum_loss_absolute: float = 5e-6
    maximum_gradient_norm_relative: float = 1e-3
    minimum_speedup: float = 1.0

    def __post_init__(self) -> None:
        if not self.batches or any(batch <= 0 for batch in self.batches):
            raise ValueError("matrix batches must be positive")
        if len(set(self.batches)) != len(self.batches):
            raise ValueError("matrix batches must be unique")
        if self.sequence_length <= 0 or self.chunk_size <= 0:
            raise ValueError("sequence length and chunk size must be positive")


DEFAULT_CONTRACT = MatrixContract()


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if not isinstance(report, dict):
        raise TypeError(f"benchmark report must be an object: {path}")
    return report


def _require(report: dict[str, Any], name: str, expected: Any) -> None:
    actual = report.get(name)
    if actual != expected:
        raise AssertionError(f"{name} differs: {actual!r} != {expected!r}")


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-12)


def compare_reports(
    representax: dict[str, Any],
    sentence_transformers: dict[str, Any],
    *,
    contract: MatrixContract,
) -> dict[str, Any]:
    """Validate one matched pair and calculate its systems deltas."""

    for report in (representax, sentence_transformers):
        _require(report, "schema_version", PROBE_SCHEMA)
        _require(report, "status", "completed")
        _require(report, "oom", False)
        _require(report, "sequence_length", contract.sequence_length)
        _require(report, "chunk_size", contract.chunk_size)
    _require(representax, "runtime", "grad-cache")
    _require(sentence_transformers, "runtime", "sentence-transformers")
    _require(
        sentence_transformers,
        "framework_version",
        contract.sentence_transformers_version,
    )
    _require(sentence_transformers, "transformers_version", TRANSFORMERS_VERSION)
    for name in ("batch_size", "checkpoint_revision", "seed", "workload_fingerprints"):
        if representax.get(name) != sentence_transformers.get(name):
            raise AssertionError(
                f"matched benchmark field {name!r} differs: "
                f"{representax.get(name)!r} != {sentence_transformers.get(name)!r}"
            )
    upstream_precision = sentence_transformers.get("precision_policy")
    expected_upstream_precision = {
        "parameters": "float32",
        "compute": "float32",
        "objective": "float32",
        "float32_matmul": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    if upstream_precision != expected_upstream_precision:
        raise AssertionError(
            f"Sentence Transformers precision differs: {upstream_precision!r}"
        )
    for name in ("parameters", "compute", "objective", "float32_matmul"):
        if (
            representax["precision_policy"].get(name)
            != expected_upstream_precision[name]
        ):
            raise AssertionError(f"Representax precision field {name!r} differs")

    native_losses = tuple(map(float, representax["losses"]))
    upstream_losses = tuple(map(float, sentence_transformers["losses"]))
    if len(native_losses) != len(upstream_losses) or not native_losses:
        raise AssertionError("matched benchmark loss series do not align")
    loss_absolute = max(
        abs(native - upstream)
        for native, upstream in zip(native_losses, upstream_losses, strict=True)
    )
    gradient_relative = _relative_difference(
        float(representax["gradient_global_norm"]),
        float(sentence_transformers["gradient_global_norm"]),
    )
    if loss_absolute > contract.maximum_loss_absolute:
        raise AssertionError(f"loss absolute difference is {loss_absolute:.6g}")
    if gradient_relative > contract.maximum_gradient_norm_relative:
        raise AssertionError(
            f"gradient norm relative difference is {gradient_relative:.6g}"
        )

    native_seconds = float(representax["steady_state_median_seconds"])
    upstream_seconds = float(sentence_transformers["steady_state_median_seconds"])
    if not all(
        math.isfinite(value) and value > 0
        for value in (native_seconds, upstream_seconds)
    ):
        raise AssertionError("steady-state measurements must be finite and positive")
    speedup = upstream_seconds / native_seconds
    native_compile = float(representax["compile_plus_first_seconds"])
    upstream_first = float(sentence_transformers["compile_plus_first_seconds"])
    savings = upstream_seconds - native_seconds
    break_even = (
        max(native_compile - upstream_first, 0.0) / savings if savings > 0 else None
    )
    return {
        "batch_size": int(representax["batch_size"]),
        "loss_absolute_difference": loss_absolute,
        "gradient_norm_relative_difference": gradient_relative,
        "representax_step_seconds": native_seconds,
        "sentence_transformers_step_seconds": upstream_seconds,
        "representax_examples_per_second": float(representax["examples_per_second"]),
        "sentence_transformers_examples_per_second": float(
            sentence_transformers["examples_per_second"]
        ),
        "speedup": speedup,
        "performance_status": (
            "pass" if speedup > contract.minimum_speedup else "warning"
        ),
        "representax_compile_plus_first_seconds": native_compile,
        "sentence_transformers_first_step_seconds": upstream_first,
        "uncached_break_even_additional_steps": break_even,
        "representax_allocator_gib": (
            int(representax["allocator_peak_device_bytes"]) / _GIB
        ),
        "sentence_transformers_allocator_gib": (
            int(sentence_transformers["allocator_peak_device_bytes"]) / _GIB
        ),
        "allocator_ratio": (
            int(representax["allocator_peak_device_bytes"])
            / int(sentence_transformers["allocator_peak_device_bytes"])
        ),
        "representax_process_gib": int(representax["process_peak_device_bytes"]) / _GIB,
        "sentence_transformers_process_gib": int(
            sentence_transformers["process_peak_device_bytes"]
        )
        / _GIB,
        "process_ratio": (
            int(representax["process_peak_device_bytes"])
            / int(sentence_transformers["process_peak_device_bytes"])
        ),
    }


def summarize_directory(
    directory: Path,
    *,
    contract: MatrixContract = DEFAULT_CONTRACT,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Validate every raw report in one directory and return one summary."""

    points = []
    for batch in contract.batches:
        native = _load_report(directory / f"representax-b{batch}-s512-c2.json")
        upstream = _load_report(
            directory / f"sentence-transformers-b{batch}-s512-c2.json"
        )
        points.append(compare_reports(native, upstream, contract=contract))
    observed = tuple(point["batch_size"] for point in points)
    if observed != contract.batches:
        raise AssertionError(f"matrix batches differ: {observed} != {contract.batches}")
    speedups = tuple(float(point["speedup"]) for point in points)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "source_commit": source_commit,
        "contract": {
            "batches": list(contract.batches),
            "sequence_length": contract.sequence_length,
            "chunk_size": contract.chunk_size,
            "sentence_transformers_version": (contract.sentence_transformers_version),
            "maximum_loss_absolute": contract.maximum_loss_absolute,
            "maximum_gradient_norm_relative": (contract.maximum_gradient_norm_relative),
            "minimum_speedup": contract.minimum_speedup,
        },
        "points": points,
        "minimum_speedup": min(speedups),
        "maximum_speedup": max(speedups),
        "geometric_mean_speedup": statistics.geometric_mean(speedups),
        "all_performance_points_pass": all(
            point["performance_status"] == "pass" for point in points
        ),
    }
    if not summary["all_performance_points_pass"]:
        warnings.warn(
            "Representax did not beat Sentence Transformers at every matrix point",
            RuntimeWarning,
            stacklevel=2,
        )
    return summary


def _run_probe(
    *,
    python: Path,
    runtime: str,
    gpu: int,
    checkpoint: Path,
    output: Path,
    batch: int,
    sequence_length: int,
    chunk_size: int,
    warmup_steps: int,
    measured_steps: int,
) -> None:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    if runtime == "grad-cache":
        environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    command = (
        str(python),
        str(Path(__file__).with_name("grad_cache_modernvbert.py")),
        "--runtime",
        runtime,
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--batch-size",
        str(batch),
        "--sequence-length",
        str(sequence_length),
        "--chunk-size",
        str(chunk_size),
        "--warmup-steps",
        str(warmup_steps),
        "--measured-steps",
        str(measured_steps),
    )
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    output.with_suffix(".log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{runtime} batch {batch} failed; see {output.with_suffix('.log')}"
        )


def run_matrix(
    *,
    checkpoint: Path,
    output_directory: Path,
    native_python: Path,
    upstream_python: Path,
    native_gpu: int,
    upstream_gpu: int,
    contract: MatrixContract = DEFAULT_CONTRACT,
    warmup_steps: int = 1,
    measured_steps: int = 5,
) -> dict[str, Any]:
    """Run each matched pair concurrently on two isolated GPUs."""

    if native_gpu == upstream_gpu:
        raise ValueError("native and upstream probes require distinct GPUs")
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    if not native_python.is_file() or not upstream_python.is_file():
        raise FileNotFoundError("native and upstream Python executables must exist")
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup steps must be non-negative and measurements positive")
    output_directory.mkdir(parents=True, exist_ok=True)
    for batch in contract.batches:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(
                    _run_probe,
                    python=native_python,
                    runtime="grad-cache",
                    gpu=native_gpu,
                    checkpoint=checkpoint,
                    output=output_directory / f"representax-b{batch}-s512-c2.json",
                    batch=batch,
                    sequence_length=contract.sequence_length,
                    chunk_size=contract.chunk_size,
                    warmup_steps=warmup_steps,
                    measured_steps=measured_steps,
                ),
                executor.submit(
                    _run_probe,
                    python=upstream_python,
                    runtime="sentence-transformers",
                    gpu=upstream_gpu,
                    checkpoint=checkpoint,
                    output=output_directory
                    / f"sentence-transformers-b{batch}-s512-c2.json",
                    batch=batch,
                    sequence_length=contract.sequence_length,
                    chunk_size=contract.chunk_size,
                    warmup_steps=warmup_steps,
                    measured_steps=measured_steps,
                ),
            )
            for future in futures:
                future.result()
    return summarize_directory(output_directory, contract=contract)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--upstream-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-gpu", type=int, default=0)
    parser.add_argument("--upstream-gpu", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    contract = MatrixContract(batches=tuple(arguments.batches))
    if arguments.summarize_only:
        summary = summarize_directory(
            arguments.output_directory,
            contract=contract,
            source_commit=arguments.source_commit,
        )
    else:
        if arguments.checkpoint is None:
            raise ValueError("--checkpoint is required unless --summarize-only")
        summary = run_matrix(
            checkpoint=arguments.checkpoint,
            output_directory=arguments.output_directory,
            native_python=arguments.native_python,
            upstream_python=arguments.upstream_python,
            native_gpu=arguments.native_gpu,
            upstream_gpu=arguments.upstream_gpu,
            contract=contract,
            warmup_steps=arguments.warmup_steps,
            measured_steps=arguments.measured_steps,
        )
        summary["source_commit"] = arguments.source_commit
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
