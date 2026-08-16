"""Shared numerical and systems acceptance machinery for model integrations."""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import warnings
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class NumericalTolerance:
    """Explicit tolerances for one named upstream comparison."""

    absolute: float
    relative: float
    cosine: float


@dataclass(frozen=True)
class NumericalResult:
    """Auditable error measurements for two equivalent output tensors."""

    max_absolute: float
    mean_absolute: float
    relative_l2: float
    cosine: float


@dataclass(frozen=True)
class ModelPerformanceCase:
    """One controlled native-versus-upstream model workload."""

    name: str
    package: str
    probe_module: str
    checkpoint_environment: str
    upstream_python_environment: str
    batch_size: int
    sequence_length: int
    image_count: int = 0
    warmup_iterations: int = 5
    measurement_iterations: int = 20
    minimum_speedup: float = 1.0
    maximum_memory_ratio: float | None = 1.0
    maximum_compilation_seconds: float | None = None
    probe_timeout_seconds: float = 300.0
    output_tolerance: NumericalTolerance = NumericalTolerance(
        absolute=5e-6,
        relative=5e-5,
        cosine=0.999999,
    )


MODEL_IMPLEMENTATIONS = (
    ModelPerformanceCase(
        name="bert-base-forward-fp32",
        package="bert",
        probe_module="tests.models.bert.performance_probe",
        checkpoint_environment="REPRESENTAX_BERT_CHECKPOINT",
        upstream_python_environment="REPRESENTAX_BERT_TRANSFORMERS_PYTHON",
        batch_size=16,
        sequence_length=128,
        maximum_memory_ratio=None,
    ),
    ModelPerformanceCase(
        name="all-minilm-l6-v2-dense-forward-fp32",
        package="bert",
        probe_module="tests.models.sentence_transformers.performance_probe",
        checkpoint_environment="REPRESENTAX_MINILM_CHECKPOINT",
        upstream_python_environment="REPRESENTAX_SENTENCE_TRANSFORMERS_PYTHON",
        batch_size=16,
        sequence_length=128,
        maximum_memory_ratio=None,
        maximum_compilation_seconds=60.0,
        probe_timeout_seconds=120.0,
        output_tolerance=NumericalTolerance(
            absolute=2e-6,
            relative=2e-6,
            cosine=0.999999,
        ),
    ),
    ModelPerformanceCase(
        name="modernvbert-text-forward-fp32",
        package="modernvbert",
        probe_module="tests.models.modernvbert.performance_probe",
        checkpoint_environment="REPRESENTAX_MODERNVBERT_CHECKPOINT",
        upstream_python_environment="REPRESENTAX_MODERNVBERT_TRANSFORMERS_PYTHON",
        batch_size=16,
        sequence_length=128,
    ),
    ModelPerformanceCase(
        name="modernvbert-multimodal-forward-fp32",
        package="modernvbert",
        probe_module="tests.models.modernvbert.performance_probe",
        checkpoint_environment="REPRESENTAX_MODERNVBERT_CHECKPOINT",
        upstream_python_environment="REPRESENTAX_MODERNVBERT_TRANSFORMERS_PYTHON",
        batch_size=1,
        sequence_length=89,
        image_count=1,
        maximum_memory_ratio=None,
        output_tolerance=NumericalTolerance(
            absolute=5e-6,
            relative=1e-5,
            cosine=0.9999998,
        ),
    ),
)


def numerical_result(actual: np.ndarray, expected: np.ndarray) -> NumericalResult:
    """Measure absolute, relative, and directional numerical agreement."""

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual - expected
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    denominator = max(float(np.linalg.norm(expected_flat)), 1e-12)
    cosine_denominator = max(
        float(np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)),
        1e-12,
    )
    return NumericalResult(
        max_absolute=float(np.max(np.abs(delta))),
        mean_absolute=float(np.mean(np.abs(delta))),
        relative_l2=float(np.linalg.norm(delta) / denominator),
        cosine=float(np.dot(actual_flat, expected_flat) / cosine_denominator),
    )


def assert_numerically_equivalent(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: NumericalTolerance,
) -> NumericalResult:
    """Assert a model output satisfies its complete numerical contract."""

    result = numerical_result(actual, expected)
    assert result.max_absolute <= tolerance.absolute, result
    assert result.relative_l2 <= tolerance.relative, result
    assert result.cosine >= tolerance.cosine, result
    return result


class _ProcessMemoryMonitor:
    """Best-effort per-process GPU memory sampling through NVML."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception:
            self._pynvml = None

    def _sample(self) -> None:
        assert self._pynvml is not None
        pynvml = self._pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
            while not self._stop.is_set() and self.process.poll() is None:
                for index in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for process in processes:
                        if process.pid == self.process.pid:
                            used = int(process.usedGpuMemory)
                            self.peak_bytes = max(self.peak_bytes or 0, used)
                self._stop.wait(0.02)
        except Exception:
            pass

    def close(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pynvml is not None:
            with suppress(Exception):
                self._pynvml.nvmlShutdown()
        return self.peak_bytes


def _run_probe(
    case: ModelPerformanceCase,
    runtime: str,
    checkpoint: Path,
    inputs: Path,
    directory: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    report_path = directory / f"{runtime}.json"
    output_path = directory / f"{runtime}.npy"
    executable = (
        os.environ.get(case.upstream_python_environment, sys.executable)
        if runtime == "transformers"
        else sys.executable
    )
    command = [
        executable,
        "-m",
        case.probe_module,
        "--runtime",
        runtime,
        "--checkpoint",
        str(checkpoint),
        "--inputs",
        str(inputs),
        "--report",
        str(report_path),
        "--output",
        str(output_path),
        "--warmups",
        str(case.warmup_iterations),
        "--iterations",
        str(case.measurement_iterations),
    ]
    environment = os.environ.copy()
    source = str(Path.cwd() / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source, environment.get("PYTHONPATH")) if item
    )
    environment.update(
        {
            "JAX_ENABLE_COMPILATION_CACHE": "false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    provenance = f"{revision}-dirty" if revision and dirty else revision
    environment.setdefault("REPRESENTAX_GIT_REVISION", provenance or "unknown")
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitor = _ProcessMemoryMonitor(process)
    try:
        stdout, stderr = process.communicate(timeout=case.probe_timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        stdout, stderr = process.communicate()
        raise AssertionError(
            f"{runtime} probe exceeded {case.probe_timeout_seconds:.1f}s; "
            "the compile or execution workload is not acceptably bounded\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from error
    external_peak = monitor.close()
    if process.returncode:
        raise AssertionError(
            f"{runtime} probe failed with exit {process.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    report = json.loads(report_path.read_text())
    report["external_peak_device_bytes"] = external_peak
    return report, np.load(output_path)


def _make_inputs(case: ModelPerformanceCase, checkpoint: Path, path: Path) -> None:
    config = json.loads((checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    vocabulary_size = int(text_config["vocab_size"])
    generator = np.random.default_rng(17)
    input_ids = generator.integers(
        0,
        vocabulary_size,
        size=(case.batch_size, case.sequence_length),
        dtype=np.int32,
    )
    attention_mask = np.ones_like(input_ids, dtype=np.int32)
    arrays: dict[str, np.ndarray] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if case.image_count:
        image_token_id = int(config["image_token_id"])
        image_tokens = (
            int(config["vision_config"]["image_size"])
            // int(config["vision_config"]["patch_size"])
        ) ** 2 // int(config["pixel_shuffle_factor"]) ** 2
        required = case.image_count * image_tokens
        if required > case.sequence_length:
            raise ValueError("performance sequence cannot hold all image tokens")
        input_ids[:, :required] = image_token_id
        image_size = int(config["vision_config"]["image_size"])
        arrays["pixel_values"] = generator.uniform(
            -1.0,
            1.0,
            size=(
                case.batch_size,
                case.image_count,
                int(config["vision_config"]["num_channels"]),
                image_size,
                image_size,
            ),
        ).astype(np.float32)
        arrays["pixel_attention_mask"] = np.ones(
            (case.batch_size, case.image_count, image_size, image_size),
            dtype=np.int32,
        )
    # NumPy's current ``savez`` typing models arbitrary named arrays as booleans.
    np.savez(path, **cast(Any, arrays))


def _memory_values(reports: dict[str, dict[str, Any]]) -> tuple[int, int, str]:
    return (
        int(reports["transformers"]["allocator_peak_device_bytes"]),
        int(reports["representax"]["allocator_peak_device_bytes"]),
        "framework_allocator_peak",
    )


def compare_model_performance(
    case: ModelPerformanceCase,
    checkpoint: Path,
    directory: Path,
) -> dict[str, Any]:
    """Run isolated probes and enforce numerical, speed, and memory gates."""

    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "inputs.npz"
    _make_inputs(case, checkpoint, inputs)
    reports: dict[str, dict[str, Any]] = {}
    outputs: dict[str, np.ndarray] = {}
    for runtime in ("transformers", "representax"):
        reports[runtime], outputs[runtime] = _run_probe(
            case, runtime, checkpoint, inputs, directory
        )
    upstream_precision = reports["transformers"].get("precision_policy", {})
    if upstream_precision != {
        "float32_matmul": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }:
        raise AssertionError(
            f"upstream probe did not enforce full FP32: {upstream_precision}"
        )
    if (
        reports["transformers"]["workload_fingerprint"]
        != reports["representax"]["workload_fingerprint"]
    ):
        raise AssertionError("runtime probes executed different workloads")

    numerical = assert_numerically_equivalent(
        outputs["representax"],
        outputs["transformers"],
        case.output_tolerance,
    )
    reference_seconds = statistics.median(
        reports["transformers"]["steady_state_seconds"]
    )
    native_seconds = statistics.median(reports["representax"]["steady_state_seconds"])
    reference_memory, native_memory, memory_measurement = _memory_values(reports)
    speedup = reference_seconds / native_seconds
    memory_ratio = native_memory / reference_memory
    reference_external = reports["transformers"].get("external_peak_device_bytes")
    native_external = reports["representax"].get("external_peak_device_bytes")
    external_memory_ratio = (
        float(native_external) / float(reference_external)
        if reference_external and native_external
        else None
    )
    memory_warnings = []
    if memory_ratio > 1.0:
        memory_warnings.append(
            "native framework allocator peak is "
            f"{(memory_ratio - 1.0) * 100:.2f}% above upstream"
        )
    if external_memory_ratio is not None and external_memory_ratio > 1.0:
        memory_warnings.append(
            "native end-to-end process peak is "
            f"{(external_memory_ratio - 1.0) * 100:.2f}% above upstream"
        )
    cold_start_delta = (
        reports["representax"]["initialization_seconds"]
        + reports["representax"]["compile_or_first_execution_seconds"]
        - reports["transformers"]["initialization_seconds"]
        - reports["transformers"]["compile_or_first_execution_seconds"]
    )
    steady_state_savings = reference_seconds - native_seconds
    break_even_steps = (
        max(cold_start_delta, 0.0) / steady_state_savings
        if steady_state_savings > 0
        else None
    )
    result = {
        "case": asdict(case),
        "numerical": asdict(numerical),
        "transformers": reports["transformers"],
        "representax": reports["representax"],
        "speedup": speedup,
        "memory_ratio": memory_ratio,
        "memory_measurement": memory_measurement,
        "external_memory_ratio": external_memory_ratio,
        "memory_warnings": memory_warnings,
        "cold_start_delta_seconds": cold_start_delta,
        "steady_state_break_even_steps": break_even_steps,
        "transformers_median_seconds": reference_seconds,
        "representax_median_seconds": native_seconds,
        "transformers_peak_device_bytes": reference_memory,
        "representax_peak_device_bytes": native_memory,
    }
    result_path = directory / "comparison.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    artifact_root = os.environ.get("REPRESENTAX_PERFORMANCE_ARTIFACTS")
    if artifact_root:
        run_name = f"{time.strftime('%Y%m%d-%H%M%S')}--{os.getpid()}"
        destination = Path(artifact_root) / case.name / run_name
        shutil.copytree(directory, destination)

    for message in memory_warnings:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    assert speedup > case.minimum_speedup, result
    if case.maximum_compilation_seconds is not None:
        compilation_seconds = float(
            reports["representax"].get(
                "compilation_seconds",
                reports["representax"]["compile_or_first_execution_seconds"],
            )
        )
        assert compilation_seconds <= case.maximum_compilation_seconds, result
    if case.maximum_memory_ratio is not None:
        assert memory_ratio <= case.maximum_memory_ratio, result
    return result
