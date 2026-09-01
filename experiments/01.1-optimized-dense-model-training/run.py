"""Tune both dense-training implementations, then run the matched winner pair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = (
    Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
    / "01.1-optimized-dense-model-training"
)
DEFAULT_CHECKPOINT = Path("/raid/representax/oracles/all-mpnet-base-v2")
DEFAULT_DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
SEQUENCE_LENGTH_BUCKETS = (16, 32, 64, 128, 256)
QUALITY_SEEDS = (7, 42, 773)
FINAL_STEPS = 256


@dataclass(frozen=True, slots=True)
class Candidate:
    framework: str
    cache_chunk_size: int
    data_threads: int
    prefetch_buffer_size: int
    persistent_workers: bool = False
    torch_compile: bool = False

    @property
    def name(self) -> str:
        prefix = "rx" if self.framework == "representax" else "st"
        values = [
            prefix,
            f"chunk-{self.cache_chunk_size}",
            f"workers-{self.data_threads}",
        ]
        if self.persistent_workers:
            values.append("persistent")
        if self.torch_compile:
            values.append("compile")
        return "-".join(values)


def _benchmark_command(command: str) -> list[str]:
    return [sys.executable, "-m", "benchmarks.dense_retrieval", command]


def _source_commits() -> dict[str, str]:
    representax = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    references = json.loads(
        (REPOSITORY_ROOT / "benchmarks/configs/paper-references-v1.json").read_text(
            encoding="utf-8"
        )
    )["references"]
    return {
        "representax": representax,
        "sentence-transformers": references["sentence-transformers"]["commit"],
    }


def _scientific_arguments(
    *,
    checkpoint: Path,
    data: Path,
    seed: int,
    steps: int,
) -> list[str]:
    return [
        "--model",
        "mpnet",
        "--checkpoint",
        str(checkpoint),
        "--data-directory",
        str(data),
        "--batch-size",
        "2048",
        "--steps",
        str(steps),
        "--maximum-length",
        "256",
        "--evaluation-batch-size",
        "128",
        "--seed",
        str(seed),
        "--world-size",
        "1",
        "--mixed-precision",
        "--telemetry",
    ]


def _candidate_command(
    candidate: Candidate,
    *,
    checkpoint: Path,
    data: Path,
    seed: int,
    steps: int,
    directory: Path,
) -> list[str]:
    command = [
        *_benchmark_command("worker"),
        "--framework",
        candidate.framework,
        *_scientific_arguments(
            checkpoint=checkpoint,
            data=data,
            seed=seed,
            steps=steps,
        ),
        "--cache-chunk-size",
        str(candidate.cache_chunk_size),
        "--run-directory",
        str(directory / "run"),
        "--report",
        str(directory / "report.json"),
    ]
    if candidate.framework == "representax":
        command.extend(
            (
                "--data-threads",
                str(candidate.data_threads),
                "--prefetch-buffer-size",
                str(candidate.prefetch_buffer_size),
            )
        )
        for bucket in SEQUENCE_LENGTH_BUCKETS:
            command.extend(("--sequence-length-bucket", str(bucket)))
    else:
        command.extend(
            (
                "--sentence-transformers-data-threads",
                str(candidate.data_threads),
                "--sentence-transformers-prefetch-buffer-size",
                str(candidate.prefetch_buffer_size),
            )
        )
        if candidate.persistent_workers:
            command.append("--sentence-transformers-persistent-workers")
        if candidate.torch_compile:
            command.append("--sentence-transformers-torch-compile")
    return command


def _environment(candidate: Candidate, gpu: int, directory: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("LD_LIBRARY_PATH", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "JAX_COMPILATION_CACHE_DIR": str(directory / "jax-cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(directory / "torchinductor-cache"),
        }
    )
    if candidate.framework == "representax":
        environment.update(
            {
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            }
        )
        environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
    return environment


def _projected_rate(report: dict[str, Any], target_steps: int = FINAL_STEPS) -> float:
    steady_seconds = float(report["steady_state_seconds"])
    steady_steps = int(report["steady_state_step_count"])
    seconds_per_step = steady_seconds / steady_steps
    observed_seconds = float(report["training_compute_seconds"])
    fixed_overhead = max(
        observed_seconds - int(report["steps"]) * seconds_per_step,
        0.0,
    )
    projected_seconds = fixed_overhead + target_steps * seconds_per_step
    return int(report["batch_size"]) * target_steps / projected_seconds


def _run_candidate(
    candidate: Candidate,
    *,
    gpu: int,
    root: Path,
    checkpoint: Path,
    data: Path,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    directory = root / "candidates" / candidate.name
    directory.mkdir(parents=True, exist_ok=False)
    command = _candidate_command(
        candidate,
        checkpoint=checkpoint,
        data=data,
        seed=seed,
        steps=steps,
        directory=directory,
    )
    started = time.perf_counter()
    with (directory / "worker.log").open("x", encoding="utf-8") as stream:
        process = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=_environment(candidate, gpu, directory),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result: dict[str, Any] = {
        "candidate": asdict(candidate),
        "name": candidate.name,
        "gpu": gpu,
        "command": command,
        "process_seconds": time.perf_counter() - started,
        "return_code": process.returncode,
        "log": str((directory / "worker.log").resolve()),
    }
    report_path = directory / "report.json"
    if process.returncode == 0 and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result.update(
            {
                "status": "accepted",
                "report": str(report_path.resolve()),
                "steady_state_examples_per_second": report[
                    "steady_state_examples_per_second"
                ],
                "amortized_examples_per_second": report[
                    "amortized_examples_per_second"
                ],
                "projected_256_update_examples_per_second": _projected_rate(report),
                "training_compute_seconds": report["training_compute_seconds"],
                "maximum_device_bytes": report.get(
                    "jax_allocator_peak_bytes_in_use",
                    report.get("torch_peak_allocated_bytes"),
                ),
            }
        )
    else:
        result["status"] = "rejected"
    print(
        f"{candidate.name}: {result['status']}"
        + (
            f" {result['projected_256_update_examples_per_second']:.2f} projected ex/s"
            if result["status"] == "accepted"
            else f" (see {result['log']})"
        ),
        flush=True,
    )
    return result


def _best(results: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [result for result in results if result["status"] == "accepted"]
    if not accepted:
        raise RuntimeError("every tuning candidate failed")
    return max(
        accepted,
        key=lambda result: result["projected_256_update_examples_per_second"],
    )


def _run_candidates(
    candidates: list[Candidate],
    *,
    gpu: int,
    root: Path,
    checkpoint: Path,
    data: Path,
    seed: int,
    steps: int,
) -> list[dict[str, Any]]:
    return [
        _run_candidate(
            candidate,
            gpu=gpu,
            root=root,
            checkpoint=checkpoint,
            data=data,
            seed=seed,
            steps=steps,
        )
        for candidate in candidates
    ]


def _tune(arguments: argparse.Namespace) -> None:
    root = arguments.artifact_root.expanduser().resolve() / "tuning"
    root.mkdir(parents=True, exist_ok=False)
    chunk_sizes = (32, 64, 128, 256)
    initial = {
        "representax": [Candidate("representax", chunk, 4, 8) for chunk in chunk_sizes],
        "sentence-transformers": [
            Candidate("sentence-transformers", chunk, 0, 8) for chunk in chunk_sizes
        ],
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            framework: executor.submit(
                _run_candidates,
                candidates,
                gpu=gpu,
                root=root,
                checkpoint=arguments.checkpoint,
                data=arguments.data,
                seed=arguments.seed,
                steps=arguments.steps,
            )
            for (framework, candidates), gpu in zip(
                initial.items(), arguments.gpus, strict=True
            )
        }
        first = {framework: future.result() for framework, future in futures.items()}

    native_chunk = _best(first["representax"])["candidate"]["cache_chunk_size"]
    reference_chunk = _best(first["sentence-transformers"])["candidate"][
        "cache_chunk_size"
    ]
    refinements = {
        "representax": [Candidate("representax", native_chunk, 8, 16)],
        "sentence-transformers": [
            Candidate("sentence-transformers", reference_chunk, 4, 8, True),
            Candidate("sentence-transformers", reference_chunk, 8, 8, True),
            Candidate(
                "sentence-transformers",
                reference_chunk,
                0,
                8,
                torch_compile=True,
            ),
            Candidate(
                "sentence-transformers",
                reference_chunk,
                4,
                8,
                persistent_workers=True,
                torch_compile=True,
            ),
        ],
    }
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            framework: executor.submit(
                _run_candidates,
                candidates,
                gpu=gpu,
                root=root,
                checkpoint=arguments.checkpoint,
                data=arguments.data,
                seed=arguments.seed,
                steps=arguments.steps,
            )
            for (framework, candidates), gpu in zip(
                refinements.items(), arguments.gpus, strict=True
            )
        }
        second = {framework: future.result() for framework, future in futures.items()}

    results = {framework: first[framework] + second[framework] for framework in first}
    winners = {framework: _best(values) for framework, values in results.items()}
    document = {
        "schema_version": "representax-optimized-dense-tuning-v1",
        "source_commits": _source_commits(),
        "selection_metric": "projected_256_update_examples_per_second",
        "scientific_contract": {
            "model": "sentence-transformers/all-mpnet-base-v2",
            "dataset": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
            "batch_size": 2048,
            "maximum_length": 256,
            "loss": "exact asymmetric cosine MNR scale=20",
            "precision": "bfloat16-compute-float32-master",
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "seed": arguments.seed,
            "tuning_steps": arguments.steps,
            "target_steps": FINAL_STEPS,
        },
        "candidates": results,
        "winners": {
            framework: winner["candidate"] for framework, winner in winners.items()
        },
    }
    output = root / "summary.json"
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps(document["winners"], indent=2, sort_keys=True))


def _confirm_compile(arguments: argparse.Namespace) -> None:
    tuning_root = arguments.artifact_root.expanduser().resolve() / "tuning"
    summary_path = tuning_root / "summary.json"
    document = json.loads(summary_path.read_text(encoding="utf-8"))
    root = tuning_root / "compile-confirmation"
    root.mkdir(parents=True, exist_ok=False)
    candidates = (
        Candidate(
            "sentence-transformers",
            128,
            0,
            8,
            torch_compile=True,
        ),
        Candidate(
            "sentence-transformers",
            128,
            4,
            8,
            persistent_workers=True,
            torch_compile=True,
        ),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _run_candidate,
                candidate,
                gpu=gpu,
                root=root,
                checkpoint=arguments.checkpoint,
                data=arguments.data,
                seed=arguments.seed,
                steps=arguments.steps,
            )
            for candidate, gpu in zip(candidates, arguments.gpus, strict=True)
        ]
        confirmations = [future.result() for future in futures]
    eager = [
        result
        for result in document["candidates"]["sentence-transformers"]
        if not result["candidate"]["torch_compile"]
    ]
    winner = _best(eager + confirmations)
    document["compile_confirmation"] = {
        "source_commits": _source_commits(),
        "isolated_torchinductor_caches": True,
        "candidates": confirmations,
    }
    document["winners"]["sentence-transformers"] = winner["candidate"]
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, summary_path)
    print(json.dumps(document["winners"], indent=2, sort_keys=True))


def _final_command(arguments: argparse.Namespace) -> list[str]:
    tuning = json.loads(
        (
            arguments.artifact_root.expanduser().resolve() / "tuning" / "summary.json"
        ).read_text(encoding="utf-8")
    )
    native = Candidate(**tuning["winners"]["representax"])
    reference = Candidate(**tuning["winners"]["sentence-transformers"])
    command = [
        *_benchmark_command("pair"),
        *_scientific_arguments(
            checkpoint=arguments.checkpoint,
            data=arguments.data,
            seed=arguments.seed,
            steps=FINAL_STEPS,
        ),
        "--evaluation-every-steps",
        "64",
        "--representax-cache-chunk-size",
        str(native.cache_chunk_size),
        "--sentence-transformers-cache-chunk-size",
        str(reference.cache_chunk_size),
        "--data-threads",
        str(native.data_threads),
        "--prefetch-buffer-size",
        str(native.prefetch_buffer_size),
        "--sentence-transformers-data-threads",
        str(reference.data_threads),
        "--sentence-transformers-prefetch-buffer-size",
        str(reference.prefetch_buffer_size),
        "--result-directory",
        str(arguments.artifact_root / "runs" / f"seed-{arguments.seed}"),
        "--representax-gpu",
        str(arguments.gpus[0]),
        "--sentence-transformers-gpu",
        str(arguments.gpus[1]),
        "--export",
    ]
    if reference.persistent_workers:
        command.append("--sentence-transformers-persistent-workers")
    if reference.torch_compile:
        command.append("--sentence-transformers-torch-compile")
    for bucket in SEQUENCE_LENGTH_BUCKETS:
        command.extend(("--sequence-length-bucket", str(bucket)))
    return command


def _run(arguments: argparse.Namespace) -> None:
    command = _final_command(arguments)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    commands = parser.add_subparsers(dest="command", required=True)
    tune = commands.add_parser("tune", help="run the bounded execution-only sweep")
    tune.add_argument("--gpus", type=int, nargs=2, required=True)
    tune.add_argument("--seed", type=int, default=7)
    tune.add_argument("--steps", type=int, default=16)
    confirm = commands.add_parser(
        "confirm-compile",
        help="repeat compiled reference candidates with isolated cold caches",
    )
    confirm.add_argument("--gpus", type=int, nargs=2, required=True)
    confirm.add_argument("--seed", type=int, default=7)
    confirm.add_argument("--steps", type=int, default=16)
    run = commands.add_parser("run", help="run the selected full matched pair")
    run.add_argument("--gpus", type=int, nargs=2, required=True)
    run.add_argument("--seed", type=int, choices=QUALITY_SEEDS, default=7)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.gpus[0] == arguments.gpus[1]:
        raise ValueError("the two frameworks require distinct GPUs")
    if arguments.command == "tune":
        _tune(arguments)
    elif arguments.command == "confirm-compile":
        _confirm_compile(arguments)
    else:
        _run(arguments)


if __name__ == "__main__":
    main()
