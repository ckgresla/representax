"""Launch serious paper experiment 02: labeled sentence-pair training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_ARTIFACT_ROOT = (
    Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
    / "02-semantic-similarity-pair-classification"
)
QUALITY_SEEDS = (7, 42, 773)
FRAMEWORKS = ("representax", "sentence-transformers")
MODELS = ("mpnet-base", "bert-base")
MICRO_BATCH_SIZE = 32
Workload = Literal["semantic-similarity", "pair-classification"]
WORKLOADS: tuple[Workload, ...] = (
    "semantic-similarity",
    "pair-classification",
)
EPOCHS = 4


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def training_steps(workload: Workload, data_directory: Path) -> int:
    from experiments.preflights.semantic_pair import _data_paths, frozen_contract

    contract = frozen_contract(workload)
    train, _ = _data_paths(workload, data_directory)
    steps_per_epoch = _line_count(train) // contract.batch_size
    if steps_per_epoch < 1:
        raise ValueError(f"{workload} has no complete global batch")
    return EPOCHS * steps_per_epoch


def _checkpoint_directory(artifact_root: Path, model: str) -> Path:
    return artifact_root / "checkpoints" / model


def _prepare_checkpoint(artifact_root: Path, model: str) -> Path:
    from experiments.preflights.semantic_pair import frozen_contract
    from huggingface_hub import snapshot_download

    contract = frozen_contract("semantic-similarity", model)
    output = _checkpoint_directory(artifact_root, model)
    if (output / "modules.json").is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if model == "mpnet-base":
        snapshot_download(
            repo_id=contract.model_id,
            revision=contract.model_revision,
            local_dir=output,
        )
        return output

    from sentence_transformers import SentenceTransformer

    source = artifact_root / "checkpoints" / "bert-base-source"
    snapshot_download(
        repo_id=contract.model_id,
        revision=contract.model_revision,
        local_dir=source,
    )
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    sentence_encoder = SentenceTransformer(
        str(source),
        device="cpu",
        local_files_only=True,
    )
    sentence_encoder.save(str(temporary))
    os.replace(temporary, output)
    return output


def prepare(artifact_root: Path) -> None:
    from experiments.preflights.semantic_pair import prepare_data

    artifact_root.mkdir(parents=True, exist_ok=True)
    data = artifact_root / "data"
    if not (data / "manifest.json").is_file():
        temporary = data.with_name(data.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        prepare_data(temporary)
        os.replace(temporary, data)
    checkpoints = {
        model: str(_prepare_checkpoint(artifact_root, model)) for model in MODELS
    }
    _write_json(
        artifact_root / "prepared.json",
        {
            "data": str(data),
            "checkpoints": checkpoints,
            "epochs": EPOCHS,
            "steps": {
                workload: training_steps(workload, data) for workload in WORKLOADS
            },
        },
    )


def _source_snapshot(output: Path) -> dict[str, Any]:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=False)

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    commit = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1")
    patch = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    (source / "working-tree.patch").write_bytes(patch)
    shutil.copy2(__file__, source / "run.py")
    shell = Path(__file__).with_name("run.sh")
    if shell.is_file():
        shutil.copy2(shell, source / "run.sh")
    digest = hashlib.sha256(patch).hexdigest()
    return {
        "representax_commit": commit,
        "working_tree_clean": not bool(status),
        "working_tree_status": status.splitlines(),
        "working_tree_patch_sha256": f"sha256:{digest}",
    }


def _worker_command(
    *,
    workload: Workload,
    model: str,
    framework: str,
    checkpoint: Path,
    data: Path,
    output: Path,
    steps: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.preflights.semantic_pair",
        "worker",
        "--workload",
        workload,
        "--model",
        model,
        "--serious",
        "--framework",
        framework,
        "--checkpoint",
        str(checkpoint),
        "--data-directory",
        str(data),
        "--run-directory",
        str(output / framework),
        "--report",
        str(output / f"{framework}.json"),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
    ]


def _worker_environment(gpu: int, output: Path, framework: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
        }
    )
    if framework == "representax":
        environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
        environment.update(
            {
                "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            }
        )
    else:
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    return environment


def _pair_metrics(model: Any, processor: Any, data: Path) -> dict[str, float]:
    from experiments.preflights.semantic_pair import (
        PairEvaluationCollator,
        _read_jsonl,
    )

    from representax.evaluation import PairClassificationEvaluator
    from representax.train import EvaluationRunner

    rows = _read_jsonl(data / "pair-evaluation.jsonl")
    collator = PairEvaluationCollator(
        processor=processor,
        pad_to_size=MICRO_BATCH_SIZE,
    )

    def batches() -> Iterable[Any]:
        for start in range(0, len(rows), MICRO_BATCH_SIZE):
            yield collator(rows[start : start + MICRO_BATCH_SIZE])

    result = EvaluationRunner(
        PairClassificationEvaluator(
            name="sprint",
            similarity_functions=("cosine",),
        )
    ).run(model, batches())
    return {name: float(value) for name, value in result.metrics.items()}


def _evaluate_model(
    workload: Workload,
    model: Any,
    processor: Any,
    data: Path,
) -> dict[str, float]:
    from experiments.preflights.semantic_pair import _similarity_metrics

    if workload == "pair-classification":
        return _pair_metrics(model, processor, data)
    metrics = _similarity_metrics(
        model,
        processor,
        data / "semantic-validation.jsonl",
        MICRO_BATCH_SIZE,
        name="stsb",
    )
    metrics.update(
        _similarity_metrics(
            model,
            processor,
            data / "semantic-sick-r.jsonl",
            MICRO_BATCH_SIZE,
            name="sick_r",
        )
    )
    return metrics


def evaluate(arguments: argparse.Namespace) -> None:
    import gc

    import jax
    from experiments.preflights.semantic_pair import (
        _model_and_processor,
        frozen_contract,
    )

    from representax.export import load_inference_bundle

    contract = frozen_contract(arguments.workload, arguments.model)
    source_model, processor = _model_and_processor(arguments.checkpoint, contract)
    del source_model
    gc.collect()
    jax.clear_caches()

    native_model, _ = load_inference_bundle(arguments.representax_export)
    native = _evaluate_model(
        arguments.workload,
        native_model,
        processor,
        arguments.data,
    )
    del native_model
    gc.collect()
    jax.clear_caches()

    reference_model, reference_processor = _model_and_processor(
        arguments.reference_export,
        contract,
    )
    reference = _evaluate_model(
        arguments.workload,
        reference_model,
        reference_processor,
        arguments.data,
    )
    _write_json(
        arguments.output,
        {
            "evaluator": "representax",
            "workload": arguments.workload,
            "model": arguments.model,
            "representax": native,
            "sentence-transformers": reference,
        },
    )


def run(arguments: argparse.Namespace) -> None:
    artifact_root = arguments.artifact_root.resolve()
    data = artifact_root / "data"
    checkpoint = _checkpoint_directory(artifact_root, arguments.model)
    prepared = (data / "manifest.json").is_file() and (
        checkpoint / "modules.json"
    ).is_file()
    if not prepared:
        raise RuntimeError(
            f"experiment is not prepared; run {Path(__file__).name} prepare"
        )
    output = (
        artifact_root
        / "runs"
        / arguments.workload
        / arguments.model
        / f"seed-{arguments.seed}"
    )
    output.mkdir(parents=True, exist_ok=False)
    source = _source_snapshot(output)
    steps = training_steps(arguments.workload, data)
    running = []
    for framework, gpu in zip(FRAMEWORKS, arguments.gpus, strict=True):
        log = output / f"{framework}.log"
        stream = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            _worker_command(
                workload=arguments.workload,
                model=arguments.model,
                framework=framework,
                checkpoint=checkpoint,
                data=data,
                output=output,
                steps=steps,
                seed=arguments.seed,
            ),
            cwd=REPOSITORY_ROOT,
            env=_worker_environment(gpu, output, framework),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        running.append((framework, gpu, process, stream, log))
    failures = []
    for framework, gpu, process, stream, log in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(
                {
                    "framework": framework,
                    "gpu": gpu,
                    "return_code": return_code,
                    "log": str(log),
                }
            )
    if failures:
        _write_json(output / "failures.json", failures)
        raise RuntimeError(f"experiment workers failed: {failures}")

    reports = {
        framework: json.loads((output / f"{framework}.json").read_text())
        for framework in FRAMEWORKS
    }
    shared = output / "shared-evaluation.json"
    evaluate_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "evaluate",
        "--workload",
        arguments.workload,
        "--model",
        arguments.model,
        "--checkpoint",
        str(checkpoint),
        "--data",
        str(data),
        "--representax-export",
        reports["representax"]["inference_bundle"],
        "--reference-export",
        reports["sentence-transformers"]["inference_bundle"],
        "--output",
        str(shared),
    ]
    evaluation_log = (output / "shared-evaluation.log").open("x", encoding="utf-8")
    try:
        subprocess.run(
            evaluate_command,
            check=True,
            cwd=REPOSITORY_ROOT,
            env=_worker_environment(arguments.gpus[0], output, "representax"),
            stdout=evaluation_log,
            stderr=subprocess.STDOUT,
        )
    finally:
        evaluation_log.close()
    native_rate = reports["representax"]["steady_state"][
        "aggregate_examples_per_second"
    ]
    reference_rate = reports["sentence-transformers"]["steady_state"][
        "aggregate_examples_per_second"
    ]
    summary = {
        "workload": arguments.workload,
        "model": arguments.model,
        "seed": arguments.seed,
        "epochs": EPOCHS,
        "steps": steps,
        "source": source,
        "reports": reports,
        "shared_evaluation": json.loads(shared.read_text()),
        "steady_state_speedup": native_rate / reference_rate,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _expected_summaries(artifact_root: Path) -> tuple[Path, ...]:
    return tuple(
        artifact_root / "runs" / workload / model / f"seed-{seed}" / "summary.json"
        for workload in WORKLOADS
        for model in MODELS
        for seed in QUALITY_SEEDS
    )


def _timing_report(
    artifact_root: Path,
    workload: Workload,
    model: str,
    seed: int,
) -> Path:
    return (
        artifact_root
        / "timing-validation"
        / workload
        / model
        / f"seed-{seed}"
        / "sentence-transformers.json"
    )


def _expected_timing_reports(artifact_root: Path) -> tuple[Path, ...]:
    return tuple(
        _timing_report(artifact_root, workload, model, seed)
        for workload in WORKLOADS
        for model in MODELS
        for seed in QUALITY_SEEDS
    )


def reference_timing(arguments: argparse.Namespace) -> None:
    artifact_root = arguments.artifact_root.resolve()
    data = artifact_root / "data"
    checkpoint = _checkpoint_directory(artifact_root, arguments.model)
    output = _timing_report(
        artifact_root,
        arguments.workload,
        arguments.model,
        arguments.seed,
    ).parent
    output.mkdir(parents=True, exist_ok=False)
    log = output / "sentence-transformers.log"
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(
            _worker_command(
                workload=arguments.workload,
                model=arguments.model,
                framework="sentence-transformers",
                checkpoint=checkpoint,
                data=data,
                output=output,
                steps=training_steps(arguments.workload, data),
                seed=arguments.seed,
            ),
            check=True,
            cwd=REPOSITORY_ROOT,
            env=_worker_environment(
                arguments.gpu,
                output,
                "sentence-transformers",
            ),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def aggregate(artifact_root: Path) -> None:
    paths = _expected_summaries(artifact_root)
    timing_paths = _expected_timing_reports(artifact_root)
    missing = [
        str(path) for path in (*paths, *timing_paths) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing experiment artifacts: {missing}")
    rows = [json.loads(path.read_text()) for path in paths]
    timing = {
        (workload, model, seed): json.loads(
            _timing_report(artifact_root, workload, model, seed).read_text()
        )
        for workload in WORKLOADS
        for model in MODELS
        for seed in QUALITY_SEEDS
    }
    cells = {}
    for workload in WORKLOADS:
        for model in MODELS:
            selected = [
                row
                for row in rows
                if row["workload"] == workload and row["model"] == model
            ]
            metric = (
                "valid/stsb/spearman_cosine"
                if workload == "semantic-similarity"
                else "valid/sprint/average_precision_max"
            )
            run_rows = []
            for row in selected:
                seed = int(row["seed"])
                native_rate = float(
                    row["reports"]["representax"]["steady_state"][
                        "aggregate_examples_per_second"
                    ]
                )
                paired_reference_rate = float(
                    row["reports"]["sentence-transformers"]["steady_state"][
                        "aggregate_examples_per_second"
                    ]
                )
                isolated_reference_rate = float(
                    timing[(workload, model, seed)]["steady_state"][
                        "aggregate_examples_per_second"
                    ]
                )
                run_rows.append(
                    {
                        "seed": seed,
                        "summary": str(
                            artifact_root
                            / "runs"
                            / workload
                            / model
                            / f"seed-{seed}"
                            / "summary.json"
                        ),
                        "representax_examples_per_second": native_rate,
                        "sentence_transformers_examples_per_second": (
                            isolated_reference_rate
                        ),
                        "steady_state_speedup": native_rate
                        / isolated_reference_rate,
                        "concurrent_reference_examples_per_second": (
                            paired_reference_rate
                        ),
                        "concurrent_speedup": native_rate / paired_reference_rate,
                        "representax_quality": float(
                            row["shared_evaluation"]["representax"][metric]
                        ),
                        "sentence_transformers_quality": float(
                            row["shared_evaluation"]["sentence-transformers"][metric]
                        ),
                    }
                )
            speedups = [row["steady_state_speedup"] for row in run_rows]
            native_quality = [row["representax_quality"] for row in run_rows]
            reference_quality = [
                row["sentence_transformers_quality"] for row in run_rows
            ]
            cells[f"{workload}/{model}"] = {
                "seeds": list(QUALITY_SEEDS),
                "mean_steady_state_speedup": statistics.mean(speedups),
                "speedup_sample_standard_deviation": statistics.stdev(speedups),
                "mean_representax_quality": statistics.mean(native_quality),
                "mean_sentence_transformers_quality": statistics.mean(
                    reference_quality
                ),
                "mean_quality_difference": statistics.mean(
                    native - reference
                    for native, reference in zip(
                        native_quality,
                        reference_quality,
                        strict=True,
                    )
                ),
                "runs": run_rows,
            }
    _write_json(
        artifact_root / "three-seed-summary.json",
        {
            "cells": cells,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")

    execute = commands.add_parser("run")
    execute.add_argument("--workload", choices=WORKLOADS, required=True)
    execute.add_argument("--model", choices=MODELS, required=True)
    execute.add_argument("--seed", type=int, choices=QUALITY_SEEDS, required=True)
    execute.add_argument("--gpus", type=int, nargs=2, required=True)

    timing = commands.add_parser("reference-timing")
    timing.add_argument("--workload", choices=WORKLOADS, required=True)
    timing.add_argument("--model", choices=MODELS, required=True)
    timing.add_argument("--seed", type=int, choices=QUALITY_SEEDS, required=True)
    timing.add_argument("--gpu", type=int, required=True)

    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--workload", choices=WORKLOADS, required=True)
    evaluation.add_argument("--model", choices=MODELS, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--representax-export", type=Path, required=True)
    evaluation.add_argument("--reference-export", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)

    commands.add_parser("aggregate")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        prepare(arguments.artifact_root.resolve())
    elif arguments.command == "run":
        if arguments.gpus[0] == arguments.gpus[1]:
            raise ValueError("paired framework workers require distinct GPUs")
        run(arguments)
    elif arguments.command == "evaluate":
        evaluate(arguments)
    elif arguments.command == "reference-timing":
        reference_timing(arguments)
    else:
        aggregate(arguments.artifact_root.resolve())


if __name__ == "__main__":
    main()
