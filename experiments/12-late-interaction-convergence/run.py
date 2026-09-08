"""Train the three-seed GTE-ModernColBERT convergence result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(
    os.environ.get(
        "REPRESENTAX_EXPERIMENT_PYTHON", ROOT / "experiments/.venv/bin/python"
    )
)
PAPER_ROOT = Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
ASSET_ROOT = Path(
    os.environ.get("REPRESENTAX_PAPER_ASSETS", "/raid/representax-paper-assets")
)
OUTPUT = PAPER_ROOT / "12-late-interaction-convergence"
DATA = ASSET_ROOT / "late-interaction-convergence"
TRAINING_DATA = ASSET_ROOT / "dense-msmarco-full-unique"
NANOBEIR = (
    Path("/raid/representax/data/dense-retrieval-msmarco-v1/artifacts")
    / "sentence-transformers/NanoBEIR-en"
)
CHECKPOINT = ASSET_ROOT / "late-checkpoint"

MODEL_ID = "lightonai/GTE-ModernColBERT-v1"
MODEL_REVISION = "cbbe53366e564450558f5e639dd499171f127538"
SEEDS = (7, 42, 773)
STEPS = 1_000
GLOBAL_BATCH_SIZE = 512
TRAINING_PRESENTATIONS = STEPS * GLOBAL_BATCH_SIZE
QUERY_BUCKETS = (16, 32)
DOCUMENT_BUCKETS = (32, 64, 128, 256)
GRAD_CACHE_MICRO_BATCH = 8
WARMUP_STEPS = round(STEPS * 0.06)


def contract() -> dict[str, Any]:
    return {
        "experiment": "12-late-interaction-convergence",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "training_data": {
            "id": "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3",
            "revision": "0d54352548089199bde15ad7e06efe895dc80b56",
            "duplicate_queries": 0,
            "duplicate_positives": 0,
            "order": "independent deterministic permutation per seed",
        },
        "seeds": list(SEEDS),
        "optimizer_steps": STEPS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "training_presentations": TRAINING_PRESENTATIONS,
        "query_sequence_length_buckets": list(QUERY_BUCKETS),
        "document_sequence_length_buckets": list(DOCUMENT_BUCKETS),
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "loss": {
            "name": "late-interaction-contrastive",
            "temperature": 0.02,
            "symmetric": False,
            "negative_scope": "global",
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 3e-6,
            "weight_decay": 0.0,
            "warmup_steps": WARMUP_STEPS,
            "schedule": "cosine",
            "gradient_clip_norm": 1.0,
            "precision": "bfloat16-compute-float32-parameters",
        },
        "evaluation": ["NanoMSMARCO-start", "NanoMSMARCO-final"],
        "checkpoint_progress": [0.5, 1.0],
        "export": "representax-and-huggingface",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ("git", "diff", "--binary"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "working_tree_clean": not bool(patch),
        "working_tree_patch_sha256": "sha256:" + hashlib.sha256(patch).hexdigest(),
    }


def prepare_data() -> None:
    import pyarrow.parquet as parquet

    if not (TRAINING_DATA / "manifest.json").is_file():
        raise FileNotFoundError("prepare experiment 11 training data first")
    for seed in SEEDS:
        destination = DATA / f"seed-{seed}"
        if (destination / "manifest.json").is_file():
            continue
        source = TRAINING_DATA / f"seed-{seed}.parquet"
        rows = min(
            TRAINING_PRESENTATIONS,
            parquet.ParquetFile(source, memory_map=True).metadata.num_rows,
        )
        subprocess.run(
            (
                str(PYTHON),
                "-m",
                "experiments.preflights.late_interaction",
                "prepare",
                "--output",
                str(destination),
                "--training-parquet",
                str(source),
                "--nanobeir-directory",
                str(NANOBEIR),
                "--training-rows",
                str(rows),
            ),
            cwd=ROOT,
            check=True,
        )


def worker_command(seed: int, gpu: int) -> list[str]:
    run = OUTPUT / "runs" / f"seed-{seed}"
    return [
        str(PYTHON),
        "-m",
        "experiments.preflights.late_interaction",
        "worker",
        "--framework",
        "representax",
        "--checkpoint",
        str(CHECKPOINT),
        "--data-directory",
        str(DATA / f"seed-{seed}"),
        "--run-directory",
        str(run / "run"),
        "--report",
        str(run / "report.json"),
        "--steps",
        str(STEPS),
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--seed",
        str(seed),
        "--platform",
        "gpu",
        "--negative-scope",
        "global",
    ]


def _environment(gpu: int) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HF_HOME": os.environ.get("HF_HOME", "/raid/.cache/huggingface"),
        "JAX_COMPILATION_CACHE_DIR": str(OUTPUT / "jax-cache"),
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }


def run_seed(seed: int, gpu: int) -> None:
    data = DATA / f"seed-{seed}" / "manifest.json"
    if not data.is_file():
        raise FileNotFoundError(f"prepared seed data is missing: {data}")
    run = OUTPUT / "runs" / f"seed-{seed}"
    if run.exists():
        raise FileExistsError(f"run already exists: {run}")
    run.mkdir(parents=True)
    command = worker_command(seed, gpu)
    _write_json(
        run / "launch.json",
        {"contract": contract(), "git": _git_state(), "command": command, "gpu": gpu},
    )
    print(shlex.join(command), flush=True)
    with (run / "worker.log").open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_all(gpus: tuple[int, ...]) -> None:
    if len(gpus) != len(SEEDS) or len(set(gpus)) != len(gpus):
        raise ValueError("all requires three distinct GPU indices")
    processes = []
    for seed, gpu in zip(SEEDS, gpus, strict=True):
        command = [str(PYTHON), __file__, "run", "--seed", str(seed), "--gpu", str(gpu)]
        processes.append((seed, subprocess.Popen(command, cwd=ROOT)))
    failures = [(seed, process.wait()) for seed, process in processes]
    failed = [(seed, code) for seed, code in failures if code]
    if failed:
        raise RuntimeError(f"late-interaction convergence workers failed: {failed}")
    aggregate()


def aggregate() -> None:
    reports = {}
    for seed in SEEDS:
        path = OUTPUT / "runs" / f"seed-{seed}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed report: {path}")
        reports[str(seed)] = json.loads(path.read_text())
    _write_json(OUTPUT / "summary.json", {"contract": contract(), "runs": reports})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("contract")
    commands.add_parser("prepare")
    run = commands.add_parser("run")
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--gpu", type=int, required=True)
    all_runs = commands.add_parser("all")
    all_runs.add_argument("--gpus", type=int, nargs=3, required=True)
    commands.add_parser("aggregate")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "contract":
        print(json.dumps(contract(), indent=2, sort_keys=True))
    elif arguments.command == "prepare":
        prepare_data()
    elif arguments.command == "run":
        run_seed(arguments.seed, arguments.gpu)
    elif arguments.command == "all":
        run_all(tuple(arguments.gpus))
    else:
        aggregate()


if __name__ == "__main__":
    main()
