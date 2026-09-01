"""Launch serious paper experiment 01: MPNet dense retrieval on MS MARCO."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = (
    Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
    / "01-dense-model-training"
)
DEFAULT_CHECKPOINT = Path("/raid/representax/oracles/all-mpnet-base-v2")
DEFAULT_DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
QUALITY_SEEDS = (7, 42, 773)
SEQUENCE_LENGTH_BUCKETS = (16, 32, 64, 128, 256)


def _common_arguments(seed: int, checkpoint: Path, data: Path) -> list[str]:
    arguments = [
        "--model",
        "mpnet",
        "--checkpoint",
        str(checkpoint),
        "--data-directory",
        str(data),
        "--batch-size",
        "2048",
        "--steps",
        "256",
        "--maximum-length",
        "256",
        "--cache-chunk-size",
        "32",
        "--evaluation-batch-size",
        "128",
        "--evaluation-every-steps",
        "64",
        "--data-threads",
        "4",
        "--prefetch-buffer-size",
        "8",
        "--seed",
        str(seed),
        "--world-size",
        "1",
        "--mixed-precision",
        "--telemetry",
        "--export",
    ]
    for bucket in SEQUENCE_LENGTH_BUCKETS:
        arguments.extend(("--sequence-length-bucket", str(bucket)))
    return arguments


def _benchmark_command(command: str) -> list[str]:
    return [sys.executable, "-m", "benchmarks.dense_retrieval", command]


def _run_command(
    seed: int,
    gpus: tuple[int, int],
    artifact_root: Path,
    checkpoint: Path,
    data: Path,
) -> list[str]:
    if gpus[0] == gpus[1]:
        raise ValueError("the paired frameworks require distinct GPUs")
    return [
        *_benchmark_command("pair"),
        *_common_arguments(seed, checkpoint, data),
        "--result-directory",
        str(artifact_root / "runs" / f"seed-{seed}"),
        "--representax-gpu",
        str(gpus[0]),
        "--sentence-transformers-gpu",
        str(gpus[1]),
    ]


def _aggregate_command(artifact_root: Path) -> list[str]:
    command = _benchmark_command("aggregate")
    for seed in QUALITY_SEEDS:
        command.extend(
            ("--summary", str(artifact_root / "runs" / f"seed-{seed}" / "summary.json"))
        )
    command.extend(("--output", str(artifact_root / "three-seed-summary.json")))
    return command


def _execute(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(
        command,
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="train both frameworks, export both models, and evaluate both",
    )
    run.add_argument("--seed", type=int, choices=QUALITY_SEEDS, required=True)
    run.add_argument("--gpus", type=int, nargs=2, required=True)

    commands.add_parser(
        "aggregate",
        help="combine runs/seed-{7,42,773}/summary.json",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        _execute(
            _run_command(
                arguments.seed,
                tuple(arguments.gpus),
                arguments.artifact_root,
                arguments.checkpoint,
                arguments.data,
            )
        )
    else:
        _execute(_aggregate_command(arguments.artifact_root))


if __name__ == "__main__":
    main()
