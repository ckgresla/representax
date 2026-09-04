"""Run the three-seed text-only ModernBERT dense-retrieval comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON = REPOSITORY_ROOT / "experiments/.venv/bin/python"
ARTIFACT_ROOT = Path(
    os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper")
) / "09-modernbert-dense-retrieval"
SOURCE_CHECKPOINT = Path(
    "/raid/.cache/huggingface/hub/"
    "models--jhu-clsp--ettin-encoder-150m/"
    "snapshots/45d08642849e5c5701b162671ac811b7654bfd9f"
)
CHECKPOINT = ARTIFACT_ROOT / "checkpoints/ettin-encoder-150m"
DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
SEEDS = (17, 42, 73)
TRAJECTORY_ROOT = ARTIFACT_ROOT / "real-trajectory-30-step" / "seed-17"
PADDING_ABLATION_ROOT = ARTIFACT_ROOT / "padding-ablation-30-step" / "seed-17"


def _prepare_checkpoint() -> None:
    required = (CHECKPOINT / "config.json", CHECKPOINT / "model.safetensors")
    if all(path.is_file() for path in required):
        return
    if not SOURCE_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"source checkpoint not found: {SOURCE_CHECKPOINT}")

    from transformers import AutoTokenizer, ModernBertModel

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    model = ModernBertModel.from_pretrained(SOURCE_CHECKPOINT, local_files_only=True)
    model.save_pretrained(CHECKPOINT, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_CHECKPOINT,
        local_files_only=True,
    )
    tokenizer.save_pretrained(CHECKPOINT)
    (CHECKPOINT / "source.json").write_text(
        json.dumps(
            {
                "model_id": "jhu-clsp/ettin-encoder-150m",
                "revision": "45d08642849e5c5701b162671ac811b7654bfd9f",
                "conversion": "ModernBertModel.save_pretrained(safe_serialization=True)",
            },
            indent=2,
        )
        + "\n"
    )


def _pair_command(seed: int, gpus: tuple[int, int]) -> list[str]:
    return [
        str(PYTHON),
        "-m",
        "benchmarks.dense_retrieval",
        "pair",
        "--model",
        "modernbert",
        "--checkpoint",
        str(CHECKPOINT),
        "--data-directory",
        str(DATA),
        "--batch-size",
        "128",
        "--steps",
        "100",
        "--maximum-length",
        "128",
        "--representax-cache-chunk-size",
        "64",
        "--sentence-transformers-cache-chunk-size",
        "128",
        "--sequence-length-bucket",
        "16",
        "--sequence-length-bucket",
        "32",
        "--sequence-length-bucket",
        "64",
        "--sequence-length-bucket",
        "128",
        "--data-threads",
        "4",
        "--prefetch-buffer-size",
        "8",
        "--sentence-transformers-data-threads",
        "0",
        "--sentence-transformers-torch-compile",
        "--mixed-precision",
        "--telemetry",
        "--seed",
        str(seed),
        "--result-directory",
        str(ARTIFACT_ROOT / "runs" / f"seed-{seed}"),
        "--representax-gpu",
        str(gpus[0]),
        "--sentence-transformers-gpu",
        str(gpus[1]),
    ]


def _aggregate_command() -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "benchmarks.dense_retrieval",
        "aggregate",
    ]
    for seed in SEEDS:
        command.extend(
            (
                "--summary",
                str(ARTIFACT_ROOT / "runs" / f"seed-{seed}" / "summary.json"),
            )
        )
    command.extend(("--output", str(ARTIFACT_ROOT / "three-seed-summary.json")))
    return command


def _trajectory_command(
    name: str,
    output: Path,
    *,
    sequence_length_buckets: tuple[int, ...] = (16, 32, 64, 128),
    fixed_lengths: tuple[int, int] | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "benchmarks.dense_retrieval",
        "worker",
        "--framework",
        (
            "representax"
            if name in {"custom-vjp", "rematerialized"}
            else "sentence-transformers"
        ),
        "--model",
        "modernbert",
        "--checkpoint",
        str(CHECKPOINT),
        "--data-directory",
        str(DATA),
        "--batch-size",
        "128",
        "--steps",
        "30",
        "--maximum-length",
        "128",
        "--evaluation-batch-size",
        "128",
        "--seed",
        "17",
        "--mixed-precision",
        "--telemetry",
        "--cache-chunk-size",
        "64" if name in {"custom-vjp", "rematerialized"} else "128",
        "--run-directory",
        str(output / "run"),
        "--report",
        str(output / "report.json"),
    ]
    if name in {"custom-vjp", "rematerialized"}:
        command.extend(
            (
                "--grad-cache-implementation",
                "custom_vjp" if name == "custom-vjp" else "rematerialized",
                "--data-threads",
                "4",
                "--prefetch-buffer-size",
                "8",
            )
        )
        for bucket in sequence_length_buckets:
            command.extend(("--sequence-length-bucket", str(bucket)))
    else:
        command.extend(
            (
                "--sentence-transformers-data-threads",
                "0",
                "--sentence-transformers-prefetch-buffer-size",
                "8",
            )
        )
        if name == "st-inductor":
            command.extend(
                (
                    "--sentence-transformers-torch-compile",
                    "--sentence-transformers-torch-compile-backend",
                    "inductor",
                )
            )
        if fixed_lengths is not None:
            command.extend(
                (
                    "--sentence-transformers-query-length",
                    str(fixed_lengths[0]),
                    "--sentence-transformers-document-length",
                    str(fixed_lengths[1]),
                )
            )
    return command


def _parallel_workers(
    gpus: tuple[int, ...],
    output: Path,
    commands: tuple[tuple[str, list[str]], ...],
) -> None:
    if (
        len(gpus) != len(commands)
        or len(set(gpus)) != len(gpus)
        or any(gpu < 0 for gpu in gpus)
    ):
        raise ValueError(
            f"comparison requires {len(commands)} unique non-negative GPU indices"
        )
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    _prepare_checkpoint()
    output.mkdir(parents=True)
    processes = []
    streams = []
    failures = []
    try:
        for (name, command), gpu in zip(commands, gpus, strict=True):
            destination = output / name
            destination.mkdir()
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            framework = command[command.index("--framework") + 1]
            if framework == "representax":
                environment["REPRESENTAX_JAX_CACHE_DIR"] = str(
                    output / "cache" / f"jax-{name}"
                )
            elif "--sentence-transformers-torch-compile" in command:
                environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                    output / "cache" / f"torchinductor-{name}"
                )
            stream = (destination / "worker.log").open("x")
            streams.append(stream)
            processes.append(
                (
                    name,
                    subprocess.Popen(
                        command,
                        cwd=REPOSITORY_ROOT,
                        env=environment,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
        for name, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((name, return_code))
    finally:
        for stream in streams:
            stream.close()
    if failures:
        raise RuntimeError(f"comparison workers failed: {failures}")


def _trajectory(gpus: tuple[int, ...], output: Path) -> None:
    names = ("custom-vjp", "rematerialized", "st-eager", "st-inductor")
    commands = tuple(
        (name, _trajectory_command(name, output / name)) for name in names
    )
    _parallel_workers(gpus, output, commands)


def _padding_ablation(gpus: tuple[int, ...], output: Path) -> None:
    commands = (
        (
            "st-eager-fixed",
            _trajectory_command(
                "st-eager",
                output / "st-eager-fixed",
                fixed_lengths=(16, 128),
            ),
        ),
        (
            "st-inductor-fixed",
            _trajectory_command(
                "st-inductor",
                output / "st-inductor-fixed",
                fixed_lengths=(16, 128),
            ),
        ),
        (
            "rx-fine-buckets",
            _trajectory_command(
                "custom-vjp",
                output / "rx-fine-buckets",
                sequence_length_buckets=(16, 64, 80, 96, 112, 128),
            ),
        ),
    )
    _parallel_workers(gpus, output, commands)


def _run(gpus: tuple[int, ...]) -> None:
    if len(gpus) != 6 or len(set(gpus)) != 6 or any(gpu < 0 for gpu in gpus):
        raise ValueError("run requires six unique non-negative GPU indices")
    if not PYTHON.is_file():
        raise FileNotFoundError(f"experiment Python not found: {PYTHON}")
    _prepare_checkpoint()
    if not DATA.is_dir():
        raise FileNotFoundError(f"prepared data not found: {DATA}")

    environment = os.environ.copy()
    environment["REPRESENTAX_JAX_CACHE_DIR"] = str(ARTIFACT_ROOT / "cache/jax")
    environment["REPRESENTAX_TORCHINDUCTOR_CACHE_DIR"] = str(
        ARTIFACT_ROOT / "cache/torchinductor"
    )
    processes = []
    for index, seed in enumerate(SEEDS):
        output = ARTIFACT_ROOT / "runs" / f"seed-{seed}" / "summary.json"
        if output.is_file():
            print(f"already complete: seed {seed}", flush=True)
            continue
        gpus_for_seed = (gpus[2 * index], gpus[2 * index + 1])
        command = _pair_command(seed, gpus_for_seed)
        print(" ".join(command), flush=True)
        processes.append(
            (seed, subprocess.Popen(command, cwd=REPOSITORY_ROOT, env=environment))
        )

    failures = []
    for seed, process in processes:
        return_code = process.wait()
        if return_code:
            failures.append((seed, return_code))
    if failures:
        raise RuntimeError(f"paired runs failed: {failures}")
    subprocess.run(_aggregate_command(), cwd=REPOSITORY_ROOT, check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run all three pairs concurrently")
    run.add_argument("--gpus", type=int, nargs=6, required=True)
    trajectory = commands.add_parser(
        "trajectory",
        help="run the four-way 30-update real-data comparison",
    )
    trajectory.add_argument("--gpus", type=int, nargs=4, required=True)
    trajectory.add_argument("--output", type=Path, default=TRAJECTORY_ROOT)
    padding = commands.add_parser(
        "padding-ablation",
        help="run the three-way 30-update padding isolation",
    )
    padding.add_argument("--gpus", type=int, nargs=3, required=True)
    padding.add_argument("--output", type=Path, default=PADDING_ABLATION_ROOT)
    commands.add_parser("prepare", help="convert the pinned encoder to safetensors")
    commands.add_parser("aggregate", help="rebuild the three-seed summary")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        _run(tuple(arguments.gpus))
    elif arguments.command == "trajectory":
        _trajectory(tuple(arguments.gpus), arguments.output.expanduser().resolve())
    elif arguments.command == "padding-ablation":
        _padding_ablation(
            tuple(arguments.gpus), arguments.output.expanduser().resolve()
        )
    elif arguments.command == "prepare":
        _prepare_checkpoint()
    else:
        subprocess.run(_aggregate_command(), cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
