"""Train the three-seed ModernBERT dense-retrieval convergence result."""

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
OUTPUT = PAPER_ROOT / "11-dense-retrieval-convergence"
DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
TRAINING_DATA = ASSET_ROOT / "dense-msmarco-full-unique"
CHECKPOINT = ASSET_ROOT / "ettin-encoder-150m"

MODEL_ID = "jhu-clsp/ettin-encoder-150m"
MODEL_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
SEEDS = (7, 42, 773)
GLOBAL_BATCH_SIZE = 128
MAXIMUM_LENGTH = 128
SEQUENCE_BUCKETS = (16, 96, 128)
QUERY_CHUNK_SIZE = 128
DOCUMENT_CHUNK_SIZE = 64
LOSS_ROW_CHUNK_SIZE = 64
WARMUP_RATIO = 0.06
SOURCE_SHARDS = 39


def contract() -> dict[str, Any]:
    return {
        "experiment": "11-dense-retrieval-convergence",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "training_data": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "selection": "all nonempty rows with globally unique queries and positives",
            "order": "independent deterministic permutation per seed",
        },
        "seeds": list(SEEDS),
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "maximum_length": MAXIMUM_LENGTH,
        "sequence_length_buckets": list(SEQUENCE_BUCKETS),
        "grad_cache": {
            "implementation": "custom_vjp",
            "query_micro_batch_size": QUERY_CHUNK_SIZE,
            "document_micro_batch_size": DOCUMENT_CHUNK_SIZE,
            "loss_row_chunk_size": LOSS_ROW_CHUNK_SIZE,
        },
        "loss": {"name": "mnr", "scale": 20.0, "symmetric": False},
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
            "warmup_ratio": WARMUP_RATIO,
            "schedule": "cosine",
            "gradient_clip_norm": 1.0,
            "precision": "bfloat16-compute-float32-parameters",
        },
        "evaluation": {
            "during_training": ["NanoMSMARCO-start", "NanoMSMARCO-final"],
            "post_training": ["TREC-DL-2019", "Natural-Questions"],
        },
        "checkpoint_progress": [0.5, 1.0],
        "export": "representax-and-huggingface",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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


def prepare_checkpoint(output: Path = CHECKPOINT) -> None:
    required = (output / "config.json", output / "model.safetensors")
    if all(path.is_file() for path in required):
        return
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, ModernBertModel

    source = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    output.mkdir(parents=True, exist_ok=True)
    ModernBertModel.from_pretrained(source, local_files_only=True).save_pretrained(
        output,
        safe_serialization=True,
    )
    AutoTokenizer.from_pretrained(source, local_files_only=True).save_pretrained(output)
    _write_json(
        output / "source.json",
        {"id": MODEL_ID, "revision": MODEL_REVISION},
    )


def prepare_evaluation_data() -> None:
    if (DATA / "manifest.json").is_file():
        return
    subprocess.run(
        (
            str(PYTHON),
            "-m",
            "benchmarks.dense_retrieval",
            "prepare",
            "--data-directory",
            str(DATA),
        ),
        cwd=ROOT,
        check=True,
    )


def prepare_training_data(output: Path = TRAINING_DATA) -> dict[str, Any]:
    """Materialize every eligible unique pair and one order per paper seed."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prepared data directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    queries: set[str] = set()
    positives: set[str] = set()
    source_files = []
    for shard in range(SOURCE_SHARDS):
        filename = f"triplet-all/train-{shard:05d}-of-{SOURCE_SHARDS:05d}.parquet"
        path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=DATASET_REVISION,
                filename=filename,
                local_dir=output / "source",
            )
        )
        source_files.append(
            {"file": filename, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
        source_row = 0
        for batch in parquet.ParquetFile(path, memory_map=True).iter_batches(
            batch_size=4096,
            columns=("query", "positive"),
        ):
            for offset, value in enumerate(batch.to_pylist()):
                query = str(value["query"]).strip()
                positive = str(value["positive"]).strip()
                if (
                    not query
                    or not positive
                    or query in queries
                    or positive in positives
                ):
                    continue
                queries.add(query)
                positives.add(positive)
                rows.append(
                    {
                        "query": query,
                        "positive": positive,
                        "source_shard": shard,
                        "source_row": source_row + offset,
                    }
                )
            source_row += batch.num_rows
    usable_rows = len(rows) - len(rows) % GLOBAL_BATCH_SIZE
    if usable_rows < GLOBAL_BATCH_SIZE:
        raise ValueError("MS MARCO did not produce one complete unique training batch")
    table = pa.Table.from_pylist(rows[:usable_rows])
    seed_files = {}
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(usable_rows)
        seeded = table.take(pa.array(permutation))
        path = output / f"seed-{seed}.parquet"
        parquet.write_table(seeded, path, compression="zstd")
        seed_files[str(seed)] = {
            "file": path.name,
            "rows": seeded.num_rows,
            "sha256": _sha256(path),
        }
    manifest = {
        **contract()["training_data"],
        "usable_rows": usable_rows,
        "optimizer_steps": usable_rows // GLOBAL_BATCH_SIZE,
        "source_files": source_files,
        "seed_files": seed_files,
    }
    _write_json(manifest_path, manifest)
    return manifest


def _training_steps(seed: int) -> int:
    manifest = json.loads((TRAINING_DATA / "manifest.json").read_text())
    record = manifest["seed_files"][str(seed)]
    path = TRAINING_DATA / record["file"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"training data hash changed: {path}")
    rows = int(record["rows"])
    if rows % GLOBAL_BATCH_SIZE:
        raise ValueError("training rows must divide the frozen global batch")
    return rows // GLOBAL_BATCH_SIZE


def worker_command(seed: int, gpu: int, *, steps: int | None = None) -> list[str]:
    steps = _training_steps(seed) if steps is None else steps
    if steps < 2:
        raise ValueError("optimizer steps must be at least two")
    run = OUTPUT / "runs" / f"seed-{seed}"
    command = [
        str(PYTHON),
        "-m",
        "benchmarks.dense_retrieval",
        "worker",
        "--framework",
        "representax",
        "--model",
        "modernbert",
        "--checkpoint",
        str(CHECKPOINT),
        "--data-directory",
        str(DATA),
        "--training-parquet",
        str(TRAINING_DATA / f"seed-{seed}.parquet"),
        "--batch-size",
        str(GLOBAL_BATCH_SIZE),
        "--steps",
        str(steps),
        "--maximum-length",
        str(MAXIMUM_LENGTH),
        "--cache-chunk-size",
        str(DOCUMENT_CHUNK_SIZE),
        "--representax-query-cache-chunk-size",
        str(QUERY_CHUNK_SIZE),
        "--representax-document-cache-chunk-size",
        str(DOCUMENT_CHUNK_SIZE),
        "--representax-loss-row-chunk-size",
        str(LOSS_ROW_CHUNK_SIZE),
        "--grad-cache-implementation",
        "custom_vjp",
        "--evaluation-batch-size",
        "128",
        "--data-threads",
        "4",
        "--prefetch-buffer-size",
        "8",
        "--seed",
        str(seed),
        "--world-size",
        "1",
        "--checkpoint-every",
        str(steps // 2),
        "--mixed-precision",
        "--telemetry",
        "--export",
        "--run-directory",
        str(run / "run"),
        "--report",
        str(run / "report.json"),
    ]
    for bucket in SEQUENCE_BUCKETS:
        command.extend(("--sequence-length-bucket", str(bucket)))
    return command


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
        raise RuntimeError(f"dense convergence workers failed: {failed}")
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
        prepare_checkpoint()
        prepare_evaluation_data()
        print(json.dumps(prepare_training_data(), indent=2, sort_keys=True))
    elif arguments.command == "run":
        run_seed(arguments.seed, arguments.gpu)
    elif arguments.command == "all":
        run_all(tuple(arguments.gpus))
    else:
        aggregate()


if __name__ == "__main__":
    main()
