"""Train the three-seed CLIP image-text convergence result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.preflights.image_text import (  # noqa: E402
    EVALUATION_BATCH_SIZE,
    ImageTextEvaluationCollator,
    _batch_unique_caption_order,
    ensure_bidirectional_flickr_evaluation,
)

PYTHON = Path(
    os.environ.get(
        "REPRESENTAX_EXPERIMENT_PYTHON", ROOT / "experiments/.venv/bin/python"
    )
)
PAPER_ROOT = Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
ASSET_ROOT = Path(
    os.environ.get("REPRESENTAX_PAPER_ASSETS", "/raid/representax-paper-assets")
)
OUTPUT = PAPER_ROOT / "13-image-text-convergence"
DATA = ASSET_ROOT / "image-text-convergence"
CHECKPOINT = ASSET_ROOT / "clip-vit-b-32"

MODEL_ID = "sentence-transformers/clip-ViT-B-32"
MODEL_REVISION = "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
DATASET_ID = "phiyodr/coco2017"
DATASET_REVISION = "036f3f8291db64d17faad9b09e59dd30bb65c4d7"
EVALUATION_ID = "mteb/flickr30kt2i"
EVALUATION_REVISION = "e819702b287bfbe084e129a61f308a802b7c108e"
SEEDS = (7, 42, 773)
GLOBAL_BATCH_SIZE = 512
TRAINING_IMAGES = 117_760
CAPTIONS_PER_IMAGE = 5
STEPS = TRAINING_IMAGES * CAPTIONS_PER_IMAGE // GLOBAL_BATCH_SIZE
GRAD_CACHE_MICRO_BATCH = 8
WARMUP_STEPS = round(STEPS * 0.06)


def contract() -> dict[str, Any]:
    return {
        "experiment": "13-image-text-convergence",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "training_data": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "images": TRAINING_IMAGES,
            "distinct_captions_per_image": CAPTIONS_PER_IMAGE,
            "presentations": TRAINING_IMAGES * CAPTIONS_PER_IMAGE,
            "batch_order": (
                "seeded caption cycles with unique images and captions per batch"
            ),
        },
        "seeds": list(SEEDS),
        "optimizer_steps": STEPS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "image_shape": [3, 224, 224],
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "loss": {
            "name": "mnr",
            "scale": 20.0,
            "symmetric": True,
            "negative_scope": "global",
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 2e-5,
            "weight_decay": 0.0,
            "warmup_steps": WARMUP_STEPS,
            "schedule": "cosine",
            "gradient_clip_norm": 1.0,
            "precision": "bfloat16-compute-float32-parameters",
        },
        "evaluation": {
            "id": EVALUATION_ID,
            "revision": EVALUATION_REVISION,
            "progress": [0.0, 1.0],
            "directions": ["text-to-image", "image-to-text"],
            "metrics": ["recall@1", "recall@5", "recall@10", "ndcg@10"],
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
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


def _seed_training_files() -> dict[str, Any]:
    base = [
        json.loads(line)
        for line in (DATA / "train.jsonl").read_text().splitlines()
        if line
    ]
    grouped = {
        index: [row for row in base if int(row["caption_index"]) == index]
        for index in range(CAPTIONS_PER_IMAGE)
    }
    if any(len(rows) != TRAINING_IMAGES for rows in grouped.values()):
        raise ValueError("prepared COCO caption cycles do not match the contract")
    seed_files = {}
    for seed in SEEDS:
        ordered = []
        for caption_index, rows in grouped.items():
            ordered.extend(
                _batch_unique_caption_order(
                    rows,
                    batch_size=GLOBAL_BATCH_SIZE,
                    seed=seed + caption_index,
                )
            )
        for start in range(0, len(ordered), GLOBAL_BATCH_SIZE):
            batch = ordered[start : start + GLOBAL_BATCH_SIZE]
            if len({row["image_id"] for row in batch}) != len(batch):
                raise RuntimeError("COCO training batch contains duplicate images")
            if len({row["caption"] for row in batch}) != len(batch):
                raise RuntimeError("COCO training batch contains duplicate captions")
        path = DATA / f"train-seed-{seed}.jsonl"
        _write_jsonl(path, ordered)
        seed_files[str(seed)] = {
            "file": path.name,
            "rows": len(ordered),
            "sha256": _sha256(path),
            "duplicate_images_within_batch": 0,
            "duplicate_captions_within_batch": 0,
        }
    return seed_files


def prepare_data() -> None:
    manifest_path = DATA / "manifest.json"
    if not manifest_path.is_file():
        subprocess.run(
            (
                str(PYTHON),
                "-m",
                "experiments.preflights.image_text",
                "prepare",
                "--output",
                str(DATA),
                "--training-images",
                str(TRAINING_IMAGES),
                "--captions-per-image",
                str(CAPTIONS_PER_IMAGE),
            ),
            cwd=ROOT,
            check=True,
        )
    manifest = json.loads(manifest_path.read_text())
    if (
        int(manifest["training_images"]) != TRAINING_IMAGES
        or int(manifest["captions_per_image"]) != CAPTIONS_PER_IMAGE
    ):
        raise ValueError("prepared image-text data does not match the contract")
    manifest["seed_files"] = _seed_training_files()
    _write_json(manifest_path, manifest)


def prepare_evaluation_data() -> None:
    ensure_bidirectional_flickr_evaluation(DATA)


def worker_command(seed: int, gpu: int) -> list[str]:
    run = OUTPUT / "runs" / f"seed-{seed}"
    return [
        str(PYTHON),
        "-m",
        "experiments.preflights.image_text",
        "worker",
        "--framework",
        "representax",
        "--checkpoint",
        str(CHECKPOINT),
        "--data-directory",
        str(DATA),
        "--training-file",
        f"train-seed-{seed}.jsonl",
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
        "--symmetric",
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
    manifest = json.loads((DATA / "manifest.json").read_text())
    record = manifest["seed_files"][str(seed)]
    training_file = DATA / record["file"]
    if _sha256(training_file) != record["sha256"]:
        raise ValueError(f"training data hash changed: {training_file}")
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
        raise RuntimeError(f"image-text convergence workers failed: {failed}")
    evaluate_all(gpus)


def _evaluation_batches(processor: Any):
    rows = [
        json.loads(line)
        for line in (DATA / "evaluation-image-to-text.jsonl").read_text().splitlines()
        if line
    ]
    collator = ImageTextEvaluationCollator(
        processor=processor,
        root_directory=DATA,
        direction="image-to-text",
    )
    for start in range(0, len(rows), EVALUATION_BATCH_SIZE):
        yield collator(rows[start : start + EVALUATION_BATCH_SIZE])


def _evaluate(model: Any, processor: Any) -> dict[str, Any]:
    from representax.config import PrecisionConfig
    from representax.evaluation import InformationRetrievalEvaluator
    from representax.precision import resolve_precision_policy
    from representax.train.evaluation import EvaluationRunner

    manifest = json.loads((DATA / "manifest.json").read_text())
    direction = manifest["evaluation_directions"]["image-to-text"]
    relevant = {
        int(query): frozenset(int(document) for document in documents)
        for query, documents in direction["relevant_documents"].items()
    }
    runner = EvaluationRunner(
        InformationRetrievalEvaluator(
            relevant_documents=relevant,
            name="flickr30k-image-to-text",
            score_functions=("cosine",),
            main_score_function="cosine",
            accuracy_at_k=(1, 5, 10),
            precision_recall_at_k=(1, 5, 10),
            mrr_at_k=(10,),
            ndcg_at_k=(10,),
            map_at_k=(10,),
        ),
        precision=resolve_precision_policy(PrecisionConfig.bfloat16_mixed()),
    )
    result = runner.run(model, _evaluation_batches(processor))
    return {
        "batches": result.batches,
        "examples": result.examples,
        "duration_seconds": result.duration_seconds,
        "compilation_seconds": result.compilation_seconds,
        "metrics": dict(result.metrics),
    }


def evaluation_worker(seed: int) -> None:
    import jax.numpy as jnp

    from representax import load_inference_bundle
    from representax.models.clip import load_clip

    path = OUTPUT / "runs" / f"seed-{seed}" / "image-to-text-evaluation.json"
    if path.exists():
        raise FileExistsError(f"evaluation already exists: {path}")
    report = json.loads(
        (OUTPUT / "runs" / f"seed-{seed}" / "report.json").read_text()
    )
    initial_model, processor = load_clip(
        CHECKPOINT,
        revision=MODEL_REVISION,
        local_files_only=True,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        rematerialization="none",
    )
    final_model, _ = load_inference_bundle(report["inference_bundle"])
    result = {
        "direction": "image-to-text",
        "dataset": {"id": EVALUATION_ID, "revision": EVALUATION_REVISION},
        "seed": seed,
        "initial": _evaluate(initial_model, processor),
        "final": _evaluate(final_model, processor),
        "git": _git_state(),
    }
    _write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def evaluate_seed(seed: int, gpu: int) -> None:
    prepare_evaluation_data()
    command = [str(PYTHON), __file__, "evaluation-worker", "--seed", str(seed)]
    log = OUTPUT / "runs" / f"seed-{seed}" / "image-to-text-evaluation.log"
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env=_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate_all(gpus: tuple[int, ...]) -> None:
    if len(gpus) != len(SEEDS) or len(set(gpus)) != len(gpus):
        raise ValueError("evaluate-all requires three distinct GPU indices")
    prepare_evaluation_data()
    processes = []
    for seed, gpu in zip(SEEDS, gpus, strict=True):
        command = [
            str(PYTHON),
            __file__,
            "evaluation-worker",
            "--seed",
            str(seed),
        ]
        log = OUTPUT / "runs" / f"seed-{seed}" / "image-to-text-evaluation.log"
        stream = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_environment(gpu),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        processes.append((seed, process, stream))
    failures = []
    for seed, process, stream in processes:
        failures.append((seed, process.wait()))
        stream.close()
    failed = [(seed, code) for seed, code in failures if code]
    if failed:
        raise RuntimeError(f"image-to-text evaluations failed: {failed}")
    aggregate()


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def aggregate() -> None:
    reports = {}
    for seed in SEEDS:
        path = OUTPUT / "runs" / f"seed-{seed}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed report: {path}")
        report = json.loads(path.read_text())
        inverse_path = path.with_name("image-to-text-evaluation.json")
        if inverse_path.is_file():
            report["image_to_text_evaluation"] = json.loads(
                inverse_path.read_text()
            )
        reports[str(seed)] = report
    throughput = [
        float(report["steady_state"]["examples_per_second"])
        for report in reports.values()
    ]
    text_to_image_initial = [
        float(report["initial_evaluation"]["valid/flickr30k/cosine_ndcg@10"])
        for report in reports.values()
    ]
    text_to_image_final = [
        float(report["final_evaluation"]["valid/flickr30k/cosine_ndcg@10"])
        for report in reports.values()
    ]
    results: dict[str, Any] = {
        "steady_state_examples_per_second": _summary(throughput),
        "text_to_image_ndcg@10": {
            "initial": _summary(text_to_image_initial),
            "final": _summary(text_to_image_final),
            "delta": _summary(
                [
                    final - initial
                    for initial, final in zip(
                        text_to_image_initial, text_to_image_final, strict=True
                    )
                ]
            ),
        },
    }
    if all("image_to_text_evaluation" in report for report in reports.values()):
        prefix = "valid/flickr30k-image-to-text/cosine_ndcg@10"
        image_to_text_initial = [
            float(report["image_to_text_evaluation"]["initial"]["metrics"][prefix])
            for report in reports.values()
        ]
        image_to_text_final = [
            float(report["image_to_text_evaluation"]["final"]["metrics"][prefix])
            for report in reports.values()
        ]
        results["image_to_text_ndcg@10"] = {
            "initial": _summary(image_to_text_initial),
            "final": _summary(image_to_text_final),
            "delta": _summary(
                [
                    final - initial
                    for initial, final in zip(
                        image_to_text_initial, image_to_text_final, strict=True
                    )
                ]
            ),
        }
    _write_json(
        OUTPUT / "summary.json",
        {"contract": contract(), "results": results, "runs": reports},
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("contract")
    commands.add_parser("prepare")
    commands.add_parser("prepare-evaluation")
    run = commands.add_parser("run")
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--gpu", type=int, required=True)
    all_runs = commands.add_parser("all")
    all_runs.add_argument("--gpus", type=int, nargs=3, required=True)
    evaluate = commands.add_parser("evaluate-seed")
    evaluate.add_argument("--seed", type=int, choices=SEEDS, required=True)
    evaluate.add_argument("--gpu", type=int, required=True)
    evaluate_all_runs = commands.add_parser("evaluate-all")
    evaluate_all_runs.add_argument("--gpus", type=int, nargs=3, required=True)
    evaluation_worker_parser = commands.add_parser("evaluation-worker")
    evaluation_worker_parser.add_argument(
        "--seed", type=int, choices=SEEDS, required=True
    )
    commands.add_parser("aggregate")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "contract":
        print(json.dumps(contract(), indent=2, sort_keys=True))
    elif arguments.command == "prepare":
        prepare_data()
    elif arguments.command == "prepare-evaluation":
        prepare_evaluation_data()
    elif arguments.command == "run":
        run_seed(arguments.seed, arguments.gpu)
    elif arguments.command == "all":
        run_all(tuple(arguments.gpus))
    elif arguments.command == "evaluate-seed":
        evaluate_seed(arguments.seed, arguments.gpu)
    elif arguments.command == "evaluate-all":
        evaluate_all(tuple(arguments.gpus))
    elif arguments.command == "evaluation-worker":
        evaluation_worker(arguments.seed)
    else:
        aggregate()


if __name__ == "__main__":
    main()
