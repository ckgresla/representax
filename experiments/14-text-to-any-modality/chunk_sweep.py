"""One-GPU real-media GradCache capacity sweep; settings live in this file."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN = importlib.import_module("experiments.14-text-to-any-modality.run")
DATA = importlib.import_module(RUN.DATA_MODULE)
CHUNKS = (1, 2, 4, 8, 16, 32)
MODALITIES = ("text", "image", "audio", "video")
CELLS = tuple((modality, chunk) for modality in MODALITIES for chunk in CHUNKS)
OUTPUT = RUN.OUTPUT / "chunk-sweep"
GLOBAL_BATCH = 32
UPDATES = 6
GPU = "0"


def probe_job(paths, modality, chunk):
    """Only modality and replay size vary across the enumerated probe cells."""
    from representax.config import JobConfig

    if (modality, chunk) not in CELLS:
        raise ValueError("cell is not in the fixed sweep")
    base, bindings = RUN.integration_job(paths)
    config = base.model_dump(mode="json")
    config["name"] = f"paper-14-chunk-{modality}-{chunk}"
    config["model"]["parameters"].update(
        sequence_length_buckets=[128, 512, 1024],
        patch_count_buckets=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
        audio_chunk_count_buckets=[8],
        audio_token_count_buckets=[256],
        image_min_pixels=256 * 256,
        image_max_pixels=256 * 256,
        video_min_pixels=224 * 224,
        video_max_pixels=224 * 224,
    )
    distribution = config["data"]["distribution"]
    distribution["sources"] = [
        s for s in distribution["sources"] if s["name"] == modality
    ]
    distribution["weights"] = [1.0]
    config["training"].update(global_batch_size=GLOBAL_BATCH, max_steps=UPDATES)
    config["training"]["batch"]["micro_batch_size"] = GLOBAL_BATCH
    config["training"]["grad_cache"]["micro_batch_size"] = chunk
    config["checkpointing"].update(every=UPDATES + 1, save_final=False)
    bindings[f"{RUN.DATA_MODULE}.map_audio"] = partial(DATA.map_audio, seconds=10.0)
    bindings[f"{RUN.DATA_MODULE}.map_video"] = partial(DATA.map_video, frames=8)
    return JobConfig.model_validate(config), bindings


def summarize(directory):
    metrics = directory / "run/metrics.jsonl"
    rows = (
        [json.loads(line) for line in metrics.read_text().splitlines()]
        if metrics.exists()
        else []
    )
    steps = [r["metrics"] for r in rows if r.get("event") == "training_step"]
    warm = [
        r
        for r in steps
        if "perf/step_seconds" in r
        and "perf/compilation_and_first_step_seconds" not in r
    ]
    seconds = [r["perf/step_seconds"] for r in warm]
    return {
        "completed_updates": len(steps),
        "all_updates_finite": all(r.get("train/numeric_finite") is True for r in steps),
        "skipped_updates": sum(bool(r.get("train/skipped_update")) for r in steps),
        "warm_observations": len(warm),
        "warm_seconds": seconds,
        "warm_examples_per_second": GLOBAL_BATCH * len(seconds) / sum(seconds)
        if seconds
        else None,
        "median_step_seconds": statistics.median(seconds) if seconds else None,
        "cold_compile_and_first_seconds": sum(
            r.get("perf/compilation_and_first_step_seconds", 0.0) for r in steps
        ),
        "losses": [r.get("train/loss") for r in steps],
        "warm_data_wait_seconds": [r.get("perf/data_wait_seconds") for r in warm],
    }


def worker(root, index):
    import jax

    from representax.train import run_job

    if len(jax.devices()) != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("this sweep requires exactly one visible GPU")
    modality, chunk = CELLS[index]
    directory = root / f"{modality}-chunk-{chunk}"
    directory.mkdir(parents=True, exist_ok=True)
    paths = json.loads((root / "data/sources.json").read_text())["sources"]
    job, bindings = probe_job(paths, modality, chunk)
    (directory / "job.json").write_text(job.model_dump_json(indent=2))
    started = time.perf_counter()
    try:
        result = run_job(job, directory / "run", mappers=bindings)
        status = "completed" if result.completed_iterations == UPDATES else "incomplete"
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = (
            "oom"
            if "out of memory" in error.lower() or "RESOURCE_EXHAUSTED" in error
            else "failed"
        )
        traceback.print_exc()
    report = {
        "modality": modality,
        "chunk": chunk,
        "status": status,
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "jax_memory_stats": jax.devices()[0].memory_stats(),
        **summarize(directory),
    }
    if status == "completed" and (
        not report["all_updates_finite"] or report["skipped_updates"]
    ):
        report["status"] = "invalid_updates"
    (directory / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--cell",
        type=int,
        choices=range(len(CELLS)),
        help="internal worker index into the fixed config matrix",
    )
    args = parser.parse_args()
    root = args.output
    if args.command == "prepare":
        DATA.integration_sources(
            root / "data", assets=RUN.ASSETS, sample_count=GLOBAL_BATCH
        )
        return
    if args.command == "worker":
        if args.cell is None:
            parser.error("worker requires --cell")
        worker(root, args.cell)
        return
    root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "gpu": GPU,
                "global_batch": GLOBAL_BATCH,
                "updates": UPDATES,
                "cells": CELLS,
                "purpose": "capacity/timing probe, text-only LoRA, not convergence or full-finetuning acceptance",
            },
            indent=2,
        )
    )
    reports = []
    for index, (modality, chunk) in enumerate(CELLS):
        directory = root / f"{modality}-chunk-{chunk}"
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "result.json"
        if report_path.exists():
            raise FileExistsError(f"refusing to overwrite existing probe: {directory}")
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=GPU,
            HF_HOME="/raid/.cache/huggingface",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
            JAX_COMPILATION_CACHE_DIR=str(root / "jax-cache"),
            TOKENIZERS_PARALLELISM="false",
        )
        print(f"START {modality} chunk={chunk}", flush=True)
        with (directory / "worker.log").open("w") as log:
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).resolve()),
                        "worker",
                        "--output",
                        str(root),
                        "--cell",
                        str(index),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=900,
                )
                failure = f"worker exited {result.returncode}"
            except subprocess.TimeoutExpired:
                failure = "worker exceeded 900 seconds"
        if report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = {
                "modality": modality,
                "chunk": chunk,
                "status": "failed",
                "error": failure,
                **summarize(directory),
            }
            report_path.write_text(json.dumps(report, indent=2))
        reports.append(report)
        (root / "summary.json").write_text(json.dumps(reports, indent=2))
        print(
            f"END {modality} chunk={chunk}: {report['status']}, {report.get('warm_examples_per_second')} examples/s",
            flush=True,
        )


if __name__ == "__main__":
    main()
