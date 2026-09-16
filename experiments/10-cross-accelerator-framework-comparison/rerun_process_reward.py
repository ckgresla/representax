"""Correct the GPU TRL padding mismatch without replacing historical artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(
    "/raid/representax-paper/10-cross-accelerator-framework-comparison/gpu-rtx4090-process-reward-padding256"
)
ORIGINAL = OUTPUT.parent / "gpu-rtx4090"
ASSETS = Path("/raid/representax-paper-assets")
PYTHON = Path(
    "/raid/representax-paper/checkouts/gpu-sweep-fbe44b1/experiments/.venv/bin/python"
)
PADDING_LENGTH = 256
STEPS = 22
GPU_SEEDS = {2: (7, 773, 2026), 3: (42, 1234)}
DATA_HASHES = {
    "manifest.json": (
        "sha256:223cf7780c6a164fcac61e6e6f8d79a9165914b90d630e620079dd827e437d53"
    ),
    "train.jsonl": (
        "sha256:eaec187faff4003cb38e9408318524efa65ec6e98d9e411ffbd53d0870e50320"
    ),
    "evaluation.jsonl": (
        "sha256:034a7d4945424b106712f64ffef58069bf3ed109e98685d24876e97dfb1c1927"
    ),
}


def campaign():
    spec = importlib.util.spec_from_file_location(
        "campaign", Path(__file__).with_name("run.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs(module):
    data = ASSETS / "process-data"
    for name, expected in DATA_HASHES.items():
        if module._sha256(data / name) != expected:
            raise ValueError(f"changed historical data: {name}")
    lengths = [
        json.loads(line)["token_count"]
        for line in (data / "train.jsonl").read_text().splitlines()
    ]
    expected_tokens = [
        sum(lengths[step * 64 : (step + 1) * 64]) for step in range(STEPS)
    ]
    for seeds in GPU_SEEDS.values():
        for seed in seeds:
            path = ORIGINAL / f"seed-{seed}/process-reward/representax/metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            metrics = [
                row["metrics"]
                for row in rows
                if "perf/tokens" in row.get("metrics", {})
            ]
            if [row["perf/tokens"] for row in metrics] != expected_tokens:
                raise ValueError(f"seed {seed}: historical batch order differs")
            if any(
                row["perf/token_capacity"] != 64 * PADDING_LENGTH for row in metrics
            ):
                raise ValueError(f"seed {seed}: historical execution length differs")


def launch():
    module = campaign()
    validate_inputs(module)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = OUTPUT / "source"
    files = [
        ROOT / "experiments/__init__.py",
        *sorted((ROOT / "experiments/preflights").glob("*.py")),
        *sorted((ROOT / "benchmarks/configs").glob("*.json")),
        Path(__file__).resolve(),
        Path(__file__).with_name("run.py"),
    ]
    hashes = {}
    for path in files:
        destination = source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        hashes[str(path.relative_to(ROOT))] = module._sha256(destination)
    (OUTPUT / "source.patch").write_bytes(module._git_patch())
    record = {
        "created_at": module._utc_now(),
        "git": module._git_state(),
        "source_hashes": hashes,
        "data_hashes": DATA_HASHES,
        "checkpoint_hashes": {
            path.name: module._sha256(path)
            for path in sorted((ASSETS / "qwen3-0.6b").iterdir())
            if path.is_file()
        },
        "padding_length": PADDING_LENGTH,
        "total_steps": STEPS,
        "excluded_steps": [1, 12],
        "global_batch": 64,
        "micro_batch": 2,
        "original_results": str(ORIGINAL),
        "gpu_seeds": GPU_SEEDS,
        "queues": [],
    }
    module._write_json(OUTPUT / "launch.json", record)
    script = source / Path(__file__).relative_to(ROOT)
    for gpu in GPU_SEEDS:
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTHONPATH=str(source),
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            PYTHONUNBUFFERED="1",
            TOKENIZERS_PARALLELISM="false",
        )
        command = [str(PYTHON), str(script), "queue", "--gpu", str(gpu)]
        with (OUTPUT / f"queue-{gpu}.log").open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record["queues"].append({"gpu": gpu, "pid": process.pid, "command": command})
        module._write_json(OUTPUT / "launch.json", record)
    print(json.dumps(record["queues"], indent=2))


def queue(gpu):
    module = campaign()
    for seed in GPU_SEEDS[gpu]:
        output = OUTPUT / f"seed-{seed}/reference"
        output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, str(Path(__file__)), "worker", "--seed", str(seed)]
        record = {
            "status": "running",
            "seed": seed,
            "gpu": gpu,
            "command": command,
            "started_at": module._utc_now(),
            "environment": module._environment_state(),
        }
        module._write_json(output / "run.json", record)
        with (output / "worker.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        record.update(
            exit_code=result.returncode,
            finished_at=module._utc_now(),
            status="complete" if result.returncode == 0 else "failed",
        )
        if result.returncode == 0:
            record = module._materialize_canonical_evidence(output, record)
        module._write_json(output / "run.json", record)
        print(f"seed={seed} status={record['status']}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


def worker(seed):
    from experiments.preflights.process_reward import _trl_worker
    from experiments.preflights.provenance import write_reference_result

    output = OUTPUT / f"seed-{seed}/reference"
    summary = _trl_worker(
        checkpoint=ASSETS / "qwen3-0.6b",
        data_directory=ASSETS / "process-data",
        run_directory=output / "run",
        steps=STEPS,
        seed=seed,
        execution_sequence_length=PADDING_LENGTH,
    )
    if summary["observed_training_shapes"] != [(2, PADDING_LENGTH)]:
        raise ValueError("unexpected optimizer microbatch shapes")
    if summary["steady_state"]["measured_steps"] != 20:
        raise ValueError("expected exactly twenty measured optimizer updates")
    write_reference_result(output / "summary.json", summary, reference="trl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("launch")
    subparsers.add_parser("queue").add_argument(
        "--gpu", type=int, choices=GPU_SEEDS, required=True
    )
    subparsers.add_parser("worker").add_argument(
        "--seed", type=int, choices=(7, 42, 773, 1234, 2026), required=True
    )
    args = parser.parse_args()
    if args.command == "launch":
        launch()
    elif args.command == "queue":
        queue(args.gpu)
    else:
        worker(args.seed)
