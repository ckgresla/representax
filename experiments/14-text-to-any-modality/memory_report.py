"""Summarize XLA buffer assignments; backfill missing dumps outside timing runs."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SWEEP = importlib.import_module("experiments.14-text-to-any-modality.chunk_sweep")


def buffer_memory(path):
    """Count each physical allocation once; aliases are not extra output storage."""
    totals = dict.fromkeys(
        ("state_aliased", "inputs", "outputs", "temporaries", "constants", "other"), 0
    )
    thread_local = 0
    temporary = False
    values = []
    for line in path.read_text().splitlines():
        allocation = re.match(r"allocation \d+: size (\d+), (.*)", line)
        if allocation:
            size, description = int(allocation[1]), allocation[2]
            temporary = "preallocated-temp" in description
            if "thread-local" in description:
                thread_local += size
                continue
            if "parameter " in description:
                category = (
                    "state_aliased" if "maybe-live-out" in description else "inputs"
                )
            elif temporary:
                category = "temporaries"
            elif "constant" in description:
                category = "constants"
            elif "maybe-live-out" in description:
                category = "outputs"
            else:
                category = "other"
            totals[category] += size
        elif temporary:
            value = re.match(r" value: (.*?) \(size=(\d+),offset=(\d+)\): (.*)", line)
            if value:
                values.append(
                    {
                        "value": value[1],
                        "bytes": int(value[2]),
                        "offset": int(value[3]),
                        "shape": value[4],
                    }
                )
    return {
        "path": str(path),
        "bytes": totals,
        "total_bytes": sum(totals.values()),
        "thread_local_bytes_excluded": thread_local,
        "largest_temporary_values": sorted(values, key=lambda v: -v["bytes"])[:10],
        "note": "Temporary values share offsets/lifetimes; do not sum their sizes.",
    }


def assignments(directory):
    return sorted(directory.glob("**/*jit_compiled_step*buffer-assignment.txt"))


def worker(root, index):
    from representax.train import run_job

    modality, chunk = SWEEP.CELLS[index]
    directory = root / f"{modality}-chunk-{chunk}" / "memory-replay"
    paths = json.loads((root / "data/sources.json").read_text())["sources"]
    job, bindings = SWEEP.probe_job(paths, modality, chunk)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "job.json").write_text(job.model_dump_json(indent=2))
    result = run_job(job, directory / "run", mappers=bindings, stop_after=1)
    assert result.completed_iterations == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "backfill", "worker"))
    parser.add_argument("--output", type=Path, default=SWEEP.OUTPUT)
    parser.add_argument("--cell", type=int, choices=range(len(SWEEP.CELLS)))
    args = parser.parse_args()
    root = args.output
    if args.command == "worker":
        if args.cell is None:
            parser.error("worker requires --cell")
        worker(root, args.cell)
        return
    if args.command == "backfill":
        for index, (modality, chunk) in enumerate(SWEEP.CELLS):
            directory = root / f"{modality}-chunk-{chunk}"
            timing = json.loads((directory / "result.json").read_text())
            if timing["status"] != "completed":
                continue
            if assignments(directory):
                continue
            replay = directory / "memory-replay"
            replay.mkdir(parents=True, exist_ok=True)
            print(f"MEMORY {modality} chunk={chunk}", flush=True)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=SWEEP.GPU,
                HF_HOME="/raid/.cache/huggingface",
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
                JAX_ENABLE_COMPILATION_CACHE="false",
                TOKENIZERS_PARALLELISM="false",
                XLA_FLAGS=(
                    os.environ.get("XLA_FLAGS", "")
                    + f" --xla_dump_to={replay / 'hlo'} --xla_dump_hlo_as_text"
                    + " --xla_dump_hlo_module_re=jit_compiled_step"
                ),
            )
            with (replay / "worker.log").open("w") as log:
                subprocess.run(
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
                    check=True,
                )
    report = []
    for modality, chunk in SWEEP.CELLS:
        directory = root / f"{modality}-chunk-{chunk}"
        reports = [buffer_memory(path) for path in assignments(directory)]
        timing = json.loads((directory / "result.json").read_text())
        report.append(
            {
                "modality": modality,
                "chunk": chunk,
                "execution_status": timing["status"],
                "executables": reports,
                "note": None
                if reports
                else "No completed compiler buffer assignment; see worker.log.",
            }
        )
    (root / "memory-summary.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {root / 'memory-summary.json'}", flush=True)


if __name__ == "__main__":
    main()
