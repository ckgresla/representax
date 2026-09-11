"""Seed-7 DDP scaling: paired 1/2-GPU workers, then a nonoverlapping 4-GPU run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("weak", "strong"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/raid")):
        parser.error("artifacts must be under /raid")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cuda,cpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "JAX_COMPILATION_CACHE_DIR": str(output.parent / "jax-cache"),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(root),
    }
    launches = []
    results = []
    with (output / "telemetry.csv").open("w") as telemetry:
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw",
                "--format=csv",
                "-l",
                "5",
            ],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
        )
        try:
            for wave in (((1, "2"), (2, "0,1")), ((4, "0,1,2,3"),)):
                workers = []
                try:
                    for count, gpus in wave:
                        case = (
                            f"weak-ddp-{count}gpu"
                            if args.kind == "weak"
                            else "strong-ddp"
                        )
                        run = output / f"{count}gpu"
                        command = [
                            sys.executable,
                            "-u",
                            str(Path(__file__).with_name("canary.py")),
                            "--case",
                            case,
                            "--output",
                            str(run),
                        ]
                        with (output / f"{count}gpu.log").open("w") as log:
                            process = subprocess.Popen(
                                command,
                                cwd=root,
                                env={**env, "CUDA_VISIBLE_DEVICES": gpus},
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        workers.append((process, run))
                        launches.append(
                            dict(
                                command=command,
                                gpus=gpus,
                                pid=process.pid,
                                output=str(run),
                            )
                        )
                        (output / "dispatch.json").write_text(
                            json.dumps(launches, indent=2)
                        )
                        print(
                            f"START {args.kind} {count}GPU: {process.pid}", flush=True
                        )
                    for process, run in workers:
                        code = process.wait()
                        if code:
                            raise RuntimeError(f"{run}: exit {code}; inspect log")
                        report = json.loads((run / "result.json").read_text())
                        results.append(report)
                        (output / "summary.json").write_text(
                            json.dumps(results, indent=2)
                        )
                        print(f"END {run}: {report['status']}", flush=True)
                        if report["status"] != "passed":
                            raise RuntimeError(f"{run}: {report['status']}")
                finally:
                    for process, _ in workers:
                        if process.poll() is None:
                            process.terminate()
                    for process, _ in workers:
                        process.wait()
        finally:
            monitor.terminate()
            monitor.wait()


if __name__ == "__main__":
    main()
