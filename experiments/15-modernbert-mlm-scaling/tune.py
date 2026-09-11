"""Tune one GPU at fixed work, then run matched two/four-GPU DDP controls."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from report_scaling import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/raid")):
        parser.error("artifacts must be under /raid")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    environment = {
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
    processes = []

    def launch(name, microbatch, gpus):
        run = output / name
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("canary.py")),
            "--case",
            f"tuned-ddp-micro{microbatch}",
            "--output",
            str(run),
        ]
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env={**environment, "CUDA_VISIBLE_DEVICES": gpus},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        processes.append(process)
        launches.append(
            dict(command=command, gpus=gpus, pid=process.pid, output=str(run))
        )
        (output / "dispatch.json").write_text(json.dumps(launches, indent=2))
        print(f"START {name}: {process.pid}", flush=True)
        return process, run

    def finish(worker):
        process, run = worker
        if process.wait():
            raise RuntimeError(f"{run}: worker failed; inspect retained log")
        report = json.loads((run / "result.json").read_text())
        print(f"END {run.name}: {report['status']}", flush=True)
        return report

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
            candidates = []
            # One compiler at a time gives an uncontended single-GPU baseline.
            for microbatch in (8, 16, 32, 64):
                report = finish(launch(f"micro{microbatch}", microbatch, "0"))
                if report["status"] == "compiled_capacity_exceeds_allocator_limit":
                    break
                if report["status"] != "passed":
                    raise RuntimeError(f"unexpected result: {report['status']}")
                candidates.append(report)
            if not candidates:
                raise RuntimeError("no physical microbatch passed")
            best = max(
                candidates, key=lambda r: r["phases"][0]["warm_input_tokens_per_second"]
            )
            microbatch = best["configuration"]["local_microbatch"]
            (output / "selection.json").write_text(
                json.dumps(
                    {
                        "microbatch": microbatch,
                        "criterion": (
                            "highest one-GPU warm tokens/s among fitting tested cases"
                        ),
                        "candidates": [
                            {
                                "microbatch": r["configuration"]["local_microbatch"],
                                "tokens_per_second": r["phases"][0][
                                    "warm_input_tokens_per_second"
                                ],
                            }
                            for r in candidates
                        ],
                    },
                    indent=2,
                )
            )
            (output / "1gpu").symlink_to(f"micro{microbatch}", target_is_directory=True)
            workers = [
                launch("2gpu", microbatch, "0,1"),
                launch("4gpu", microbatch, "2,3,4,5"),
            ]
            for worker in workers:
                report = finish(worker)
                if report["status"] != "passed":
                    raise RuntimeError(f"scaling control failed: {report['status']}")
            summarize(output)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
            monitor.terminate()
            monitor.wait()


if __name__ == "__main__":
    main()
