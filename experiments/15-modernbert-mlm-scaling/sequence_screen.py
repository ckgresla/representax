"""Run matched one/two-GPU sequence-length canaries on GPUs 2 and 0,1."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--screen", choices=("fsdp", "ddp"), default="fsdp")
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
    for length in (128, 512, 1024):
        microbatches = (2,) if args.screen == "ddp" and length == 1024 else (8, 4, 2)
        for microbatch in microbatches:
            case = f"length-{length}-micro{microbatch}"
            if args.screen == "ddp":
                case = f"ddp-{case}"
            processes = []
            handles = []
            reports = []
            try:
                for label, gpus in (("one-gpu", "2"), ("two-gpu", "0,1")):
                    run = output / f"{case}-{label}"
                    command = [
                        sys.executable,
                        "-u",
                        str(Path(__file__).with_name("canary.py")),
                        "--case",
                        case,
                        "--output",
                        str(run),
                    ]
                    handle = (output / f"{run.name}.log").open("w")
                    handles.append(handle)
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        env={**environment, "CUDA_VISIBLE_DEVICES": gpus},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append((process, run))
                    launches.append(
                        {
                            "command": command,
                            "gpus": gpus,
                            "pid": process.pid,
                            "output": str(run),
                        }
                    )
                    (output / "dispatch.json").write_text(
                        json.dumps(launches, indent=2)
                    )
                    print(
                        f"START {run.name}: PID {process.pid}, GPUs {gpus}", flush=True
                    )
                for process, run in processes:
                    code = process.wait()
                    if code:
                        raise RuntimeError(f"{run.name} exited {code}; inspect its log")
                    report = json.loads((run / "result.json").read_text())
                    reports.append(report)
                    print(f"END {run.name}: {report['status']}", flush=True)
            finally:
                for process, _ in processes:
                    if process.poll() is None:
                        process.terminate()
                for process, _ in processes:
                    process.wait()
                for handle in handles:
                    handle.close()
            if all(r["status"] == "passed" for r in reports):
                break
            if any(
                r["status"]
                not in (
                    "passed",
                    "compiled_capacity_exceeds_allocator_limit",
                )
                for r in reports
            ):
                raise RuntimeError("unexpected canary status")
            print(
                f"CAPACITY {length}: retain both results, reduce matched microbatch",
                flush=True,
            )
        else:
            raise RuntimeError(f"no matched fitting microbatch at length {length}")
    print(f"COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
