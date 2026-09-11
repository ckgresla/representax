"""Capture uncontended warm CUDA timelines for the accepted DDP recipe."""

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
    parser.add_argument(
        "--devices", type=int, nargs="+", choices=(1, 2, 4), default=[1, 2, 4]
    )
    parser.add_argument(
        "--protocol", choices=("Simple",), help="diagnostic NCCL protocol override"
    )
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
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV",
    }
    launches = []
    if args.protocol:
        environment["NCCL_PROTO"] = args.protocol
    for count, gpus in ((1, "0"), (2, "0,1"), (4, "2,3,4,5")):
        if count not in args.devices:
            continue
        trace = output / f"{count}gpu"
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--output",
            str(trace),
            sys.executable,
            "-u",
            str(Path(__file__).with_name("canary.py")),
            "--case",
            "profile-ddp",
            "--profile",
            "--output",
            str(trace),
        ]
        with (output / f"{count}gpu.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=root,
                env={**environment, "CUDA_VISIBLE_DEVICES": gpus},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            launches.append(
                dict(
                    command=command,
                    gpus=gpus,
                    pid=process.pid,
                    nccl_protocol=environment.get("NCCL_PROTO"),
                )
            )
            (output / "dispatch.json").write_text(json.dumps(launches, indent=2))
            print(f"START {count}GPU: {process.pid}", flush=True)
            try:
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
        if code:
            raise RuntimeError(f"{count}GPU profiler exited {code}; inspect log")
        report = json.loads((trace / "result.json").read_text())
        if report.get("status") != "passed":
            raise RuntimeError(f"{count}GPU update failed")
        subprocess.run(
            [
                "nsys",
                "export",
                "--type=sqlite",
                "--output",
                str(trace.with_suffix(".sqlite")),
                str(trace.with_suffix(".nsys-rep")),
            ],
            check=True,
        )
        print(f"DONE {count}GPU", flush=True)


if __name__ == "__main__":
    main()
