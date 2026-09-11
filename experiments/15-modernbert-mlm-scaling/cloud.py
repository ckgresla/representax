"""Bounded A100 strong scaling: setup, topology, tuning, then three matched seeds."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT / "experiments/.venv/bin/python"
SEEDS = (7, 42, 773)
DEVICES = (1, 2, 4, 8)
MICROBATCHES = (8, 16, 32)
WARM_UPDATES = 20
WALL_SECONDS = 100 * 60
RUN_LIMIT_SECONDS = 15 * 60
MAX_STEP_CV = 0.10


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def configuration(seed, microbatch, *, tuning=False):
    # Import only in workers with the locked environment, never the bootstrap process.
    from dataclasses import asdict, replace

    from canary import CASES

    return asdict(
        replace(
            CASES["strong-ddp"],
            seed=seed,
            local_microbatch=microbatch,
            steps_per_length=5 if tuning else WARM_UPDATES + 1,
        )
    )


def terminate(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def checked_run(command, *, env, log, timeout):
    if timeout <= 0:
        raise TimeoutError("cloud wall-clock budget exhausted")
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"exit {code}: {log}")
        finally:
            terminate(process)


def row_from_report(report, count, path):
    if report.get("status") != "passed":
        raise ValueError(f"worker did not pass: {path}")
    phase = report["phases"][0]
    observations = phase["observations"]
    warm = [r["seconds"] for r in observations[1:]]
    if len(warm) != WARM_UPDATES or not all(
        r["finite"] and not r["skipped"] for r in observations
    ):
        raise ValueError("expected 20 finite, non-skipped warm updates")
    seconds = statistics.mean(warm)
    return dict(
        seed=report["configuration"]["seed"],
        gpus=count,
        step_seconds=seconds,
        steps_per_second=1 / seconds,
        examples_per_second=phase["global_batch"] / seconds,
        tokens_per_second=report["configuration"]["tokens_per_update"] / seconds,
        step_time_cv=statistics.stdev(warm) / seconds,
        compile_or_load_seconds=phase["compile_seconds"],
        compiled_bytes_per_device=phase["compiled_required_bytes_per_device"],
        first_loss=observations[0]["loss"],
        last_loss=observations[-1]["loss"],
        result=str(path),
    )


def verify_batches(baseline, other):
    left = baseline["phases"][0]["observations"]
    right = other["phases"][0]["observations"]
    keys = (
        "tokens_sha256",
        "corrupted_sha256",
        "labels_sha256",
        "positions_sha256",
        "supervised_tokens",
    )
    for a, b in zip(left, right, strict=True):
        if any(a[k] != b[k] for k in keys):
            raise ValueError("cross-device-count input or masking mismatch")


def has_nvlink(topology):
    return any(
        re.match(r"\s*GPU\d+\s", line) and re.search(r"\bNV\d+\b", line)
        for line in topology.splitlines()
    )


def preflight(output):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    devices = jax.devices()
    if len(devices) != 8 or any(d.platform != "gpu" for d in devices):
        raise RuntimeError("this cloud panel requires eight visible GPUs")
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    if not has_nvlink(topology):
        raise RuntimeError("no NVLink topology reported; inspect before benchmarking")
    # Reduction of a sharded leading dimension exercises real cross-GPU collectives.
    mesh = jax.make_mesh((8,), ("data",), devices=devices)
    sharding = NamedSharding(mesh, P("data", None))
    x = jax.device_put(np.ones((8, 1048576), dtype=np.float32), sharding)
    reduction = jax.jit(
        lambda value: value.sum(axis=0), out_shardings=NamedSharding(mesh, P())
    )
    started = time.monotonic()
    y = reduction(x).block_until_ready()
    cold = time.monotonic() - started
    np.testing.assert_array_equal(np.asarray(y), np.full(1048576, 8, np.float32))
    timings = []
    for _ in range(5):
        started = time.monotonic()
        reduction(x).block_until_ready()
        timings.append(time.monotonic() - started)
    write(
        output / "collective.json",
        dict(
            jax=jax.__version__,
            devices=[str(d) for d in devices],
            cold_seconds=cold,
            warm_seconds=timings,
            passed=True,
            note="Application reduction sanity check, not a link-bandwidth benchmark.",
        ),
    )


def run(output):
    output = output.resolve()
    if not output.is_relative_to(Path("/raid")):
        raise ValueError("map /raid to the instance SSD before running")
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 8 or any("A100" not in name for name in names):
        raise RuntimeError("run this launcher on the dedicated eight-A100 instance")
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        raise RuntimeError(f"GPU processes already running: {active}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + WALL_SECONDS
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cuda,cpu",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_FLAGS": "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_memory_limit_slop_factor=90",
        "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
        "UV_CACHE_DIR": str(output.parent / "uv-cache"),
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": str(ROOT),
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV",
    }
    for key in ("NCCL_PROTO", "NCCL_ALGO", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"):
        env.pop(key, None)
    state = dict(
        status="running",
        started_unix=started,
        deadline_unix=deadline,
        seeds=SEEDS,
        devices=DEVICES,
        launches=[],
        rows=[],
        environment=env,
        note="Timeout stops workloads, NOT instance billing.",
    )
    # Record only relevant environment values, never inherited credentials.
    state["environment"] = {
        k: v
        for k, v in env.items()
        if k.startswith(
            ("JAX_", "XLA_", "NCCL_", "OMP_", "OPENBLAS_", "TOKENIZERS_", "UV_CACHE")
        )
    }
    write(output / "dispatch.json", state)

    def execute(command, name, count=8, limit=RUN_LIMIT_SECONDS):
        state["launches"].append(
            dict(name=name, command=command, gpus=list(range(count)))
        )
        write(output / "dispatch.json", state)
        print(f"START {name}", flush=True)
        checked_run(
            command,
            env={**env, "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(count)))},
            log=output / f"{name}.log",
            timeout=min(limit, deadline - time.time()),
        )
        print(f"DONE {name}", flush=True)

    def worker(seed, microbatch, count, *, tuning=False):
        name = f"tune-micro{microbatch}" if tuning else f"seed-{seed}-{count}gpu"
        command = [
            str(PYTHON),
            str(__file__),
            "worker",
            "--output",
            str(output / name),
            "--seed",
            str(seed),
            "--microbatch",
            str(microbatch),
        ]
        if tuning:
            command.append("--tuning")
        execute(command, name, count)
        return json.loads((output / name / "result.json").read_text()), output / name

    telemetry = None
    try:
        for name, command in {
            "hardware": ["nvidia-smi", "-q"],
            "topology": ["nvidia-smi", "topo", "-m"],
            "p2p-read": ["nvidia-smi", "topo", "-p2p", "r"],
            "p2p-write": ["nvidia-smi", "topo", "-p2p", "w"],
        }.items():
            execute(command, name)
        execute(["bash", str(HERE / "setup.sh")], "setup", limit=20 * 60)
        execute(["uv", "pip", "freeze", "--python", str(PYTHON)], "packages")
        execute(
            [str(PYTHON), str(__file__), "preflight", "--output", str(output)],
            "preflight",
        )
        with (output / "telemetry.csv").open("w") as stream:
            telemetry = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw",
                    "--format=csv",
                    "-l",
                    "5",
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        candidates = []
        for microbatch in MICROBATCHES:
            report, _ = worker(7, microbatch, 1, tuning=True)
            if report.get("status") == "compiled_capacity_exceeds_allocator_limit":
                break
            if report.get("status") != "passed":
                raise RuntimeError(f"tuning failed: {report.get('status')}")
            candidates.append(
                (report["phases"][0]["warm_input_tokens_per_second"], microbatch)
            )
        if not candidates:
            raise RuntimeError("no fitting microbatch")
        _, selected = max(candidates)
        state["selection"] = dict(
            local_microbatch=selected,
            candidates=candidates,
            note="Bounded four-warm-step tuning; fresh baseline below.",
        )
        write(output / "dispatch.json", state)
        for seed in SEEDS:
            baseline = None
            for count in DEVICES:
                report, path = worker(seed, selected, count)
                row = row_from_report(report, count, path / "result.json")
                if baseline is None:
                    baseline = report
                verify_batches(baseline, report)
                reference = baseline["phases"][0]["warm_input_tokens_per_second"]
                row["speedup"] = row["tokens_per_second"] / reference
                row["efficiency"] = row["speedup"] / count
                state["rows"].append(row)
                write(output / "dispatch.json", state)
                if row["step_time_cv"] > MAX_STEP_CV:
                    raise RuntimeError(
                        "warm timing CV >10%; ask user before extending runs"
                    )
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "incomplete"
        state["error"] = repr(error)
        raise
    finally:
        if telemetry is not None:
            terminate(telemetry)
        state["elapsed_seconds"] = time.time() - started
        write(output / "dispatch.json", state)
        aggregate = []
        for count in DEVICES:
            rows = [r for r in state["rows"] if r["gpus"] == count]
            if rows:
                rates = [r["tokens_per_second"] for r in rows]
                aggregate.append(
                    dict(
                        gpus=count,
                        completed_seeds=len(rows),
                        mean_tokens_per_second=statistics.mean(rates),
                        stdev_tokens_per_second=statistics.stdev(rates)
                        if len(rows) > 1
                        else None,
                        mean_paired_speedup=statistics.mean(r["speedup"] for r in rows),
                    )
                )
        write(
            output / "summary.json",
            dict(status=state["status"], rows=state["rows"], aggregate=aggregate),
        )
        print(
            f"{state['status']}: {output}; "
            "terminate the instance after copying results."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name == "worker":
            p.add_argument("--seed", type=int, choices=SEEDS, required=True)
            p.add_argument(
                "--microbatch", type=int, choices=MICROBATCHES, required=True
            )
            p.add_argument("--tuning", action="store_true")
    args = parser.parse_args()
    if args.command == "run":

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        run(args.output)
    elif args.command == "preflight":
        preflight(args.output)
    else:
        from canary import CanaryConfig
        from canary import main as train

        values = configuration(args.seed, args.microbatch, tuning=args.tuning)
        sys.argv = [
            str(HERE / "canary.py"),
            "--case",
            "strong-ddp",
            "--output",
            str(args.output),
        ]
        train(config=CanaryConfig(**values))


if __name__ == "__main__":
    main()
