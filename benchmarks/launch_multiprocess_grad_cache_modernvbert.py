"""Launch the ModernVBERT GradCache acceptance probe as two JAX processes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-id",
        default="ModernVBERT/modernvbert-embed",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args()


def _coordinator_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"127.0.0.1:{port}"


def main() -> None:
    arguments = _arguments()
    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    coordinator_address = _coordinator_address()
    worker = Path(__file__).with_name("distributed_grad_cache_modernvbert.py")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment.pop(name, None)

    processes: list[subprocess.Popen[str]] = []
    logs = []
    for process_id, local_device_ids in enumerate(("0,1", "2,3")):
        output = arguments.output_directory / f"process-{process_id}.json"
        log_path = arguments.output_directory / f"process-{process_id}.log"
        log = log_path.open("w")
        logs.append(log)
        command = [
            sys.executable,
            str(worker),
            "--checkpoint",
            str(arguments.checkpoint),
            "--checkpoint-id",
            arguments.checkpoint_id,
            "--output",
            str(output),
            "--world-size",
            "4",
            "--process-count",
            "2",
            "--process-id",
            str(process_id),
            "--coordinator-address",
            coordinator_address,
            "--local-device-ids",
            local_device_ids,
            "--global-batch-size",
            str(arguments.global_batch_size),
            "--sequence-length",
            str(arguments.sequence_length),
            "--chunk-size",
            str(arguments.chunk_size),
            "--warmup-steps",
            str(arguments.warmup_steps),
            "--measured-steps",
            str(arguments.measured_steps),
            "--seed",
            str(arguments.seed),
        ]
        processes.append(
            subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    deadline = time.monotonic() + arguments.timeout_seconds
    try:
        while True:
            statuses = [process.poll() for process in processes]
            failed = [status for status in statuses if status not in (None, 0)]
            if failed:
                raise RuntimeError(f"a JAX process failed with status {failed[0]}")
            if all(status is not None for status in statuses):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("multi-process acceptance probe timed out")
            time.sleep(1.0)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        raise
    finally:
        for log in logs:
            log.close()

    return_codes = [process.returncode for process in processes]
    if any(return_code != 0 for return_code in return_codes):
        raise SystemExit(f"JAX process exit statuses: {return_codes}")

    artifacts = [
        json.loads(
            (arguments.output_directory / f"process-{process_id}.json").read_text()
        )
        for process_id in range(2)
    ]
    accepted = all(artifact["status"] == "accepted" for artifact in artifacts)
    summary = {
        "schema_version": "representax-multiprocess-grad-cache-modernvbert-v1",
        "status": "accepted" if accepted else "rejected",
        "physical_hosts": 1,
        "process_count": 2,
        "devices_per_process": 2,
        "world_size": 4,
        "process_artifacts": ["process-0.json", "process-1.json"],
        "differences": [artifact["differences"] for artifact in artifacts],
    }
    summary_path = arguments.output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
