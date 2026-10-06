"""Matched dense Sentence Transformers forward-performance probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np

ORACLE_VERSION = "5.6.1"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime", choices=("representax", "transformers"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    return parser.parse_args()


def _fingerprint(inputs: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256(b"sentence-transformers-dense-forward-fp32-v1")
    for name in sorted(inputs):
        value = np.ascontiguousarray(inputs[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _measure(
    invoke: Callable[[], Any],
    synchronize: Callable[[Any], None],
    warmups: int,
    iterations: int,
) -> tuple[Any, float, list[float]]:
    started = time.perf_counter()
    output = invoke()
    synchronize(output)
    first_seconds = time.perf_counter() - started
    for _ in range(warmups):
        output = invoke()
        synchronize(output)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = invoke()
        synchronize(output)
        samples.append(time.perf_counter() - started)
    return output, first_seconds, samples


def _representax(
    arguments: argparse.Namespace,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from representax.core import Route
    from representax.integrations import load_sentence_transformer_artifact

    started = time.perf_counter()
    loaded = load_sentence_transformer_artifact(
        arguments.checkpoint,
        local_files_only=True,
        parameter_dtype=jnp.float32,
        compute_dtype=jnp.float32,
    )
    batch = loaded.encoder.make_batch(
        input_ids=jnp.asarray(inputs["input_ids"]),
        attention_mask=jnp.asarray(inputs["attention_mask"]),
    )
    initialization_seconds = time.perf_counter() - started

    @eqx.filter_jit
    def invoke(candidate, value):
        return candidate.encode(value, route=Route.QUERY)

    with jax.default_matmul_precision("highest"):
        started = time.perf_counter()
        compiled = cast(Any, invoke).lower(loaded.encoder, batch).compile()
        compilation_seconds = time.perf_counter() - started
        output, first_seconds, samples = _measure(
            lambda: compiled(loaded.encoder, batch),
            lambda value: value.block_until_ready(),
            arguments.warmups,
            arguments.iterations,
        )
    memory = jax.devices()[0].memory_stats() or {}
    return np.asarray(output, dtype=np.float32), {
        "framework": "representax",
        "framework_version": _distribution_version("representax"),
        "backend_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "initialization_seconds": initialization_seconds,
        "compilation_seconds": compilation_seconds,
        "first_execution_seconds": first_seconds,
        "compile_or_first_execution_seconds": compilation_seconds + first_seconds,
        "steady_state_seconds": samples,
        "allocator_peak_device_bytes": int(
            memory.get("peak_bytes_in_use", memory.get("bytes_in_use", 0))
        ),
    }


def _transformers(
    arguments: argparse.Namespace,
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    import sentence_transformers
    import torch

    from tests.models.upstream import (
        configure_torch_float32_highest,
        transformers_tacet,
    )

    version = importlib.metadata.version("sentence-transformers")
    if version != ORACLE_VERSION:
        raise RuntimeError(
            f"dense performance requires sentence-transformers=={ORACLE_VERSION}; "
            f"found {version}"
        )
    configure_torch_float32_highest()
    started = time.perf_counter()
    with transformers_tacet():
        model = sentence_transformers.SentenceTransformer(
            str(arguments.checkpoint),
            device="cuda",
            local_files_only=True,
        )
    model.eval()
    features = {
        "input_ids": torch.as_tensor(
            inputs["input_ids"], device="cuda", dtype=torch.long
        ),
        "attention_mask": torch.as_tensor(
            inputs["attention_mask"], device="cuda", dtype=torch.long
        ),
    }
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()

    def invoke():
        with torch.inference_mode():
            return model(features)["sentence_embedding"]

    output, first_seconds, samples = _measure(
        invoke,
        lambda _value: torch.cuda.synchronize(),
        arguments.warmups,
        arguments.iterations,
    )
    return output.detach().cpu().numpy().astype(np.float32), {
        "framework": "sentence-transformers",
        "framework_version": version,
        "backend_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "initialization_seconds": initialization_seconds,
        "compilation_seconds": 0.0,
        "first_execution_seconds": first_seconds,
        "compile_or_first_execution_seconds": first_seconds,
        "steady_state_seconds": samples,
        "allocator_peak_device_bytes": int(torch.cuda.max_memory_reserved()),
        "precision_policy": {
            "float32_matmul": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }


def main() -> None:
    arguments = _arguments()
    with np.load(arguments.inputs) as loaded:
        inputs = {name: loaded[name] for name in loaded.files}
    if arguments.runtime == "representax":
        output, report = _representax(arguments, inputs)
    else:
        output, report = _transformers(arguments, inputs)
    samples = report["steady_state_seconds"]
    report.update(
        {
            "workload_fingerprint": _fingerprint(inputs),
            "batch_size": int(inputs["input_ids"].shape[0]),
            "sequence_length": int(inputs["input_ids"].shape[1]),
            "image_count": 0,
            "dtype": "float32",
            "python": platform.python_version(),
            "git_revision": os.environ.get("REPRESENTAX_GIT_REVISION", "unknown"),
            "checkpoint_revision": arguments.checkpoint.name,
            "steady_state_median_seconds": statistics.median(samples),
            "steady_state_examples_per_second": (
                inputs["input_ids"].shape[0] / statistics.median(samples)
            ),
        }
    )
    np.save(arguments.output, output)
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
