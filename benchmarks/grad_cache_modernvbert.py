"""Isolated ModernVBERT training probe for direct and cached MNR runtimes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import threading
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        choices=("direct", "grad-cache", "sentence-transformers"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float)
    parser.add_argument(
        "--compute-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--repeat-sample", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument(
        "--grad-cache-implementation",
        choices=("rematerialized", "custom_vjp"),
        default="rematerialized",
    )
    parser.add_argument("--updated-text-weights", type=Path)
    parser.add_argument("--text-gradients", type=Path)
    parser.add_argument(
        "--rematerialization",
        choices=("none", "selective", "full"),
        default="full",
        help="Representax text-layer activation policy; ignored by the reference",
    )
    return parser.parse_args()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "source-tree"


class _ProcessMemorySampler:
    """Best-effort NVML process peak, including model and optimizer setup."""

    def __init__(self) -> None:
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        except Exception:
            self._pynvml = None

    def _sample(self) -> None:
        assert self._pynvml is not None
        process_id = os.getpid()
        try:
            while not self._stop.wait(0.01):
                for index in range(self._pynvml.nvmlDeviceGetCount()):
                    handle = self._pynvml.nvmlDeviceGetHandleByIndex(index)
                    processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(
                        handle
                    )
                    for process in processes:
                        if process.pid == process_id:
                            used = int(process.usedGpuMemory)
                            self.peak_bytes = max(self.peak_bytes or 0, used)
        except Exception:
            pass

    def close(self) -> int | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._pynvml is not None:
            with suppress(Exception):
                self._pynvml.nvmlShutdown()
        return self.peak_bytes


def _inputs(arguments: argparse.Namespace, vocabulary_size: int) -> dict[str, Any]:
    generator = np.random.default_rng(arguments.seed)
    generated_batch_size = 1 if arguments.repeat_sample else arguments.batch_size
    shape = (generated_batch_size, arguments.sequence_length)
    query = generator.integers(1, vocabulary_size, size=shape, dtype=np.int32)
    document = generator.integers(1, vocabulary_size, size=shape, dtype=np.int32)
    if arguments.repeat_sample:
        query = np.repeat(query, arguments.batch_size, axis=0)
        document = np.repeat(document, arguments.batch_size, axis=0)
    return {
        "query_input_ids": query,
        "document_input_ids": document,
        "attention_mask": np.ones(
            (arguments.batch_size, arguments.sequence_length), dtype=np.int32
        ),
    }


def _workload_fingerprints(
    arguments: argparse.Namespace,
    arrays: dict[str, Any],
) -> dict[str, str]:
    input_digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        input_digest.update(name.encode())
        input_digest.update(str(value.dtype).encode())
        input_digest.update(json.dumps(value.shape).encode())
        input_digest.update(value.tobytes())
    scientific_spec = {
        "batch_size": arguments.batch_size,
        "checkpoint_revision": arguments.checkpoint.name,
        "objective": "diagonal-cosine-mnr",
        "pooling": "attention-mask-mean-then-l2-normalize",
        "precision": arguments.compute_dtype,
        "repeat_sample": arguments.repeat_sample,
        "scale": 20.0,
        "seed": arguments.seed,
        "sequence_length": arguments.sequence_length,
    }
    spec_payload = json.dumps(scientific_spec, sort_keys=True).encode()
    return {
        "input_sha256": input_digest.hexdigest(),
        "scientific_spec_sha256": hashlib.sha256(spec_payload).hexdigest(),
    }


def _array_tree_bytes(tree: Any) -> int:
    import jax

    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "dtype") and hasattr(leaf, "size")
    )


def _canonical_text_name(name: str) -> str:
    name = name.removeprefix("_orig_mod.")
    if not name.startswith("text_model."):
        name = f"text_model.{name}"
    return f"model.{name}"


def _representax(arguments: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import optax

    from representax.config import PrecisionConfig
    from representax.models.modernvbert import (
        ModernVBERTTextBatch,
        ModernVBERTTextCheckpointAdapter,
    )
    from representax.precision import resolve_precision_policy
    from representax.tasks.retrieval import MNRTask, retrieval_batch
    from representax.train import GradCache, build_train_step, make_train_state

    jax.config.update("jax_default_matmul_precision", "highest")
    if len(jax.devices("gpu")) != 1:
        raise RuntimeError("the probe requires exactly one visible GPU")
    config = json.loads((arguments.checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    arrays = _inputs(arguments, int(text_config["vocab_size"]))
    fingerprints = _workload_fingerprints(arguments, arrays)
    compute_dtype = jnp.dtype(arguments.compute_dtype)
    precision = resolve_precision_policy(
        PrecisionConfig.bfloat16_mixed()
        if arguments.compute_dtype == "bfloat16"
        else PrecisionConfig()
    )

    setup_started = time.perf_counter()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        model = ModernVBERTTextCheckpointAdapter(weight_prefix="").load(
            arguments.checkpoint,
            parameter_dtype=jnp.float32,
            compute_dtype=compute_dtype,
            rematerialization=arguments.rematerialization,
        )
        optimizer = optax.adamw(
            learning_rate=arguments.learning_rate,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.0,
        )
        state = make_train_state(model, optimizer)
    device = jax.devices("gpu")[0]
    state = jax.device_put(state, device)
    query = ModernVBERTTextBatch(
        input_ids=jax.device_put(arrays["query_input_ids"], device),
        attention_mask=jax.device_put(arrays["attention_mask"], device),
    )
    document = ModernVBERTTextBatch(
        input_ids=jax.device_put(arrays["document_input_ids"], device),
        attention_mask=jax.device_put(arrays["attention_mask"], device),
    )
    positive_mask = jax.device_put(np.eye(arguments.batch_size, dtype=np.bool_), device)
    batch = retrieval_batch(
        query=query,
        document=document,
        positive_mask=positive_mask,
    )
    optimizer = optax.adamw(
        learning_rate=arguments.learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    )
    execution = (
        None
        if arguments.runtime == "direct"
        else GradCache(
            query_chunk_size=arguments.chunk_size,
            document_chunk_size=arguments.chunk_size,
            loss_row_chunk_size=arguments.chunk_size,
            implementation=arguments.grad_cache_implementation,
        )
    )
    step = build_train_step(
        MNRTask(scale=20.0),
        optimizer,
        max_grad_norm=arguments.max_gradient_norm,
        execution=execution,
        donate_state=True,
        precision=precision,
    )
    jax.block_until_ready(state)
    resident_bytes = {
        "model_parameters": _array_tree_bytes(state.model),
        "optimizer_state": _array_tree_bytes(state.optimizer_state),
        "training_state": _array_tree_bytes(state),
        "batch": _array_tree_bytes(batch),
    }
    setup_seconds = time.perf_counter() - setup_started

    key = jax.device_put(jax.random.key(arguments.seed), device)
    first_started = time.perf_counter()
    result = step(state, batch, key)
    result.metrics.loss.block_until_ready()
    compile_plus_first_seconds = time.perf_counter() - first_started
    losses = [float(result.metrics.loss)]
    state = result.state
    for index in range(arguments.warmup_steps):
        result = step(state, batch, jax.random.fold_in(key, index + 1))
        result.metrics.loss.block_until_ready()
        state = result.state
        losses.append(float(result.metrics.loss))
    samples = []
    for index in range(arguments.measured_steps):
        started = time.perf_counter()
        result = step(
            state,
            batch,
            jax.random.fold_in(key, arguments.warmup_steps + index + 1),
        )
        result.metrics.loss.block_until_ready()
        samples.append(time.perf_counter() - started)
        state = result.state
        losses.append(float(result.metrics.loss))
    memory = device.memory_stats() or {}
    if arguments.text_gradients is not None:
        if arguments.warmup_steps != 0 or arguments.measured_steps != 0:
            raise ValueError("gradient export requires exactly one optimizer update")
        from safetensors.numpy import save_file

        adam_state = state.optimizer_state[0]
        clipped_gradients = jax.tree.map(
            lambda value: value / jnp.asarray(0.1, dtype=value.dtype),
            adam_state.mu,
        )
        gradients = ModernVBERTTextCheckpointAdapter().state_dict(clipped_gradients)
        arguments.text_gradients.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: np.asarray(jax.device_get(value))
                for name, value in gradients.items()
            },
            arguments.text_gradients,
        )
    if arguments.updated_text_weights is not None:
        from safetensors.numpy import save_file

        updated = ModernVBERTTextCheckpointAdapter().state_dict(state.model)
        arguments.updated_text_weights.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                name: np.asarray(jax.device_get(value))
                for name, value in updated.items()
            },
            arguments.updated_text_weights,
        )
    return {
        "framework": "representax",
        "framework_version": _version("representax"),
        "backend": "jax",
        "backend_version": jax.__version__,
        "device": str(device),
        "setup_seconds": setup_seconds,
        "compile_plus_first_seconds": compile_plus_first_seconds,
        "steady_state_seconds": samples,
        "losses": losses,
        "gradient_global_norm": float(result.metrics.gradient_global_norm),
        "resident_bytes": resident_bytes,
        "allocator_peak_device_bytes": int(
            memory.get("peak_bytes_in_use", memory.get("bytes_in_use", 0))
        ),
        "precision_policy": {
            "parameters": "float32",
            "compute": arguments.compute_dtype,
            "objective": "float32",
            "float32_matmul": "highest",
            "xla_python_client_preallocate": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE"
            ),
        },
        "workload_fingerprints": fingerprints,
    }


def _sentence_transformers(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import transformers
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )
    from transformers import AutoModel

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError("the probe requires exactly one visible GPU")
    device = torch.device("cuda:0")
    config = json.loads((arguments.checkpoint / "config.json").read_text())
    text_config = config.get("text_config", config)
    arrays = _inputs(arguments, int(text_config["vocab_size"]))
    fingerprints = _workload_fingerprints(arguments, arrays)

    class Encoder(torch.nn.Module):
        def __init__(self, base: torch.nn.Module) -> None:
            super().__init__()
            self.base = base

        def __getitem__(self, index: int) -> torch.nn.Module:
            if index != 0:
                raise IndexError(index)
            return self.base

        def forward(self, features: dict[str, torch.Tensor]):
            hidden = self.base(
                input_ids=features["input_ids"],
                attention_mask=features["attention_mask"],
            ).last_hidden_state
            mask = features["attention_mask"].unsqueeze(-1).bool()
            pooled = torch.where(mask, hidden.float(), 0.0).sum(1)
            pooled = pooled / mask.sum(1).clamp_min(1)
            return {"sentence_embedding": functional.normalize(pooled, p=2, dim=-1)}

    setup_started = time.perf_counter()
    transformers.utils.logging.disable_progress_bar()
    base = AutoModel.from_pretrained(
        arguments.checkpoint,
        dtype=torch.float32,
        local_files_only=True,
    ).to(device)
    if arguments.torch_compile:
        base = torch.compile(base, backend="inductor")
    model = Encoder(base)
    model.train()
    loss_function = CachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=arguments.chunk_size,
        show_progress_bar=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    features = [
        {
            "input_ids": torch.as_tensor(
                arrays[f"{role}_input_ids"], device=device, dtype=torch.long
            ),
            "attention_mask": torch.as_tensor(
                arrays["attention_mask"], device=device, dtype=torch.long
            ),
        }
        for role in ("query", "document")
    ]
    labels = torch.empty(arguments.batch_size, device=device)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    torch.cuda.reset_peak_memory_stats()

    def update() -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=arguments.compute_dtype == "bfloat16",
        ):
            loss = loss_function(features, labels)
        loss.backward()
        norm = torch.nn.utils.get_total_norm(
            [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
        )
        if arguments.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                arguments.max_gradient_norm,
            )
        optimizer.step()
        torch.cuda.synchronize()
        return float(loss.detach()), float(norm.detach())

    first_started = time.perf_counter()
    first_loss, gradient_norm = update()
    compile_plus_first_seconds = time.perf_counter() - first_started
    losses = [first_loss]
    for _ in range(arguments.warmup_steps):
        loss, gradient_norm = update()
        losses.append(loss)
    samples = []
    for _ in range(arguments.measured_steps):
        started = time.perf_counter()
        loss, gradient_norm = update()
        samples.append(time.perf_counter() - started)
        losses.append(loss)
    resident_bytes = {
        "model_parameters": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ),
        "optimizer_state": sum(
            value.numel() * value.element_size()
            for state in optimizer.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor)
        ),
        "gradients": sum(
            parameter.grad.numel() * parameter.grad.element_size()
            for parameter in model.parameters()
            if parameter.grad is not None
        ),
        "batch": sum(
            value.numel() * value.element_size()
            for item in features
            for value in item.values()
        ),
    }
    if arguments.updated_text_weights is not None:
        from safetensors.torch import save_file

        text_names = {
            _canonical_text_name(name): value.detach().cpu().contiguous()
            for name, value in base.state_dict().items()
        }
        arguments.updated_text_weights.parent.mkdir(parents=True, exist_ok=True)
        save_file(text_names, arguments.updated_text_weights)
    if arguments.text_gradients is not None:
        if arguments.warmup_steps != 0 or arguments.measured_steps != 0:
            raise ValueError("gradient export requires exactly one optimizer update")
        from safetensors.torch import save_file

        text_gradients = {
            _canonical_text_name(name): parameter.grad.detach().cpu().contiguous()
            for name, parameter in base.named_parameters()
            if parameter.grad is not None
        }
        arguments.text_gradients.parent.mkdir(parents=True, exist_ok=True)
        save_file(text_gradients, arguments.text_gradients)
    return {
        "framework": "sentence-transformers",
        "framework_version": _version("sentence-transformers"),
        "backend": "torch",
        "backend_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": torch.cuda.get_device_name(device),
        "setup_seconds": setup_seconds,
        "compile_plus_first_seconds": compile_plus_first_seconds,
        "steady_state_seconds": samples,
        "losses": losses,
        "gradient_global_norm": gradient_norm,
        "resident_bytes": resident_bytes,
        "allocator_peak_device_bytes": int(torch.cuda.max_memory_allocated(device)),
        "allocator_peak_reserved_device_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
        "precision_policy": {
            "parameters": "float32",
            "compute": arguments.compute_dtype,
            "objective": "float32",
            "float32_matmul": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "workload_fingerprints": fingerprints,
    }


def main() -> None:
    arguments = _arguments()
    if arguments.batch_size <= 0 or arguments.sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    if arguments.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if arguments.runtime != "sentence-transformers":
        # Process-memory measurements are meaningless when JAX eagerly reserves
        # nearly the entire visible GPU. The allocator peak remains reported too.
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    sampler = _ProcessMemorySampler()
    started = time.perf_counter()
    try:
        result = (
            _sentence_transformers(arguments)
            if arguments.runtime == "sentence-transformers"
            else _representax(arguments)
        )
        result["status"] = "completed"
        result["oom"] = False
    except Exception as error:
        rendered = traceback.format_exc()
        normalized = rendered.lower()
        result = {
            "status": "failed",
            "oom": any(
                marker in normalized
                for marker in ("out of memory", "resource_exhausted", "oom")
            ),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": rendered,
        }
    result.update(
        {
            "schema_version": "representax-grad-cache-modernvbert-probe-v1",
            "runtime": arguments.runtime,
            "checkpoint": str(arguments.checkpoint),
            "checkpoint_revision": arguments.checkpoint.name,
            "batch_size": arguments.batch_size,
            "sequence_length": arguments.sequence_length,
            "chunk_size": arguments.chunk_size,
            "warmup_steps": arguments.warmup_steps,
            "measured_steps": arguments.measured_steps,
            "seed": arguments.seed,
            "compute_dtype": arguments.compute_dtype,
            "repeat_sample": arguments.repeat_sample,
            "torch_compile": arguments.torch_compile,
            "grad_cache_implementation": arguments.grad_cache_implementation,
            "learning_rate": arguments.learning_rate,
            "max_gradient_norm": arguments.max_gradient_norm,
            "optimizer_updates": (
                1 + arguments.warmup_steps + arguments.measured_steps
            ),
            "updated_text_weights": (
                None
                if arguments.updated_text_weights is None
                else str(arguments.updated_text_weights)
            ),
            "jax_enable_compilation_cache": os.environ.get(
                "JAX_ENABLE_COMPILATION_CACHE"
            ),
            "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
            "rematerialization": (
                arguments.rematerialization
                if arguments.runtime != "sentence-transformers"
                else "upstream-managed"
            ),
            "wall_seconds": time.perf_counter() - started,
            "process_peak_device_bytes": sampler.close(),
            "python": platform.python_version(),
        }
    )
    if result["status"] == "completed" and result["steady_state_seconds"]:
        samples = result["steady_state_seconds"]
        median = statistics.median(samples)
        result["steady_state_median_seconds"] = median
        result["examples_per_second"] = arguments.batch_size / median
        result["tokens_per_second"] = (
            2 * arguments.batch_size * arguments.sequence_length / median
        )
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "completed" and not result["oom"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
