"""Measure MPNet encoder and MNR-loss scaling independently of input loading."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = Path("/raid/representax/oracles/all-mpnet-base-v2")


def _timed(call: Callable[[], Any], *, repetitions: int) -> tuple[float, list[float]]:
    started = time.perf_counter()
    result = call()
    _block(result)
    cold_seconds = time.perf_counter() - started
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = call()
        _block(result)
        durations.append(time.perf_counter() - started)
    return cold_seconds, durations


def _block(value: Any) -> None:
    try:
        import jax

        leaves = jax.tree.leaves(value)
    except (ImportError, TypeError):
        leaves = (value,)
    for leaf in leaves:
        blocker = getattr(leaf, "block_until_ready", None)
        if callable(blocker):
            blocker()
        if getattr(leaf, "is_cuda", False):
            import torch

            torch.cuda.synchronize()


def _cuda_profile(call: Callable[[], Any]) -> None:
    runtime = ctypes.CDLL("libcudart.so")
    if runtime.cudaProfilerStart() != 0:
        raise RuntimeError("cudaProfilerStart failed")
    try:
        _block(call())
    finally:
        if runtime.cudaProfilerStop() != 0:
            raise RuntimeError("cudaProfilerStop failed")


def _summary(
    *,
    framework: str,
    mode: str,
    batch_size: int,
    cold_seconds: float,
    durations: list[float],
    memory: dict[str, Any],
) -> dict[str, Any]:
    median_seconds = statistics.median(durations)
    return {
        "framework": framework,
        "mode": mode,
        "batch_size": batch_size,
        "sequence_length": 256,
        "cold_first_call_seconds": cold_seconds,
        "warm_seconds": durations,
        "warm_median_seconds": median_seconds,
        "warm_examples_per_second": batch_size / median_seconds,
        "memory": memory,
    }


def _representax_encoder(arguments: argparse.Namespace) -> dict[str, Any]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from representax.config import PrecisionConfig
    from representax.core import Route
    from representax.models import SentenceEncoder
    from representax.precision import precision_context, resolve_precision_policy
    from representax.train.grad_cache import _rematerialized_encode

    model, _ = SentenceEncoder.load_from_hf(
        arguments.checkpoint,
        local_files_only=True,
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        sequence_length_buckets=(256,),
    )
    if not hasattr(model.backbone, "unroll_layers"):
        raise TypeError("the selected sentence encoder is not native MPNet")
    backbone = replace(model.backbone, unroll_layers=arguments.unroll == "full")
    model = eqx.tree_at(lambda candidate: candidate.backbone, model, backbone)
    vocabulary_size = model.backbone.tower.config.vocab_size
    input_ids = (
        jnp.arange(arguments.batch_size * 256, dtype=jnp.int32).reshape(
            arguments.batch_size, 256
        )
        % (vocabulary_size - 5)
    ) + 5
    attention_mask = jnp.broadcast_to(
        jnp.arange(256) < arguments.valid_sequence_length,
        input_ids.shape,
    )
    input_ids = jnp.where(
        attention_mask,
        input_ids,
        model.backbone.tower.config.pad_token_id,
    )
    batch = model.make_batch(input_ids=input_ids, attention_mask=attention_mask)
    target = jnp.linspace(
        -1.0,
        1.0,
        model.metadata.output_dimension,
        dtype=jnp.float32,
    )
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())

    @eqx.filter_value_and_grad
    def objective(
        candidate: SentenceEncoder,
        inputs: Any,
        key: jax.Array,
    ) -> jax.Array:
        if arguments.rematerialize_encoder:
            chunk_size = arguments.chunk_size or arguments.batch_size
            embedding = _rematerialized_encode(
                candidate,
                inputs,
                route=Route.QUERY,
                batch_size=arguments.batch_size,
                chunk_size=chunk_size,
                key=key,
            )
        else:
            embedding = candidate.encode(inputs, route=Route.QUERY, key=key)
        return jnp.mean(embedding.astype(jnp.float32) * target)

    compiled = eqx.filter_jit(objective)
    key = jax.random.key(7)

    def call() -> Any:
        with precision_context(policy):
            return compiled(model, batch, key)

    cold_seconds, durations = _timed(call, repetitions=arguments.repetitions)
    if arguments.profile_cuda:
        _cuda_profile(call)
    stats = jax.devices()[0].memory_stats() or {}
    memory = {
        name: int(value)
        for name, value in stats.items()
        if name
        in {
            "bytes_in_use",
            "peak_bytes_in_use",
            "peak_pool_bytes",
            "pool_bytes",
        }
    }
    chunk_size = arguments.chunk_size or arguments.batch_size
    return _summary(
        framework="representax",
        mode=(
            "encoder-rematerialized-forward-backward-"
            f"{arguments.unroll}-scan-chunk-{chunk_size}"
            if arguments.rematerialize_encoder
            else f"encoder-forward-backward-{arguments.unroll}-scan"
        ),
        batch_size=arguments.batch_size,
        cold_seconds=cold_seconds,
        durations=durations,
        memory=memory,
    )


def _sentence_transformers_encoder(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(arguments.checkpoint), local_files_only=True)
    model.max_seq_length = 256
    model.cuda().train()
    vocabulary_size = model.tokenizer.vocab_size
    input_ids = (
        torch.arange(arguments.batch_size * 256, device="cuda", dtype=torch.long)
        .reshape(arguments.batch_size, 256)
        .remainder(vocabulary_size - 5)
        + 5
    )
    attention_mask = torch.arange(256, device="cuda") < arguments.valid_sequence_length
    attention_mask = attention_mask.broadcast_to(input_ids.shape)
    input_ids = torch.where(
        attention_mask,
        input_ids,
        torch.as_tensor(model.tokenizer.pad_token_id, device="cuda"),
    )
    features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    target = torch.linspace(
        -1.0,
        1.0,
        model.get_embedding_dimension(),
        device="cuda",
    )

    def objective(
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embedding = model({"input_ids": token_ids, "attention_mask": token_mask})[
                "sentence_embedding"
            ]
            loss = (embedding.float() * target).mean()
        loss.backward()
        return loss

    compiled_objective = torch.compile(objective) if arguments.compile else objective

    def call() -> Any:
        return compiled_objective(features["input_ids"], features["attention_mask"])

    torch.cuda.reset_peak_memory_stats()
    cold_seconds, durations = _timed(call, repetitions=arguments.repetitions)
    if arguments.profile_cuda:
        _cuda_profile(call)
    torch.cuda.synchronize()
    memory = {
        "peak_bytes_in_use": torch.cuda.max_memory_allocated(),
        "peak_pool_bytes": torch.cuda.max_memory_reserved(),
    }
    return _summary(
        framework="sentence-transformers",
        mode=(
            "encoder-forward-backward-inductor"
            if arguments.compile
            else "encoder-forward-backward-eager"
        ),
        batch_size=arguments.batch_size,
        cold_seconds=cold_seconds,
        durations=durations,
        memory=memory,
    )


def _representax_loss(arguments: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from representax.tasks.retrieval import MNRTask, retrieval_batch

    count = arguments.batch_size
    width = 768
    query = jax.random.normal(jax.random.key(1), (count, width), dtype=jnp.float32)
    document = jax.random.normal(jax.random.key(2), (count, width), dtype=jnp.float32)
    query /= jnp.linalg.norm(query, axis=-1, keepdims=True)
    document /= jnp.linalg.norm(document, axis=-1, keepdims=True)
    batch = retrieval_batch(
        query=query,
        document=document,
        positive_mask=jnp.eye(count, dtype=jnp.bool_),
    )
    task = MNRTask(scale=20.0)

    @jax.jit
    @jax.value_and_grad
    def objective(
        query_embeddings: jax.Array, document_embeddings: jax.Array
    ) -> jax.Array:
        return task.loss_from_embeddings(
            query_embeddings,
            document_embeddings,
            batch,
            row_chunk_size=arguments.chunk_size,
        ).loss

    def call() -> Any:
        return objective(query, document)

    cold_seconds, durations = _timed(call, repetitions=arguments.repetitions)
    if arguments.profile_cuda:
        _cuda_profile(call)
    stats = jax.devices()[0].memory_stats() or {}
    memory = {
        name: int(value)
        for name, value in stats.items()
        if name
        in {
            "bytes_in_use",
            "peak_bytes_in_use",
            "peak_pool_bytes",
            "pool_bytes",
        }
    }
    return _summary(
        framework="representax",
        mode=f"mnr-loss-forward-backward-rows-{arguments.chunk_size}",
        batch_size=count,
        cold_seconds=cold_seconds,
        durations=durations,
        memory=memory,
    )


def _representax_gradcache(arguments: argparse.Namespace) -> dict[str, Any]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from representax.config import PrecisionConfig
    from representax.models import SentenceEncoder
    from representax.precision import precision_context, resolve_precision_policy
    from representax.tasks.retrieval import MNRTask, retrieval_batch
    from representax.train import GradCache, build_train_step, init_train_state

    model, _ = SentenceEncoder.load_from_hf(
        arguments.checkpoint,
        local_files_only=True,
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        sequence_length_buckets=(256,),
    )
    count = arguments.batch_size
    vocabulary_size = model.backbone.tower.config.vocab_size
    input_ids = (
        jnp.arange(count * 256, dtype=jnp.int32).reshape(count, 256)
        % (vocabulary_size - 5)
    ) + 5
    attention_mask = jnp.broadcast_to(
        jnp.arange(256) < arguments.valid_sequence_length,
        input_ids.shape,
    )
    input_ids = jnp.where(
        attention_mask,
        input_ids,
        model.backbone.tower.config.pad_token_id,
    )
    query = model.make_batch(input_ids=input_ids, attention_mask=attention_mask)
    batch = retrieval_batch(
        query=query,
        document=query,
        positive_mask=jnp.eye(count, dtype=jnp.bool_),
    )
    task = MNRTask(scale=20.0)
    policy = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    execution = GradCache(
        query_chunk_size=arguments.chunk_size,
        document_chunk_size=arguments.chunk_size,
        loss_row_chunk_size=arguments.chunk_size,
    )
    key_holder = [jax.random.key(7)]
    if arguments.objective_only:

        @eqx.filter_value_and_grad
        def objective(
            candidate: SentenceEncoder,
            inputs: Any,
            key: jax.Array,
        ) -> jax.Array:
            return execution.evaluate(task, candidate, inputs, key=key).loss

        compiled_objective = eqx.filter_jit(objective)

        def call() -> Any:
            key_holder[0], step_key = jax.random.split(key_holder[0])
            with precision_context(policy):
                return compiled_objective(model, batch, step_key)

    else:
        optimizer = optax.adamw(learning_rate=2e-5, weight_decay=0.0)
        state = init_train_state(model, optimizer, precision=policy)
        step = build_train_step(
            task,
            optimizer,
            execution=execution,
            precision=policy,
            donate_state=True,
        )
        state_holder = [state]

        def call() -> Any:
            key_holder[0], step_key = jax.random.split(key_holder[0])
            result = step(state_holder[0], batch, step_key)
            state_holder[0] = result.state
            return result

    cold_seconds, durations = _timed(call, repetitions=arguments.repetitions)
    if arguments.profile_cuda:
        _cuda_profile(call)
    stats = jax.devices()[0].memory_stats() or {}
    memory = {
        name: int(value)
        for name, value in stats.items()
        if name
        in {
            "bytes_in_use",
            "peak_bytes_in_use",
            "peak_pool_bytes",
            "pool_bytes",
        }
    }
    return _summary(
        framework="representax",
        mode=(
            f"gradcache-objective-gradient-chunk-{arguments.chunk_size}"
            if arguments.objective_only
            else f"full-gradcache-update-chunk-{arguments.chunk_size}"
        ),
        batch_size=count,
        cold_seconds=cold_seconds,
        durations=durations,
        memory=memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind",
        choices=(
            "representax-encoder",
            "sentence-transformers-encoder",
            "representax-loss",
            "representax-gradcache",
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--valid-sequence-length", type=int, default=256)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--unroll", choices=("full", "compact"), default="full")
    parser.add_argument("--rematerialize-encoder", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--objective-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not 1 <= arguments.valid_sequence_length <= 256:
        parser.error("--valid-sequence-length must be between 1 and 256")
    if (
        arguments.kind in {"representax-loss", "representax-gradcache"}
        and arguments.chunk_size is None
    ):
        parser.error(f"{arguments.kind} requires --chunk-size")

    if arguments.kind == "representax-encoder":
        result = _representax_encoder(arguments)
    elif arguments.kind == "sentence-transformers-encoder":
        result = _sentence_transformers_encoder(arguments)
    elif arguments.kind == "representax-loss":
        result = _representax_loss(arguments)
    else:
        result = _representax_gradcache(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
