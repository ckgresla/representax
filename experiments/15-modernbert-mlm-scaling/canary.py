"""Real-text, fixed-work MLM readiness check; not the final scaling campaign."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

from representax.config import PrecisionConfig
from representax.models.modernvbert import (
    ModernBERTMaskedLM,
    ModernVBERTTextBatch,
    ModernVBERTTextConfig,
)
from representax.precision import resolve_precision_policy
from representax.tasks.masked_language_modeling import (
    MaskedLanguageModelTask,
    mask_tokens,
    masked_language_model_batch,
)
from representax.train import ShardingPlan, build_train_step, init_train_state


@dataclass(frozen=True)
class CanaryConfig:
    seed: int = 7
    lengths: tuple[int, ...] = (128, 512)
    steps_per_length: int = 3
    tokens_per_update: int = 32768
    local_microbatch: int = 8
    sharding: Literal["fsdp", "ddp"] = "fsdp"
    repeat_corpus: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    dataset: str = "Salesforce/wikitext"
    dataset_revision: str = "b08601e04326c79dfdd32d625aee71d232d685c3"
    dataset_file: str = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
    tokenizer: str = "google-bert/bert-base-uncased"
    tokenizer_revision: str = "86b5e0934494bd15c9632b12f734a8a67f723594"
    cache: str = "/raid/representax-paper-assets/mlm-hf-cache"


CONFIG = CanaryConfig()
CASES = {
    "baseline": CONFIG,
    "large-batch-micro8": replace(
        CONFIG,
        lengths=(128,),
        tokens_per_update=131072,
        steps_per_length=4,
    ),
    "large-batch-micro32": replace(
        CONFIG,
        lengths=(128,),
        tokens_per_update=131072,
        steps_per_length=4,
        local_microbatch=32,
    ),
    "large-batch-micro64": replace(
        CONFIG,
        lengths=(128,),
        tokens_per_update=131072,
        steps_per_length=4,
        local_microbatch=64,
    ),
    **{
        f"length-{length}-micro{microbatch}": replace(
            CONFIG,
            lengths=(length,),
            tokens_per_update=131072,
            steps_per_length=4,
            local_microbatch=microbatch,
        )
        for length in (128, 512, 1024)
        for microbatch in (2, 4, 8)
    },
    "length-1024-micro16": replace(
        CONFIG,
        lengths=(1024,),
        tokens_per_update=131072,
        steps_per_length=4,
        local_microbatch=16,
    ),
}
CASES.update(
    {
        f"ddp-length-{length}-micro{microbatch}": replace(
            CASES[f"length-{length}-micro{microbatch}"], sharding="ddp"
        )
        for length in (128, 512, 1024)
        for microbatch in (2, 4, 8)
    }
)
MODEL = ModernVBERTTextConfig(
    vocab_size=32768,
    hidden_size=1280,
    intermediate_size=2048,
    num_hidden_layers=32,
    num_attention_heads=20,
    layer_types=tuple(
        "full_attention" if i % 3 == 0 else "sliding_attention" for i in range(32)
    ),
    local_attention=128,
    full_attention_rope_theta=160000.0,
    sliding_attention_rope_theta=10000.0,
    norm_epsilon=1e-5,
    max_position_embeddings=1024,
)

CASES.update(
    {
        **{
            f"weak-ddp-{devices}gpu": replace(
                CONFIG,
                lengths=(512,),
                tokens_per_update=512 * 8 * devices,
                steps_per_length=21,
                sharding="ddp",
            )
            for devices in (1, 2, 4)
        },
        "strong-ddp": replace(
            CONFIG,
            lengths=(512,),
            tokens_per_update=131072,
            steps_per_length=21,
            sharding="ddp",
            repeat_corpus=True,
        ),
    }
)


CASES.update(
    {
        f"tuned-ddp-micro{microbatch}": replace(
            CASES["strong-ddp"], local_microbatch=microbatch
        )
        for microbatch in (8, 16, 32, 64)
    }
)


CASES["profile-ddp"] = replace(CASES["strong-ddp"], steps_per_length=6)


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


def collective_summary(hlo: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(hlo.splitlines(), start=1):
        if not re.search(r"\ball-reduce(?:-start)?\(", line):
            continue
        shapes = re.findall(
            r"(?:bf16|f16|f32|f64)\[([0-9,]*)\]", line.split(" all-reduce")[0]
        )
        elements = [int(np.prod([int(d) for d in s.split(",") if d])) for s in shapes]
        rows.append(
            {
                "line": number,
                "in_loop": "while/body" in line,
                "largest_float_array_elements": max(elements, default=0),
                "instruction": line.strip(),
            }
        )
    return rows


def prepare_tokens(config=CONFIG):
    path = hf_hub_download(
        config.dataset,
        config.dataset_file,
        repo_type="dataset",
        revision=config.dataset_revision,
        cache_dir=config.cache,
    )
    tokenizer_path = snapshot_download(
        config.tokenizer,
        revision=config.tokenizer_revision,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "config.json",
        ],
        cache_dir=config.cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer is None:
        raise RuntimeError("tokenizer loading failed")
    rows = [text for text in pq.read_table(path)["text"].to_pylist() if text.strip()]
    np.random.default_rng(config.seed).shuffle(rows)
    encoded = tokenizer(rows, add_special_tokens=False, truncation=False)["input_ids"]
    # Concatenated real-text blocks with separator boundaries, not repeated samples.
    tokens = np.asarray(
        [token for row in encoded for token in [*row, tokenizer.sep_token_id]],
        dtype=np.int32,
    )
    required = len(config.lengths) * config.steps_per_length * config.tokens_per_update
    if (len(tokens) < required and not config.repeat_corpus) or len(
        tokenizer
    ) > MODEL.vocab_size:
        raise ValueError("insufficient corpus tokens or model vocabulary")
    provenance = {
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "nonempty_training_rows": len(rows),
        "corpus_tokens": len(tokens),
        "token_stream_sha256": digest(tokens),
        "tokenizer_vocab_size": len(tokenizer),
        "model_vocab_size": MODEL.vocab_size,
        "vocabulary_note": (
            "30522 tokenizer IDs; 32768 output rows retain the capacity-control model"
        ),
        "packing": (
            "seed-shuffled nonempty train rows, SEP between rows, "
            "contiguous fixed-length blocks"
        ),
        "masking": (
            "15% fixed-count eligible tokens; 80/10/10; "
            "random replacements use tokenizer vocabulary"
        ),
    }
    provenance["requested_tokens"] = required
    provenance["corpus_presentations"] = required / len(tokens)
    if config.repeat_corpus and required > len(tokens):
        tokens = np.tile(tokens, (required + len(tokens) - 1) // len(tokens))[:required]
    provenance["consumed_token_stream_sha256"] = digest(tokens[:required])
    return tokens, tokenizer, provenance


def host_batch(tokens, tokenizer, length, iteration, config=CONFIG):
    start = iteration * config.tokens_per_update
    ids = tokens[start : start + config.tokens_per_update].reshape(-1, length)
    corrupted, positions, labels = mask_tokens(
        ids,
        attention_mask=np.ones_like(ids, dtype=bool),
        special_tokens_mask=np.isin(ids, tokenizer.all_special_ids),
        vocabulary_size=len(tokenizer),
        mask_token_id=tokenizer.mask_token_id,
        seed=config.seed + iteration,
    )
    batch = masked_language_model_batch(
        inputs=ModernVBERTTextBatch(
            input_ids=cast(Any, corrupted),
            attention_mask=np.ones_like(ids),
        ),
        positions=positions,
        labels=labels,
    )
    return batch, {
        "tokens_sha256": digest(ids),
        "corrupted_sha256": digest(corrupted),
        "labels_sha256": digest(labels),
        "positions_sha256": digest(positions),
        "supervised_tokens": int((labels != -100).sum()),
    }


def main(config: CanaryConfig | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), default="baseline")
    parser.add_argument(
        "--profile", action="store_true", help="trace warm steps 3-5 with Nsight"
    )
    args = parser.parse_args()
    config = CASES[args.case] if config is None else config
    if args.profile and args.case != "profile-ddp":
        parser.error("--profile requires the bounded profile-ddp case")
    if args.profile:
        runtime = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
        nvtx = ctypes.CDLL("/usr/local/cuda/lib64/libnvToolsExt.so")
        nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
    if not args.output.resolve().is_relative_to(Path("/raid")):
        parser.error("artifacts must be under /raid")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = jax.devices()
    if len(devices) not in (1, 2, 4, 8) or any(d.platform != "gpu" for d in devices):
        raise ValueError("expose 1, 2, 4, or 8 GPUs through CUDA_VISIBLE_DEVICES")
    if args.case.startswith("weak-ddp-") and config.tokens_per_update != 512 * 8 * len(
        devices
    ):
        raise ValueError("weak-scaling case must match the exposed GPU count")
    report = {
        "kind": "real-text-MLM-readiness-not-final-paper-throughput",
        "case": args.case,
        "profiled_step_numbers": [3, 4, 5] if args.profile else [],
        "configuration": asdict(config),
        "model": MODEL.model_dump(mode="json"),
        "devices": [str(d) for d in devices],
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "NCCL_PROTO",
                "NCCL_ALGO",
                "NCCL_DEBUG",
                "NCCL_DEBUG_SUBSYS",
            )
        },
        "jax": jax.__version__,
        "precision": "FP32 masters and AdamW; BF16 compute",
        "rematerialization": "full",
        "attention": "xla",
        "state_donation": True,
        "strategy": (
            "ddp with replicated model and optimizer state, data-sharded batches"
            if config.sharding == "ddp"
            else "fsdp with data and parameter sharding on the same axis; "
            "one GPU degenerates to replication"
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "phases": [],
    }
    root = Path(__file__).resolve().parents[2]
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "src",
            "experiments",
            "tests/tasks/test_masked_language_modeling.py",
        ],
        text=True,
    ).splitlines()
    report["source_sha256"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if (root / p).is_file()
    }
    (args.output / "working-tree.patch").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], text=True)
    )
    for p in subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "src", "experiments"],
        text=True,
    ).splitlines():
        target = args.output / "untracked-source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / p).read_bytes())

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2))

    save()
    tokens, tokenizer, report["data"] = prepare_tokens(config)
    save()
    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    with jax.default_device(jax.devices("cpu")[0]):
        model = ModernBERTMaskedLM.init(
            MODEL,
            key=jax.random.key(config.seed),
            compute_dtype=jnp.bfloat16,
            attention_implementation="xla",
            rematerialization="full",
        )
        state = init_train_state(model, optimizer, precision=precision)
    report["parameters"] = sum(
        x.size for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)
    )
    mesh = jax.make_mesh((len(devices),), ("data",), devices=devices)
    plan = (
        ShardingPlan.ddp(state, optimizer, mesh, axis_name="data")
        if len(devices) == 1 or config.sharding == "ddp"
        else ShardingPlan.fsdp(
            state, optimizer, mesh, parameter_axis_name="data", data_axis_name="data"
        )
    )
    state = plan.place_state(state)
    del model
    jax.block_until_ready(state)
    for phase_index, length in enumerate(config.lengths):
        batch_size = config.tokens_per_update // length
        accumulation = batch_size // (config.local_microbatch * len(devices))
        if batch_size % (config.local_microbatch * len(devices)):
            raise ValueError("microbatch must divide scientific batch")
        step = build_train_step(
            MaskedLanguageModelTask(),
            optimizer,
            plan=plan,
            precision=precision,
            max_grad_norm=config.max_grad_norm,
            donate_state=True,
            gradient_accumulation_steps=accumulation,
        )

        @eqx.filter_jit(donate="all-except-first")
        def advance(inputs, train_state, train_step=step):
            batch, key = inputs
            return train_step(train_state, batch, key)

        iteration = phase_index * config.steps_per_length
        batch, _ = host_batch(tokens, tokenizer, length, iteration, config)
        batch = plan.place_batch(batch)
        key = plan.place_replicated(jax.random.key(config.seed + iteration))
        jax.block_until_ready((batch, key))
        print(
            f"compile length={length} devices={len(devices)} "
            f"accumulation={accumulation}",
            flush=True,
        )
        started = time.perf_counter()
        executable = cast(Any, advance).lower((batch, key), state).compile()
        compile_seconds = time.perf_counter() - started
        memory = executable.compiled.memory_analysis()
        hlo = executable.compiled.as_text()
        (args.output / f"length-{length}.hlo.txt").write_text(hlo)
        collectives = collective_summary(hlo)
        (args.output / f"length-{length}.collectives.json").write_text(
            json.dumps(collectives, indent=2)
        )
        phase = {
            "length": length,
            "global_batch": batch_size,
            "accumulation": accumulation,
            "compile_seconds": compile_seconds,
            "memory_bytes": {
                k: getattr(memory, k)
                for k in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
            "observations": [],
        }
        report["phases"].append(phase)
        required_bytes = (
            memory.argument_size_in_bytes
            + memory.output_size_in_bytes
            + memory.temp_size_in_bytes
            - memory.alias_size_in_bytes
        )
        phase["compiled_required_bytes_per_device"] = required_bytes
        save()
        if (
            args.case in ("strong-ddp", "profile-ddp")
            or args.case.startswith("tuned-ddp-")
        ) and any(
            row["in_loop"] and row["largest_float_array_elements"] >= 1_000_000
            for row in collectives
        ):
            report["status"] = "large_gradient_reduction_still_inside_accumulation"
            save()
            raise RuntimeError(
                "deferred-gradient HLO gate failed; training not executed"
            )
        limits = [d.memory_stats().get("bytes_limit") for d in devices]
        if all(limit is not None for limit in limits) and required_bytes > min(limits):
            report["status"] = "compiled_capacity_exceeds_allocator_limit"
            report["allocator_limits_bytes"] = limits
            report["completed_steps"] = int(state.step)
            save()
            print(
                "CAPACITY: executable exceeds allocator limit; not executed", flush=True
            )
            return
        for index in range(config.steps_per_length):
            iteration = phase_index * config.steps_per_length + index
            if args.profile and index == 2 and runtime.cudaProfilerStart() != 0:
                raise RuntimeError("cudaProfilerStart failed")
            if args.profile:
                nvtx.nvtxRangePushA(f"optimizer_step_{index + 1}".encode())
            started = time.perf_counter()
            if args.profile:
                nvtx.nvtxRangePushA(b"prepare_inputs")
            batch, hashes = host_batch(tokens, tokenizer, length, iteration, config)
            prepared = time.perf_counter()
            if args.profile:
                nvtx.nvtxRangePop()
                nvtx.nvtxRangePushA(b"place_inputs")
            batch = plan.place_batch(batch)
            key = plan.place_replicated(jax.random.key(config.seed + iteration))
            placed = time.perf_counter()
            if args.profile:
                nvtx.nvtxRangePop()
                nvtx.nvtxRangePushA(b"execute_and_synchronize")
            result = executable((batch, key), state)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - started
            if args.profile:
                nvtx.nvtxRangePop()
                nvtx.nvtxRangePop()
                if index == 4 and runtime.cudaProfilerStop() != 0:
                    raise RuntimeError("cudaProfilerStop failed")
            state = result.state
            row = {
                "step": int(state.step),
                "seconds": elapsed,
                "host_prepare_seconds": prepared - started,
                "host_placement_seconds": placed - prepared,
                "dispatch_and_wait_seconds": elapsed - (placed - started),
                "loss": float(result.metrics.loss),
                "gradient_norm": float(result.metrics.gradient_global_norm),
                "update_norm": float(result.metrics.update_global_norm),
                "finite": bool(result.metrics.numeric_finite),
                "skipped": bool(result.metrics.skipped_update),
                **hashes,
            }
            phase["observations"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if not row["finite"] or row["skipped"] or row["step"] != iteration + 1:
                raise RuntimeError("MLM optimizer update failed")
        warm = [r["seconds"] for r in phase["observations"][1:]]
        phase["warm_step_seconds"] = statistics.mean(warm)
        phase["warm_input_tokens_per_second"] = (
            config.tokens_per_update / statistics.mean(warm)
        )
        phase["device_memory"] = [d.memory_stats() for d in devices]
        save()
        del executable, advance, step, result
    report["completed_steps"] = int(state.step)
    report["status"] = "passed"
    save()
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
