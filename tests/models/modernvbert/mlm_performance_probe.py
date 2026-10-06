"""Bounded synthetic MLM capacity/timing control, not a scientific training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
from representax.train import build_train_step, init_train_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 2 or args.batch < 1:
        parser.error("positive batch and at least two steps required")
    if not args.output.resolve().is_relative_to(Path("/raid")):
        parser.error("probe artifacts must live under /raid")
    args.output.mkdir(parents=True, exist_ok=False)
    config = ModernVBERTTextConfig(
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
    model = ModernBERTMaskedLM.init(
        config,
        key=jax.random.key(7),
        compute_dtype=jnp.bfloat16,
        attention_implementation="xla",
        rematerialization="full",
    )
    optimizer = optax.adamw(1e-4, weight_decay=0.01)
    state = init_train_state(model, optimizer)
    step = build_train_step(
        MaskedLanguageModelTask(),
        optimizer,
        precision=resolve_precision_policy(PrecisionConfig.bfloat16_mixed()),
    )
    compiled_step: Any = eqx.filter_jit(step)
    report = {
        "kind": "synthetic-MLM-capacity-not-real-data-throughput",
        "config": config.model_dump(mode="json"),
        "parameters": sum(
            x.size for x in jax.tree.leaves(model) if eqx.is_inexact_array(x)
        ),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_patch": subprocess.check_output(
            ["git", "diff", "HEAD"], text=True
        ),
        "gpu": str(jax.devices()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch": args.batch,
        "precision": PrecisionConfig.bfloat16_mixed().model_dump(mode="json"),
        "attention": "xla",
        "rematerialization": "full",
        "unroll_layers": True,
        "state_donation": False,
        "results": [],
    }
    root = Path(__file__).resolve().parents[3]
    sources = sorted(
        (root / "src/representax/tasks/masked_language_modeling").glob("*.py")
    )
    sources += [
        root / "src/representax/models/modernvbert/mlm.py",
        Path(__file__).resolve(),
    ]
    report["source_snapshot"] = {
        str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "text": path.read_text(),
        }
        for path in sources
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2))
    for length in args.lengths:
        ids = np.random.default_rng(7).integers(
            3, config.vocab_size, (args.batch, length), dtype=np.int32
        )
        corrupted, positions, labels = mask_tokens(
            ids,
            attention_mask=np.ones_like(ids, dtype=bool),
            special_tokens_mask=np.zeros_like(ids, dtype=bool),
            vocabulary_size=config.vocab_size,
            mask_token_id=1,
            seed=7,
        )
        batch = masked_language_model_batch(
            inputs=ModernVBERTTextBatch(
                input_ids=jnp.asarray(corrupted),
                attention_mask=jnp.ones_like(jnp.asarray(ids)),
            ),
            positions=positions,
            labels=labels,
        )
        key = jax.random.key(9)
        jax.block_until_ready((state, batch))
        started = time.perf_counter()
        executable = compiled_step.lower(state, batch, key).compile()
        compile_seconds = time.perf_counter() - started
        memory = executable.compiled.memory_analysis()
        executable_text = executable.compiled.as_text()
        (args.output / f"length-{length}.hlo.txt").write_text(executable_text)
        observations = []
        for index in range(args.steps + 1):
            started = time.perf_counter()
            result = executable(state, batch, key)
            jax.block_until_ready(result)
            elapsed = time.perf_counter() - started
            if not bool(result.metrics.numeric_finite):
                raise RuntimeError("nonfinite MLM update")
            state = result.state
            observations.append(
                {
                    "iteration": index,
                    "seconds": elapsed,
                    "loss": float(result.metrics.loss),
                }
            )
        warm = [row["seconds"] for row in observations[1:]]
        result = {
            "length": length,
            "masked_positions_per_sequence": labels.shape[1],
            "compile_seconds": compile_seconds,
            "observations": observations,
            "warm_mean_step_seconds": statistics.mean(warm),
            "warm_tokens_per_second": args.batch * length / statistics.mean(warm),
            "hlo_lines": len(executable_text.splitlines()),
            "memory": {
                name: getattr(memory, name)
                for name in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            },
            "device_memory_stats": jax.devices()[0].memory_stats(),
        }
        report["results"].append(result)
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
