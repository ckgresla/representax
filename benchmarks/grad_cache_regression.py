"""Compare a committed GradCache implementation with the working tree on one GPU.

This is a bounded synthetic regression control, not a paper throughput result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from representax.models.modernvbert import ModernVBERTTextConfig, ModernVBERTTextEncoder
from representax.precision import FP32_POLICY
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state


def max_difference(left, right):
    differences = []
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if eqx.is_array(a):
            a, b = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            differences.append(float(np.max(np.abs(a - b))))
    return max(differences, default=0.0)


def committed_grad_cache(revision):
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", f"{revision}:src/representax/train/grad_cache.py"],
        cwd=root,
        text=True,
    )
    module = types.ModuleType("representax.train._regression_baseline")
    module.__package__ = "representax.train"
    sys.modules[module.__name__] = module
    exec(compile(source, "grad_cache_baseline.py", "exec"), module.__dict__)
    return module.GradCache, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", required=True, help="Git revision before the extension"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline, source = committed_grad_cache(args.baseline)
    (args.output / "baseline.py").write_text(source)
    root = Path(__file__).resolve().parents[1]
    (args.output / "current.py").write_text(
        (root / "src/representax/train/grad_cache.py").read_text()
    )
    (args.output / "working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=root)
    )
    config = ModernVBERTTextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        layer_types=("full_attention", "sliding_attention") * 2,
        local_attention=64,
        max_position_embeddings=128,
        full_attention_rope_theta=160000.0,
        sliding_attention_rope_theta=10000.0,
        norm_epsilon=1e-5,
    )
    model = ModernVBERTTextEncoder.init(config, key=jax.random.key(7))

    def inputs(key, length):
        ids = jax.random.randint(key, (128, length), 1, config.vocab_size)
        return model.make_batch(input_ids=ids, attention_mask=jnp.ones_like(ids))

    batch = retrieval_batch(
        query=inputs(jax.random.key(8), 32),
        document=inputs(jax.random.key(9), 128),
        positive_mask=jnp.eye(128, dtype=bool),
    )
    task = MNRTask(scale=20.0, symmetric=True)
    optimizer = optax.adamw(1e-5)
    state = init_train_state(model, optimizer)
    key = jax.random.key(10)
    precision = replace(
        FP32_POLICY,
        compute_dtype=jnp.dtype(jnp.bfloat16),
        activation_dtype=jnp.dtype(jnp.bfloat16),
        matrix_dtype=jnp.dtype(jnp.bfloat16),
    )
    rows = []
    for implementation in ("rematerialized", "custom_vjp"):
        for chunk in (32, 64, 128):
            programs = {}
            timings = {"baseline": [], "current": []}
            details = {}
            for name, cls in (("baseline", baseline), ("current", GradCache)):
                step = eqx.filter_jit(
                    build_train_step(
                        task,
                        optimizer,
                        precision=precision,
                        execution=cls(
                            query_chunk_size=chunk, implementation=implementation
                        ),
                    )
                )
                started = time.perf_counter()
                lowered = step.lower(state, batch, key)
                hlo = lowered.as_text()
                compiled = lowered.compile()
                compile_seconds = time.perf_counter() - started
                output = jax.block_until_ready(compiled(state, batch, key))
                programs[name] = compiled
                stem = f"{implementation}-{chunk}-{name}"
                (args.output / f"{stem}.stablehlo").write_text(hlo)
                (args.output / f"{stem}.optimized-hlo").write_text(
                    compiled.compiled.as_text()
                )
                details[name] = {
                    "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                    "compile_seconds": compile_seconds,
                    "loss": float(output.metrics.loss),
                    "self_repeat_max_abs": max_difference(
                        output, jax.block_until_ready(compiled(state, batch, key))
                    ),
                }
                if name == "baseline":
                    expected = output
                else:
                    details[name]["baseline_max_abs"] = max_difference(output, expected)
                for _ in range(5):
                    jax.block_until_ready(compiled(state, batch, key))
            # Identical lowering is the regression gate. GPU scatter reductions
            # can vary between executions even of the exact same executable;
            # preserve self-repeat differences rather than claiming bitwise parity.
            assert details["baseline"]["hlo_sha256"] == details["current"]["hlo_sha256"]
            # Alternate order to reduce thermal/clock drift bias. Identical state
            # per call isolates execution cost, not data loading or convergence.
            for index in range(args.steps):
                for name in (
                    ("baseline", "current")
                    if index % 2 == 0
                    else ("current", "baseline")
                ):
                    start = time.perf_counter()
                    jax.block_until_ready(programs[name](state, batch, key))
                    timings[name].append(time.perf_counter() - start)
            row = {
                "implementation": implementation,
                "chunk": chunk,
                "details": details,
                "seconds": timings,
                "identical_stablehlo": details["baseline"]["hlo_sha256"]
                == details["current"]["hlo_sha256"],
            }
            row["current_over_baseline_rate"] = statistics.median(
                timings["baseline"]
            ) / statistics.median(timings["current"])
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "seconds"}), flush=True
            )
            (args.output / "results.json").write_text(
                json.dumps(
                    {
                        "baseline": args.baseline,
                        "jax": jax.__version__,
                        "device": str(jax.devices()[0]),
                        "steps": args.steps,
                        "config": config.model_dump(),
                        "rows": rows,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
