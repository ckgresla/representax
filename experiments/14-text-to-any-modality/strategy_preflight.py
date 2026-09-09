"""Three real-media parameter-selection lifecycle checks on one GPU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = importlib.import_module("experiments.14-text-to-any-modality.run")
SWEEP = importlib.import_module("experiments.14-text-to-any-modality.chunk_sweep")
STRATEGIES = ("connectors", "connectors-lora", "full")
OUTPUT = RUN.OUTPUT / "strategy-preflight"
CONNECTORS = r"^\.(vision\.merger|audio\.projection)\."


def strategy_job(paths, strategy):
    from representax.config import JobConfig

    if strategy not in STRATEGIES:
        raise ValueError("unknown parameter strategy")
    base, _ = RUN.integration_job(paths)
    capacity, bindings = SWEEP.probe_job(paths, "text", 2)
    config = capacity.model_dump(mode="json")
    config["name"] = f"paper-14-preflight-{strategy}"
    config["data"]["distribution"] = base.data.distribution.model_dump(mode="json")
    distribution = config["data"]["distribution"]
    if strategy == "connectors":
        distribution["sources"] = [
            s for s in distribution["sources"] if s["name"] != "text"
        ]
        distribution["weights"] = [1.0] * 3
    training = config["training"]
    training["max_steps"] = 8
    training["adapter"] = (
        {"rank": 8, "alpha": 16, "target_pattern": ".*"}
        if strategy == "connectors-lora"
        else None
    )
    training["trainable_pattern"] = (
        ".*"
        if strategy == "full"
        else CONNECTORS + (r"|\.lora_[ab]$" if strategy == "connectors-lora" else "")
    )
    training["trainable_embedding_rows"] = (
        {} if strategy == "full" else {".text.token_embedding": [128257, 128258]}
    )
    config["checkpointing"].update(every=4, keep=1, save_final=True)
    return JobConfig.model_validate(config), bindings


def parameter_snapshot(model, selected):
    import equinox as eqx
    import jax
    import numpy as np

    flags = dict(
        (jax.tree_util.keystr(p), v)
        for p, v in jax.tree_util.tree_flatten_with_path(selected)[0]
    )
    result = {}
    for path, value in jax.tree_util.tree_flatten_with_path(model)[0]:
        if not eqx.is_inexact_array(value):
            continue
        name = jax.tree_util.keystr(path)
        array = np.ascontiguousarray(jax.device_get(value))
        result[name] = {
            "trainable": flags[name],
            "elements": int(array.size),
            "sha256": hashlib.sha256(array).hexdigest(),
        }
    return result


def worker(root, strategy):
    import jax

    from representax.train import run_job
    from representax.train.job import load_model, prepare_model

    directory = root / strategy
    paths = json.loads((SWEEP.OUTPUT / "data/sources.json").read_text())["sources"]
    job, bindings = strategy_job(paths, strategy)
    (directory / "job.json").write_text(job.model_dump_json(indent=2))
    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.key(job.training.seed)
        model, _ = load_model(
            job.model,
            key=jax.random.fold_in(key, 0),
            activation_rematerialization=job.training.activation_rematerialization,
        )
        model, selected = prepare_model(
            model,
            adapter=job.training.adapter,
            key=jax.random.fold_in(key, 1),
            trainable_pattern=job.training.trainable_pattern,
            trainable_embedding_rows=job.training.trainable_embedding_rows,
        )
        before = parameter_snapshot(model, selected)
    (directory / "parameters-before.json").write_text(json.dumps(before, indent=2))
    del model
    gc.collect()
    result = run_job(job, directory / "run", mappers=bindings, stop_after=4)
    assert result.completed_iterations == 4
    del result
    gc.collect()
    result = run_job(job, directory / "run", mappers=bindings, resume=True)
    assert result.completed_iterations == 8 and result.resumed
    after = parameter_snapshot(result.state.model, selected)
    (directory / "parameters-after.json").write_text(json.dumps(after, indent=2))
    changes = [
        name for name in before if before[name]["sha256"] != after[name]["sha256"]
    ]
    frozen_changes = [name for name in changes if not before[name]["trainable"]]
    summary = SWEEP.summarize(directory)
    assert not frozen_changes and changes
    assert summary["all_updates_finite"] and summary["skipped_updates"] == 0
    report = {
        "strategy": strategy,
        "resumed": True,
        "trainable_parameters": sum(
            v["elements"] for v in before.values() if v["trainable"]
        ),
        "total_parameters": sum(v["elements"] for v in before.values()),
        "changed_parameters": changes,
        "frozen_changes": frozen_changes,
        "jax_memory_stats": jax.devices()[0].memory_stats(),
        **summary,
    }
    (directory / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "worker"))
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "worker":
        if args.strategy is None:
            parser.error("worker requires --strategy")
        worker(args.output, args.strategy)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "git-revision.txt").write_text(
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        HF_HOME="/raid/.cache/huggingface",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
        JAX_COMPILATION_CACHE_DIR=str(args.output / "jax-cache"),
        TOKENIZERS_PARALLELISM="false",
    )
    for strategy in STRATEGIES:
        directory = args.output / strategy
        directory.mkdir(parents=True, exist_ok=False)
        print(f"START {strategy}", flush=True)
        with (directory / "worker.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "worker",
                    "--strategy",
                    strategy,
                    "--output",
                    str(args.output),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        print(f"END {strategy}: exit={result.returncode}", flush=True)


if __name__ == "__main__":
    main()
