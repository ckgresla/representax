"""Experiment 14: prepare, inspect, and run Jina text-to-any-modality training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_MODULE = "experiments.14-text-to-any-modality.data"
ASSETS = Path(
    os.environ.get("REPRESENTAX_PAPER_ASSETS", "/raid/representax-paper-assets")
)
OUTPUT = (
    Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
    / "14-text-to-any-modality"
)
MODEL_ID = "jinaai/jina-embeddings-v5-omni-nano-retrieval"
MODEL_REVISION = "b7287f6b6b562e25bc4a28b939d1f936484b4137"


def integration_job(paths):
    """Small real-data plumbing check, not a scientific convergence recipe."""
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        ExportConfig,
        GradCacheConfig,
        JobConfig,
        LoggingConfig,
        LoRAConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import mix, source
    from representax.tasks.modifiers import MatryoshkaModifierConfig
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig

    data = importlib.import_module(DATA_MODULE)
    bindings = {
        f"{DATA_MODULE}.map_image": data.map_image,
        f"{DATA_MODULE}.map_audio": partial(data.map_audio, seconds=2.0),
        f"{DATA_MODULE}.map_video": partial(data.map_video, frames=2),
    }
    datasets = [
        source(
            paths[name],
            name=name,
            map=f"{DATA_MODULE}.map_{name}"
            if name != "text"
            else "representax.data.identity",
        )
        for name in ("image", "audio", "video", "text")
    ]
    job = JobConfig(
        name="paper-14-real-media-integration",
        model=ModelConfig(
            target="representax.models.jina_v5.load_jina_v5_omni",
            parameters={
                "model_name_or_path": str(ASSETS / "jina-v5-omni-nano"),
                "revision": MODEL_REVISION,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": [512],
                "patch_count_buckets": [64],
                "audio_chunk_count_buckets": [1, 2],
                "audio_token_count_buckets": [64],
                "image_min_pixels": 4096,
                "image_max_pixels": 4096,
                "video_min_pixels": 4096,
                "video_max_pixels": 4096,
            },
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=50.0, symmetric=True),
        loss_modifiers=(
            MatryoshkaModifierConfig(dimensions=(32, 64, 128, 256, 512, 768)),
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-5, "weight_decay": 0.01},
            )
        ),
        data=DataConfig(
            distribution=mix(
                *datasets, weights=[1, 1, 1, 1], seed=7, sampling_unit="batch"
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval.RetrievalCollator"
            ),
            num_threads=2,
            prefetch_buffer_size=2,
        ),
        training=TrainingConfig(
            global_batch_size=2,
            max_steps=8,
            seed=7,
            batch=BatchConfig(micro_batch_size=2),
            grad_cache=GradCacheConfig(
                implementation="custom_vjp",
                micro_batch_size=1,
            ),
            adapter=LoRAConfig(rank=4, alpha=8, target_pattern="text"),
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(every=4, keep=2, asynchronous=False),
        logging=LoggingConfig(timing=True, accelerator=True),
        export=ExportConfig(enabled=False),
    )
    return job, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "inspect",
            "check-data",
            "canary",
            "prepare-training",
            "inspect-training",
            "train",
            "launch",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT / "integration")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=(7, 42, 773), default=7)
    parser.add_argument(
        "--strategy",
        choices=("connectors", "connectors-lora", "full"),
        default="connectors",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    data = importlib.import_module(DATA_MODULE)
    if args.command == "prepare-training":
        data.prepare_training(OUTPUT / "training-data", assets=ASSETS)
        return
    if args.command in {"inspect-training", "train", "launch"}:
        if args.command == "launch":
            launch_training(args.gpus)
        elif args.command == "inspect-training":
            print(serious_job(args.strategy, args.seed).model_dump_json(indent=2))
        else:
            train(args.strategy, args.seed, resume=args.resume)
        return
    if args.command == "prepare":
        print(
            json.dumps(
                data.integration_sources(args.output / "data", assets=ASSETS), indent=2
            )
        )
        return
    paths = json.loads((args.output / "data/sources.json").read_text())["sources"]
    job, bindings = integration_job(paths)
    if args.command == "inspect":
        print(job.model_dump_json(indent=2))
        return
    if args.command == "check-data":
        check_data(job, bindings, args.output)
        return
    from representax.train import run_job

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "job.json").write_text(job.model_dump_json(indent=2) + "\n")
    if not args.resume:
        (args.output / "git-revision.txt").write_text(
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        )
        result = run_job(
            job,
            args.output / "run",
            mappers=bindings,
            stop_after=job.checkpointing.every,
        )
        assert result.completed_iterations == job.checkpointing.every
        del result
        gc.collect()
    result = run_job(job, args.output / "run", mappers=bindings, resume=True)
    assert result.completed_iterations == job.training.max_steps and result.resumed
    summary = {
        "completed_iterations": result.completed_iterations,
        "resumed": result.resumed,
        "purpose": "real-data integration with text LoRA; not a final strategy recipe",
    }
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def check_data(job, bindings, output):
    """Exercise real processing and exact resumed batches without model weights."""
    import jax
    import numpy as np

    from representax.models.jina_v5.config import JinaV5OmniConfig
    from representax.models.jina_v5.processing import make_jina_v5_omni_processor
    from representax.train.job import build_batches

    checkpoint = Path(job.model.parameters["model_name_or_path"])
    config = JinaV5OmniConfig.from_hf_config(
        json.loads((checkpoint / "config.json").read_text())
    )
    processor = make_jina_v5_omni_processor(
        checkpoint,
        config,
        **{
            key: value
            for key, value in job.model.parameters.items()
            if key.endswith("_buckets") or key.endswith("_pixels")
        },
    )
    loader = build_batches(
        job.data,
        batch_size=job.training.global_batch_size,
        processor=processor,
        mappers=bindings,
        repeat=True,
    )
    distribution = job.data.distribution
    choices = np.random.default_rng(distribution.seed).choice(
        len(distribution.sources),
        size=job.training.max_steps,
        p=distribution.normalized_weights,
    )
    records = []
    with closing(iter(loader)) as iterator, closing(iter(loader)) as restored:
        for index in range(job.training.max_steps):
            if index == job.checkpointing.every:
                restored.set_state(iterator.get_state())
            batch = next(iterator)
            leaves = jax.tree.leaves(batch)
            assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
            if index >= job.checkpointing.every:
                replay = next(restored)
                assert jax.tree.structure(batch) == jax.tree.structure(replay)
                for left, right in zip(leaves, jax.tree.leaves(replay), strict=True):
                    np.testing.assert_array_equal(left, right)
            records.append(
                {
                    "step": index + 1,
                    "source": distribution.sources[choices[index]].name,
                    "query_shape": list(batch.query.input_ids.shape),
                    "document_shape": list(batch.document.input_ids.shape),
                    "finite": True,
                }
            )
    assert {row["source"] for row in records} == {"image", "audio", "video", "text"}
    result = {"batches": records, "exact_resumed_batches": True}
    (output / "processor-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def serious_job(strategy, seed):
    config = importlib.import_module("experiments.14-text-to-any-modality.config")
    manifest = json.loads((OUTPUT / "training-data/manifest.json").read_text())
    return config.training_job(manifest, strategy=strategy, seed=seed, assets=ASSETS)


def train(strategy, seed, *, resume=False):
    """Use the native lifecycle with independent per-modality IR accumulators."""
    from importlib.metadata import distributions

    import jax

    from representax.precision import resolve_precision_policy
    from representax.train.evaluation import EvaluationRunner
    from representax.train.job import build_job_runtime
    from representax.train.loop import run_training

    data = importlib.import_module(DATA_MODULE)
    evaluation = importlib.import_module(
        "experiments.14-text-to-any-modality.evaluation"
    )
    directory = OUTPUT / "runs" / strategy / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and (directory / "job.json").exists():
        raise FileExistsError(f"existing run: {directory}; resume explicitly")
    job = serious_job(strategy, seed)
    (directory / "job.json").write_text(job.model_dump_json(indent=2) + "\n")
    if not resume:
        (directory / "provenance.json").write_text(
            json.dumps(
                {
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "command": sys.argv,
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "jax": jax.__version__,
                    "devices": [str(d) for d in jax.devices()],
                    "packages": sorted(
                        f"{d.metadata['Name']}=={d.version}" for d in distributions()
                    ),
                    "data_manifest_sha256": hashlib.sha256(
                        (OUTPUT / "training-data/manifest.json").read_bytes()
                    ).hexdigest(),
                    "environment_lock_sha256": hashlib.sha256(
                        (ROOT / "experiments/uv.lock").read_bytes()
                    ).hexdigest(),
                },
                indent=2,
            )
        )
    runtime = build_job_runtime(
        job,
        mappers={
            f"{DATA_MODULE}.map_audio": partial(data.map_audio, seconds=10.0),
            f"{DATA_MODULE}.map_video": partial(data.map_video, frames=8),
        },
        place_initial_state=not resume,
    )
    runner = EvaluationRunner(
        evaluation.make_evaluator(job.evaluation),
        precision=resolve_precision_policy(job.training.precision),
    )
    result = run_training(
        state=runtime.state,
        step=runtime.step,
        batches=runtime.batches,
        job=job,
        run_directory=directory / "run",
        resume=resume,
        place_state=runtime.place_state,
        place_batch=runtime.place_batch,
        evaluation_runners=(runner,),
        evaluation_batches=lambda: evaluation.evaluation_batches(
            job.evaluation, runtime.processor
        ),
        startup_metrics=runtime.startup_metrics,
        export_inference=job.export.enabled,
    )
    assert result.completed_iterations == job.training.max_steps
    (directory / "result.json").write_text(
        json.dumps(
            {
                "completed_iterations": result.completed_iterations,
                "resumed": result.resumed,
                "inference_bundle": str(result.inference_bundle),
                "memory": jax.devices()[0].memory_stats(),
            },
            indent=2,
        )
    )


def launch_training(gpus):
    """One worker per GPU; start later seeds after that strategy warms up."""
    if not gpus or len(gpus) != len(set(gpus)) or any(g < 0 or g > 3 for g in gpus):
        raise ValueError("provide distinct authorized GPUs from 0,1,2,3")
    config = importlib.import_module("experiments.14-text-to-any-modality.config")
    pending = [
        (strategy, seed) for seed in config.SEEDS for strategy in config.LEARNING_RATES
    ]
    running, finished, failed = {}, [], []
    warmed = set()
    root = OUTPUT / "runs"
    root.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        HF_HOME="/raid/.cache/huggingface",
        TOKENIZERS_PARALLELISM="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
        OMP_NUM_THREADS="4",
        JAX_COMPILATION_CACHE_DIR=str(OUTPUT / "training-jax-cache"),
    )
    while pending or running:
        for gpu, (process, log, strategy, seed) in list(running.items()):
            metrics = root / strategy / f"seed-{seed}/run/metrics.jsonl"
            if metrics.exists():
                for line in metrics.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        row.get("event") == "training_step"
                        and row.get("iteration", 0) >= 10
                    ):
                        warmed.add(strategy)
            status = process.poll()
            if status is None:
                continue
            log.close()
            del running[gpu]
            (finished if status == 0 else failed).append((strategy, seed, status))
            if status == 0:
                warmed.add(strategy)
            else:
                # Preserve failure evidence and do not spend two more seeds on that arm.
                pending = [item for item in pending if item[0] != strategy]
            print(f"END gpu={gpu} {strategy} seed={seed} exit={status}", flush=True)
        for gpu in gpus:
            if gpu in running:
                continue
            index = next(
                (i for i, (s, seed) in enumerate(pending) if seed == 7 or s in warmed),
                None,
            )
            if index is None:
                continue
            strategy, seed = pending.pop(index)
            directory = root / strategy / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=False)
            log = (directory / "worker.log").open("w")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    __file__,
                    "train",
                    "--strategy",
                    strategy,
                    "--seed",
                    str(seed),
                ],
                cwd=ROOT,
                env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = process, log, strategy, seed
            print(
                f"START gpu={gpu} {strategy} seed={seed} pid={process.pid}", flush=True
            )
        (root / "dispatch.json").write_text(
            json.dumps(
                {
                    "running": {
                        str(g): {"pid": p.pid, "strategy": s, "seed": seed}
                        for g, (p, _, s, seed) in running.items()
                    },
                    "pending": pending,
                    "completed": finished,
                    "failed": failed,
                },
                indent=2,
            )
        )
        if running:
            time.sleep(15)
    if failed:
        raise RuntimeError(f"failed training jobs: {failed}")


if __name__ == "__main__":
    main()
