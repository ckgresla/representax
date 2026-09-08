"""Experiment 14 integration runner; serious training settings await review."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import subprocess
import sys
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


def integration_job(paths, *, execution="direct", chunk_size=1, matryoshka=False):
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

    if execution not in {"direct", "rematerialized", "custom_vjp"}:
        raise ValueError("unknown integration execution mode")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
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
            (MatryoshkaModifierConfig(dimensions=(32, 64, 128, 256, 512, 768)),)
            if matryoshka
            else ()
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
            grad_cache=(
                None
                if execution == "direct"
                else GradCacheConfig(
                    implementation=execution,
                    micro_batch_size=chunk_size,
                )
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
        "command", choices=("prepare", "inspect", "check-data", "canary")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT / "integration")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execution",
        choices=("direct", "rematerialized", "custom_vjp"),
        default="direct",
    )
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--matryoshka", action="store_true")
    args = parser.parse_args()
    data = importlib.import_module(DATA_MODULE)
    if args.command == "prepare":
        print(
            json.dumps(
                data.integration_sources(args.output / "data", assets=ASSETS), indent=2
            )
        )
        return
    paths = json.loads((args.output / "data/sources.json").read_text())["sources"]
    job, bindings = integration_job(
        paths,
        execution=args.execution,
        chunk_size=args.chunk_size,
        matryoshka=args.matryoshka,
    )
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
        result = run_job(job, args.output / "run", mappers=bindings, stop_after=4)
        assert result.completed_iterations == 4
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
            if index == 4:
                restored.set_state(iterator.get_state())
            batch = next(iterator)
            leaves = jax.tree.leaves(batch)
            assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
            if index >= 4:
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


if __name__ == "__main__":
    main()
