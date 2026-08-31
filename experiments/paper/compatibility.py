"""Manifest-driven model lifecycle acceptance for the paper compatibility table."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/configs/paper-compatibility-v1.json"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    name: str
    repo_id: str
    revision: str
    family: str
    modalities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LifecycleRecipe:
    loader: str
    scorer: bool
    adapter_target: str | None
    loader_parameters: dict[str, Any]
    devices: int = 1


class CompatibilityPairwiseCollator:
    """Build a routed text pair for any native encoder processor."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, rows: Any) -> Any:
        import numpy as np

        from representax.core import Route
        from representax.tasks.pairwise import pairwise_batch

        return pairwise_batch(
            left=self.processor(
                tuple(row["sentence1"] for row in rows),
                route=Route.QUERY,
            ),
            right=self.processor(
                tuple(row["sentence2"] for row in rows),
                route=Route.DOCUMENT,
            ),
            labels=np.asarray(
                tuple(row["score"] for row in rows),
                dtype=np.float32,
            ),
        )


def _read_manifest(path: Path = MANIFEST) -> tuple[ModelEntry, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "representax-paper-compatibility-v1":
        raise ValueError(f"unsupported compatibility manifest: {path}")
    return tuple(
        ModelEntry(
            name=row["name"],
            repo_id=row["repo_id"],
            revision=row["revision"],
            family=row["family"],
            modalities=tuple(row["modalities"]),
        )
        for row in document["models"]
    )


def _recipe(entry: ModelEntry) -> LifecycleRecipe:
    common = {
        "parameter_dtype": "bfloat16",
        "compute_dtype": "bfloat16",
        "sequence_length_buckets": [64],
    }
    if entry.family == "qwen3_vl":
        scorer = "reranker" in entry.name
        return LifecycleRecipe(
            loader=(
                "representax.models.qwen3_vl:load_qwen3_vl_reranker"
                if scorer
                else "representax.models.qwen3_vl:load_qwen3_vl_embedding"
            ),
            scorer=scorer,
            adapter_target="model.text" if scorer else "text",
            loader_parameters={
                **common,
                "patch_count_buckets": [64],
                **(
                    {"processor_mode": "eager_embedding"}
                    if entry.name == "eager-embed-v1"
                    else {}
                ),
            },
            devices=2 if entry.name == "eager-embed-v1" else 1,
        )
    if entry.family == "qwen2_5_omni":
        return LifecycleRecipe(
            loader="representax.models.qwen2_5_omni:load_qwen2_5_omni",
            scorer=False,
            adapter_target="text",
            loader_parameters={
                **common,
                "patch_count_buckets": [64],
                "audio_chunk_count_buckets": [1],
                "audio_token_count_buckets": [64],
            },
            devices=2,
        )
    if entry.family == "clip":
        return LifecycleRecipe(
            loader="representax.models.clip:load_clip",
            scorer=False,
            adapter_target=".*",
            loader_parameters={
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
            },
        )
    if entry.family == "distilbert":
        return LifecycleRecipe(
            loader="representax.models:SentenceEncoder.load_from_hf",
            scorer=False,
            adapter_target="backbone",
            loader_parameters=common,
        )
    if entry.family == "qwen2_vl":
        scorer = "reranker" in entry.name
        imported_adapter = entry.name.startswith("nomic-embed-multimodal-")
        devices = 4 if entry.name == "nomic-embed-multimodal-7b" else 2
        return LifecycleRecipe(
            loader=(
                "representax.models.qwen2_vl:load_qwen2_vl_reranker"
                if scorer
                else "representax.models.qwen2_vl:load_qwen2_vl_embedding"
            ),
            scorer=scorer,
            adapter_target=(
                None if imported_adapter else ("model.text" if scorer else "text")
            ),
            loader_parameters={
                **common,
                "sequence_length_buckets": [64 if imported_adapter else 128],
                "patch_count_buckets": [64],
            },
            devices=devices,
        )
    if entry.family == "llava_next":
        return LifecycleRecipe(
            loader="representax.models.llava_next:load_llava_next",
            scorer=False,
            adapter_target="text.layers",
            loader_parameters={**common, "image_count_buckets": [1]},
            devices=2,
        )
    if entry.family == "nemotron_vl":
        scorer = "rerank" in entry.name
        return LifecycleRecipe(
            loader="representax.models.nemotron_vl:load_nemotron_vl",
            scorer=scorer,
            adapter_target="model.text.layers",
            loader_parameters={**common, "tile_count_buckets": [1]},
        )
    if entry.family == "bidirlm_omni":
        return LifecycleRecipe(
            loader="representax.models.bidirlm_omni:load_bidirlm_omni",
            scorer=False,
            adapter_target="text.layers",
            loader_parameters={
                **common,
                "patch_count_buckets": [64],
                "audio_chunk_buckets": [1],
            },
            devices=2,
        )
    if entry.family == "qwen_reranker":
        return LifecycleRecipe(
            loader="representax.models.qwen_reranker:load_qwen_reranker",
            scorer=True,
            adapter_target="text.layers",
            loader_parameters={**common, "sequence_length_buckets": [128]},
        )
    if entry.family in {"bert", "mpnet"}:
        return LifecycleRecipe(
            loader="representax.models:SentenceEncoder.load_from_hf",
            scorer=False,
            adapter_target="backbone",
            loader_parameters=common,
        )
    raise ValueError(f"no lifecycle recipe for model family {entry.family!r}")


def _records(scorer: bool) -> tuple[dict[str, Any], ...]:
    if scorer:
        return (
            {
                "query": "Which passage describes a harbor?",
                "document": "Morning light falls across a quiet harbor.",
                "label": 1.0,
            },
            {
                "query": "Which passage describes a harbor?",
                "document": "A microscope records dividing cells.",
                "label": 0.0,
            },
            {
                "query": "Which passage describes a laboratory?",
                "document": "A microscope records dividing cells.",
                "label": 1.0,
            },
            {
                "query": "Which passage describes a laboratory?",
                "document": "Morning light falls across a quiet harbor.",
                "label": 0.0,
            },
        )
    return (
        {
            "sentence1": "A quiet harbor at sunrise.",
            "sentence2": "Morning light over calm water.",
            "score": 0.9,
        },
        {
            "sentence1": "A microscope image of living cells.",
            "sentence2": "A crowded street during a rainstorm.",
            "score": 0.1,
        },
        {
            "sentence1": "A small boat crosses still water.",
            "sentence2": "A vessel moves across a calm lake.",
            "score": 0.9,
        },
        {
            "sentence1": "A sample is inspected in a laboratory.",
            "sentence2": "Heavy traffic fills a city avenue.",
            "score": 0.1,
        },
    )


def _job(entry: ModelEntry, checkpoint: Path) -> Any:
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        EvaluatorConfig,
        ExportConfig,
        FSDPConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        QuantizedLoRAConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.cross_encoder import (
        BinaryCrossEntropyConfig,
        PointwiseScoringConfig,
    )
    from representax.tasks.pairwise import CosineRegressionConfig, PairwiseConfig

    recipe = _recipe(entry)
    artifact = source("compatibility://records", map=identity)
    data = DataConfig(
        distribution=mix(artifact, shuffle=False),
        collate=ComponentConfig(
            target=(
                "representax.tasks.cross_encoder:PointwiseCollator"
                if recipe.scorer
                else ("experiments.paper.compatibility:CompatibilityPairwiseCollator")
            )
        ),
        num_threads=0,
        prefetch_buffer_size=0,
    )
    return JobConfig(
        name=f"paper-compatibility-{entry.name}",
        model=ModelConfig(
            target=recipe.loader,
            parameters={
                "model_name_or_path": str(checkpoint),
                "local_files_only": False,
                **recipe.loader_parameters,
            },
        ),
        task=(PointwiseScoringConfig() if recipe.scorer else PairwiseConfig()),
        loss=(
            BinaryCrossEntropyConfig() if recipe.scorer else CosineRegressionConfig()
        ),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-4, "weight_decay": 0.0},
            )
        ),
        data=data,
        training=TrainingConfig(
            global_batch_size=1,
            max_steps=2,
            seed=42,
            batch=BatchConfig(micro_batch_size=1),
            adapter=(
                None
                if recipe.adapter_target is None
                else QuantizedLoRAConfig(
                    rank=4,
                    alpha=8.0,
                    target_pattern=recipe.adapter_target,
                )
            ),
            precision=PrecisionConfig.bfloat16_mixed(),
            activation_rematerialization=(
                "full" if entry.name == "nomic-embed-multimodal-7b" else "none"
            ),
            **(
                {
                    "mesh": MeshConfig(
                        axis_shapes=(recipe.devices,),
                        axis_names=("model",),
                    ),
                    "sharding": FSDPConfig(
                        data_axis=None,
                        parameter_axis="model",
                    ),
                }
                if recipe.devices > 1
                else {}
            ),
        ),
        checkpointing=CheckpointConfig(every=1, keep=2),
        logging=LoggingConfig(console_every=1, timing=True),
        evaluation=EvaluationConfig(
            data=data,
            batch_size=1,
            evaluators=(EvaluatorConfig(),),
            on_end=True,
            max_batches=1,
            save_best=False,
        ),
        export=ExportConfig(enabled=True),
    )


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _resolve_checkpoint(entry: ModelEntry, cache_directory: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            entry.repo_id,
            revision=entry.revision,
            cache_dir=None if cache_directory is None else str(cache_directory),
        )
    ).resolve()


def run_model(
    entry: ModelEntry,
    output: Path,
    *,
    cache_directory: Path | None = None,
) -> dict[str, Any]:
    """Exercise load, update, checkpoint resume, validation, export, and reload."""

    import jax

    from representax import load_inference_bundle
    from representax.train import run_job

    if jax.default_backend() != "gpu":
        raise RuntimeError("paper compatibility acceptance requires a GPU")
    recipe = _recipe(entry)
    if len(jax.devices()) != recipe.devices:
        raise RuntimeError(
            f"{entry.name} requires {recipe.devices} visible GPUs; "
            f"found {len(jax.devices())}"
        )
    checkpoint = _resolve_checkpoint(entry, cache_directory)
    job = _job(entry, checkpoint)
    run_directory = output / "run"
    records = _records(recipe.scorer)
    resolvers = {"compatibility": lambda _artifact: records}

    started = time.perf_counter()
    paused = run_job(
        job,
        run_directory,
        resolvers=resolvers,
        stop_after=1,
    )
    if paused.completed_iterations != 1:
        raise RuntimeError("lifecycle preemption did not stop after one update")
    del paused
    gc.collect()
    jax.clear_caches()
    completed = run_job(
        job,
        run_directory,
        resolvers=resolvers,
        resume=True,
    )
    jax.block_until_ready(completed.state)
    if (
        completed.completed_iterations != job.training.max_steps
        or not completed.resumed
    ):
        raise RuntimeError("lifecycle job did not resume to completion")
    if completed.inference_bundle is None:
        raise RuntimeError("lifecycle job did not export an inference bundle")
    completed_iterations = completed.completed_iterations
    resumed = completed.resumed
    inference_bundle = completed.inference_bundle
    del completed
    gc.collect()
    jax.clear_caches()
    _, restored_job = load_inference_bundle(inference_bundle)
    if restored_job.name != job.name:
        raise RuntimeError("inference reload reconstructed a different job")

    rows = _metric_rows(run_directory / "metrics.jsonl")
    updates = [row for row in rows if row.get("event") == "training_step"]
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    if len(updates) != 2 or not evaluations:
        raise RuntimeError("lifecycle evidence is missing updates or evaluation")
    if any(float(row["metrics"]["train/update_global_norm"]) <= 0 for row in updates):
        raise RuntimeError("lifecycle update did not change trainable parameters")
    result = {
        "schema_version": "representax-paper-compatibility-result-v1",
        "name": entry.name,
        "repo_id": entry.repo_id,
        "revision": entry.revision,
        "family": entry.family,
        "modalities": list(entry.modalities),
        "devices": [device.device_kind for device in jax.devices()],
        "checkpoint": str(checkpoint),
        "updates": len(updates),
        "completed_iterations": completed_iterations,
        "evaluations": len(evaluations),
        "resumed": resumed,
        "native_export": str(inference_bundle),
        "elapsed_seconds": time.perf_counter() - started,
        "final_loss": float(updates[-1]["metrics"]["train/loss"]),
        "final_update_global_norm": float(
            updates[-1]["metrics"]["train/update_global_norm"]
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _model_command(arguments: argparse.Namespace) -> None:
    entries = {entry.name: entry for entry in _read_manifest(arguments.manifest)}
    try:
        entry = entries[arguments.name]
    except KeyError as error:
        raise ValueError(f"unknown compatibility model {arguments.name!r}") from error
    result = run_model(
        entry,
        arguments.output,
        cache_directory=arguments.cache_directory,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _sweep_command(arguments: argparse.Namespace) -> None:
    entries = _read_manifest(arguments.manifest)
    if arguments.names:
        selected = set(arguments.names)
        entries = tuple(entry for entry in entries if entry.name in selected)
        missing = selected - {entry.name for entry in entries}
        if missing:
            raise ValueError(f"unknown compatibility models: {sorted(missing)}")
    arguments.output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    available = list(arguments.gpus)
    order = {gpu: index for index, gpu in enumerate(arguments.gpus)}

    def worker(
        entry: ModelEntry,
        assigned: tuple[str, ...],
    ) -> tuple[bool, dict[str, Any]]:
        output = arguments.output_root / entry.name
        output.mkdir()
        log_path = output / "worker.log"
        command = [
            sys.executable,
            "-m",
            "experiments.paper.compatibility",
            "model",
            "--manifest",
            str(arguments.manifest),
            "--name",
            entry.name,
            "--output",
            str(output),
        ]
        if arguments.cache_directory is not None:
            command.extend(("--cache-directory", str(arguments.cache_directory)))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
        with log_path.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode == 0:
            result = json.loads((output / "result.json").read_text())
            return True, {**result, "gpus": assigned}
        return False, {
            "name": entry.name,
            "gpus": assigned,
            "returncode": process.returncode,
            "log": str(log_path),
        }

    with ThreadPoolExecutor(max_workers=len(arguments.gpus)) as executor:
        pending = list(entries)
        running: dict[Future[tuple[bool, dict[str, Any]]], tuple[str, ...]] = {}
        while pending or running:
            while pending:
                entry = pending[0]
                required = _recipe(entry).devices
                if required > len(arguments.gpus):
                    pending.pop(0)
                    failures.append(
                        {
                            "name": entry.name,
                            "required_gpus": required,
                            "available_gpus": len(arguments.gpus),
                        }
                    )
                    continue
                if required > len(available):
                    break
                pending.pop(0)
                assigned = tuple(available[:required])
                del available[:required]
                running[executor.submit(worker, entry, assigned)] = assigned
            if not running:
                continue
            completed, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in completed:
                assigned = running.pop(future)
                available.extend(assigned)
                available.sort(key=order.__getitem__)
                passed, evidence = future.result()
                (results if passed else failures).append(evidence)
    summary = {
        "schema_version": "representax-paper-compatibility-sweep-v1",
        "manifest": str(arguments.manifest),
        "gpus": arguments.gpus,
        "passed": sorted(results, key=lambda row: row["name"]),
        "failed": sorted(failures, key=lambda row: row["name"]),
    }
    (arguments.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    model = subparsers.add_parser("model", help="run one isolated lifecycle job")
    model.add_argument("--manifest", type=Path, default=MANIFEST)
    model.add_argument("--name", required=True)
    model.add_argument("--output", type=Path, required=True)
    model.add_argument("--cache-directory", type=Path)
    model.set_defaults(function=_model_command)

    sweep = subparsers.add_parser("sweep", help="schedule the pinned panel")
    sweep.add_argument("--manifest", type=Path, default=MANIFEST)
    sweep.add_argument("--output-root", type=Path, required=True)
    sweep.add_argument("--cache-directory", type=Path)
    sweep.add_argument("--gpus", nargs="+", default=["0", "1"])
    sweep.add_argument("--names", nargs="*")
    sweep.set_defaults(function=_sweep_command)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
