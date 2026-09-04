#!/usr/bin/env python3
"""Matched MS MARCO dense retrieval training and NanoMSMARCO evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from experiments.preflights.provenance import reference_source, write_reference_result

TRAIN_DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
TRAIN_DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
TRAIN_DATASET_FILE = "triplet-all/train-00000-of-00039.parquet"
EVALUATION_DATASET_ID = "sentence-transformers/NanoBEIR-en"
EVALUATION_DATASET_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
EVALUATION_SUBSET = "NanoMSMARCO"
ORACLE_VERSION = reference_source("sentence-transformers").release
if ORACLE_VERSION is None:  # pragma: no cover - frozen manifest invariant
    raise ValueError("the Sentence Transformers reference requires a release")
TRANSFORMERS_VERSION = "5.6.0"
THREE_RUN_95_PERCENT_T_CRITICAL = 4.302652729911275


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One native dense family with an identical Sentence Transformers oracle."""

    name: str
    model_id: str
    revision: str
    native_target: str = "representax.models:SentenceEncoder.load_from_hf"
    initial_metric_tolerance: float = 1e-6
    checkpoint: Path | None = None


MODEL_SPECS = {
    "minilm": ModelSpec(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    ),
    "mpnet": ModelSpec(
        name="mpnet",
        model_id="sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        initial_metric_tolerance=5e-3,
    ),
    "modernbert": ModelSpec(
        name="modernbert",
        model_id="jhu-clsp/ettin-encoder-150m",
        revision="45d08642849e5c5701b162671ac811b7654bfd9f",
        native_target=(
            "experiments.preflights.dense_retrieval:load_raw_modernbert_encoder"
        ),
        initial_metric_tolerance=2.5e-2,
    ),
    "modernvbert": ModelSpec(
        name="modernvbert",
        model_id="ModernVBERT/modernvbert-embed",
        revision="da507113c3fdbc2e49d39c4b0148025c6bd008f9",
        native_target="representax.integrations.load_modernvbert_text_encoder",
        initial_metric_tolerance=1e-4,
    ),
}


def _model_spec(arguments: argparse.Namespace) -> ModelSpec:
    checkpoint = arguments.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    return replace(MODEL_SPECS[arguments.model], checkpoint=checkpoint)


def _checkpoint(spec: ModelSpec) -> Path:
    if spec.checkpoint is None:  # pragma: no cover - established by _model_spec
        raise AssertionError("benchmark model spec requires a local checkpoint")
    return spec.checkpoint


def _native_model_target(spec: ModelSpec, *, packing: bool) -> str:
    if packing:
        return "experiments.preflights.padding:load_packed_mpnet_sentence_encoder"
    return spec.native_target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    if not files:
        raise ValueError(f"artifact directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _maximum_host_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _evaluation_filenames() -> tuple[str, ...]:
    return tuple(
        f"{kind}/{EVALUATION_SUBSET}-00000-of-00001.parquet"
        for kind in ("queries", "corpus", "qrels")
    )


def _prepare_data(directory: Path) -> None:
    """Download immutable upstream Parquets and record their exact byte identity."""

    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    directory = directory.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"data directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = directory / "artifacts"
    requested = (
        (TRAIN_DATASET_ID, TRAIN_DATASET_REVISION, TRAIN_DATASET_FILE),
        *(
            (EVALUATION_DATASET_ID, EVALUATION_DATASET_REVISION, filename)
            for filename in _evaluation_filenames()
        ),
    )
    files = {}
    for dataset_id, revision, filename in requested:
        path = Path(
            hf_hub_download(
                repo_id=dataset_id,
                repo_type="dataset",
                revision=revision,
                filename=filename,
                local_dir=artifacts / dataset_id,
            )
        ).resolve()
        files[f"{dataset_id}/{filename}"] = {
            "path": str(path.relative_to(directory)),
            "sha256": _sha256(path),
            "rows": parquet.ParquetFile(path).metadata.num_rows,
        }
    _atomic_json(
        directory / "manifest.json",
        {
            "schema_version": "representax-dense-retrieval-data-v1",
            "training": {
                "dataset_id": TRAIN_DATASET_ID,
                "revision": TRAIN_DATASET_REVISION,
                "file": TRAIN_DATASET_FILE,
            },
            "evaluation": {
                "dataset_id": EVALUATION_DATASET_ID,
                "revision": EVALUATION_DATASET_REVISION,
                "subset": EVALUATION_SUBSET,
                "files": list(_evaluation_filenames()),
            },
            "files": files,
        },
    )


def _artifact_path(data_directory: Path, dataset_id: str, filename: str) -> Path:
    manifest = json.loads((data_directory / "manifest.json").read_text())
    record = manifest["files"][f"{dataset_id}/{filename}"]
    path = (data_directory / record["path"]).resolve()
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"benchmark artifact hash changed: {path}")
    return path


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationData:
    queries: tuple[tuple[int, str], ...]
    documents: tuple[tuple[int, str], ...]
    relevant_documents: Mapping[int, frozenset[int]]


def _evaluation_data(data_directory: Path) -> RetrievalEvaluationData:
    import pyarrow.parquet as parquet

    paths = {
        kind: _artifact_path(
            data_directory,
            EVALUATION_DATASET_ID,
            f"{kind}/{EVALUATION_SUBSET}-00000-of-00001.parquet",
        )
        for kind in ("queries", "corpus", "qrels")
    }
    queries = tuple(
        (int(row["_id"]), str(row["text"]))
        for row in parquet.read_table(paths["queries"]).to_pylist()
    )
    documents = tuple(
        (int(row["_id"]), str(row["text"]))
        for row in parquet.read_table(paths["corpus"]).to_pylist()
    )
    qrels: dict[int, set[int]] = {}
    for row in parquet.read_table(paths["qrels"]).to_pylist():
        qrels.setdefault(int(row["query-id"]), set()).add(int(row["corpus-id"]))
    if set(qrels) != {query_id for query_id, _ in queries}:
        raise ValueError("NanoMSMARCO queries and relevance judgments differ")
    document_ids = {document_id for document_id, _ in documents}
    missing = set().union(*qrels.values()) - document_ids
    if missing:
        raise ValueError(f"NanoMSMARCO qrels reference missing documents: {missing}")
    return RetrievalEvaluationData(
        queries=queries,
        documents=documents,
        relevant_documents={
            query_id: frozenset(document_ids)
            for query_id, document_ids in qrels.items()
        },
    )


def _training_table(path: Path, rows: int) -> Any:
    """Read only the row groups needed by one fixed benchmark job."""

    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path, memory_map=True)
    if rows > source.metadata.num_rows:
        raise ValueError(
            f"benchmark requests {rows} rows from a {source.metadata.num_rows}-row file"
        )
    groups = []
    available = 0
    for index in range(source.num_row_groups):
        groups.append(index)
        available += source.metadata.row_group(index).num_rows
        if available >= rows:
            break
    return source.read_row_groups(groups, columns=("query", "positive")).slice(0, rows)


def _training_compile_summary(run_directory: Path) -> tuple[float, int]:
    durations = []
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            value = row["metrics"].get("perf/compilation_and_first_step_seconds")
            if value is not None:
                durations.append(float(value))
    if not durations:
        raise ValueError("Representax run did not report executable first use")
    return sum(durations), len(durations)


def _steady_state_totals(rows: Sequence[tuple[float, int]]) -> tuple[float, int, int]:
    if len(rows) >= 8:
        rows = rows[len(rows) // 2 :]
    if not rows:
        raise ValueError("run did not report a steady-state training step")
    return sum(row[0] for row in rows), len(rows), sum(row[1] for row in rows)


def _training_steady_state_summary(run_directory: Path) -> tuple[float, int, int]:
    rows = []
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "training_step":
                continue
            metrics = row["metrics"]
            if metrics.get("perf/compilation_and_first_step_seconds") is not None:
                continue
            step_seconds = metrics.get("perf/step_seconds")
            if step_seconds is None:
                continue
            rows.append((float(step_seconds), int(metrics["perf/examples"])))
    try:
        return _steady_state_totals(rows)
    except ValueError as error:
        raise ValueError(
            "Representax run did not report a steady-state training step"
        ) from error


def _reference_steady_state_summary(run_directory: Path) -> tuple[float, int, int]:
    rows = []
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "training_step" or row["iteration"] <= 1:
                continue
            metrics = row["metrics"]
            rows.append(
                (float(metrics["perf/step_seconds"]), int(metrics["perf/examples"]))
            )
    try:
        return _steady_state_totals(rows)
    except ValueError as error:
        raise ValueError(
            "reference run did not report a steady-state training step"
        ) from error


def _paired_steady_state_summary(
    representax_directory: Path,
    reference_directory: Path,
) -> tuple[float, float, int, int, int, int]:
    def rows(
        directory: Path,
        *,
        exclude_compilation: bool,
    ) -> dict[int, tuple[float, int]]:
        result = {}
        with (directory / "metrics.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") != "training_step":
                    continue
                metrics = row["metrics"]
                if (
                    exclude_compilation
                    and metrics.get("perf/compilation_and_first_step_seconds")
                    is not None
                ):
                    continue
                step_seconds = metrics.get("perf/step_seconds")
                if step_seconds is None:
                    continue
                result[int(row["iteration"])] = (
                    float(step_seconds),
                    int(metrics["perf/examples"]),
                )
        return result

    native = rows(representax_directory, exclude_compilation=True)
    reference = rows(reference_directory, exclude_compilation=False)
    iterations = sorted(native.keys() & reference.keys())
    if len(iterations) >= 8:
        iterations = iterations[len(iterations) // 2 :]
    if not iterations:
        raise ValueError("paired run did not report matching warmed training steps")
    native_seconds = 0.0
    reference_seconds = 0.0
    examples = 0
    for iteration in iterations:
        native_duration, native_examples = native[iteration]
        reference_duration, reference_examples = reference[iteration]
        if native_examples != reference_examples:
            raise ValueError(
                f"paired update {iteration} processed different example counts"
            )
        native_seconds += native_duration
        reference_seconds += reference_duration
        examples += native_examples
    return (
        native_seconds,
        reference_seconds,
        len(iterations),
        examples,
        iterations[0],
        iterations[-1],
    )


def _final_training_loss(run_directory: Path) -> float:
    final_loss: float | None = None
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            value = row["metrics"].get("train/loss")
            if value is not None:
                final_loss = float(value)
    if final_loss is None:
        raise ValueError("Representax run did not report a training loss")
    return final_loss


def _representax_evaluation_history(run_directory: Path) -> list[dict[str, Any]]:
    started: datetime | None = None
    with (run_directory / "events.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "training_started":
                started = datetime.fromisoformat(row["timestamp"])
                break
    if started is None:
        raise ValueError("Representax run did not report training start")
    losses: dict[int, float] = {}
    history = []
    evaluation_seconds = 0.0
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            iteration = int(row["iteration"])
            metrics = row["metrics"]
            if "train/loss" in metrics:
                losses[iteration] = float(metrics["train/loss"])
            if row.get("event") != "evaluation":
                continue
            duration = float(metrics["perf/evaluation_seconds"])
            elapsed = (
                (datetime.fromisoformat(row["timestamp"]) - started).total_seconds()
                - evaluation_seconds
                - duration
            )
            evaluation_seconds += duration
            history.append(
                {
                    "updates": iteration,
                    "metrics": {
                        name: float(value)
                        for name, value in metrics.items()
                        if name.startswith("valid/")
                    },
                    "final_train_loss": losses.get(iteration),
                    "evaluation_seconds": duration,
                    "evaluation_compilation_seconds": float(
                        metrics["perf/evaluation_compilation_seconds"]
                    ),
                    "training_elapsed_seconds": max(elapsed, 0.0),
                }
            )
    if not history:
        raise ValueError("Representax run did not report periodic evaluation")
    return history


def _representax_report(
    spec: ModelSpec,
    *,
    data_directory: Path,
    run_directory: Path,
    batch_size: int,
    steps: int,
    maximum_length: int,
    sequence_length_buckets: tuple[int, ...],
    cache_chunk_size: int | None,
    query_cache_chunk_size: int | None,
    document_cache_chunk_size: int | None,
    loss_row_chunk_size: int | None,
    grad_cache_implementation: str,
    evaluation_batch_size: int,
    evaluation_every_steps: int | None,
    data_threads: int,
    prefetch_buffer_size: int,
    seed: int,
    world_size: int,
    mixed_precision: bool,
    telemetry: bool,
    checkpoint_every: int | None,
    stop_after: int | None,
    resume: bool,
    export: bool,
    packing: bool,
    packing_query_shape: tuple[int, int] | None,
    packing_document_shape: tuple[int, int] | None,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from experiments.preflights.dense_retrieval import (
        evaluation_rows,
        event_metrics,
        fixed_rows_resolver,
    )

    from representax import load_inference_bundle
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        DDPConfig,
        EvaluationConfig,
        ExportConfig,
        GradCacheConfig,
        InformationRetrievalEvaluatorConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig
    from representax.train import run_job

    if batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    if not sequence_length_buckets or any(
        value <= 0 for value in sequence_length_buckets
    ):
        raise ValueError("sequence length buckets must be positive")
    if max(sequence_length_buckets) != maximum_length:
        raise ValueError("the largest sequence length bucket must equal maximum length")

    checkpoint = _checkpoint(spec)
    evaluation_data = _evaluation_data(data_directory)
    evaluation_source = "paper-evaluation://nano-msmarco"
    evaluation_records = evaluation_rows(
        evaluation_data.queries,
        evaluation_data.documents,
        batch_size=evaluation_batch_size,
    )
    training_path = _artifact_path(
        data_directory,
        TRAIN_DATASET_ID,
        TRAIN_DATASET_FILE,
    )
    training_data = DataConfig(
        distribution=mix(source(str(training_path), map=identity), shuffle=False),
        collate=ComponentConfig(
            target="representax.tasks.retrieval.RetrievalCollator",
        ),
        drop_remainder=True,
        num_threads=data_threads,
        prefetch_buffer_size=prefetch_buffer_size,
    )
    model_parameters: dict[str, Any] = {
        "model_name_or_path": str(checkpoint),
        "revision": spec.revision,
        "local_files_only": True,
        "parameter_dtype": "float32",
        "compute_dtype": "bfloat16" if mixed_precision else "float32",
    }
    model_target = _native_model_target(spec, packing=packing)
    if packing:
        model_parameters.update(
            {
                "maximum_batch_size": max(batch_size, evaluation_batch_size),
                "sequence_lengths": sequence_length_buckets,
            }
        )
        if packing_query_shape is not None and packing_document_shape is not None:
            model_parameters.update(
                {
                    "fixed_batch_size": batch_size,
                    "fixed_query_shape": packing_query_shape,
                    "fixed_document_shape": packing_document_shape,
                }
            )
    else:
        model_parameters["sequence_length_buckets"] = sequence_length_buckets

    job = JobConfig(
        name=f"matched-dense-retrieval-{spec.name}",
        model=ModelConfig(
            target=model_target,
            parameters=model_parameters,
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=20.0, symmetric=False),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            schedule=ComponentConfig(
                target="optax.warmup_cosine_decay_schedule",
                parameters={
                    "init_value": 0.0,
                    "peak_value": 2e-5,
                    "warmup_steps": round(steps * 0.06),
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=training_data,
        training=TrainingConfig(
            global_batch_size=batch_size,
            max_steps=steps,
            seed=seed,
            mesh=MeshConfig(axis_shapes=(world_size,), axis_names=("data",)),
            sharding=DDPConfig(axis="data"),
            batch=BatchConfig(micro_batch_size=batch_size // world_size),
            grad_cache=(
                None
                if cache_chunk_size is None
                else GradCacheConfig(
                    micro_batch_size=cache_chunk_size,
                    query_micro_batch_size=query_cache_chunk_size,
                    document_micro_batch_size=document_cache_chunk_size,
                    loss_row_chunk_size=loss_row_chunk_size,
                    implementation=grad_cache_implementation,
                )
            ),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=(
                PrecisionConfig.bfloat16_mixed()
                if mixed_precision
                else PrecisionConfig()
            ),
        ),
        checkpointing=(
            None
            if checkpoint_every is None
            else CheckpointConfig(every=checkpoint_every, keep=3)
        ),
        logging=LoggingConfig(
            console_every=steps,
            timing=telemetry,
            accelerator=telemetry,
        ),
        evaluation=EvaluationConfig(
            data=DataConfig(
                distribution=mix(
                    source(evaluation_source, map=identity),
                    shuffle=False,
                ),
                collate=ComponentConfig(
                    target=(
                        "experiments.preflights.dense_retrieval:RetrievalEvaluationCollator"
                    ),
                ),
                drop_remainder=True,
                num_threads=0,
                prefetch_buffer_size=0,
            ),
            batch_size=evaluation_batch_size,
            evaluators=(
                InformationRetrievalEvaluatorConfig(
                    name=EVALUATION_SUBSET,
                    relevant_documents=dict(evaluation_data.relevant_documents),
                    score_functions=("cosine",),
                    main_score_function="cosine",
                    accuracy_at_k=(1, 3, 5, 10),
                    precision_recall_at_k=(1, 3, 5, 10, 100),
                    mrr_at_k=(10,),
                    ndcg_at_k=(10,),
                    map_at_k=(100,),
                ),
            ),
            every_steps=evaluation_every_steps,
            on_start=True,
            on_end=True,
            primary_metric=(f"valid/{EVALUATION_SUBSET}/cosine_ndcg@10"),
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(enabled=export),
    )

    lifecycle_started = time.perf_counter()
    result = run_job(
        job,
        run_directory,
        resolvers={
            "paper-evaluation": fixed_rows_resolver(evaluation_records),
        },
        stop_after=stop_after,
        resume=resume,
    )
    jax.block_until_ready(result.state)
    lifecycle_seconds = time.perf_counter() - lifecycle_started
    metric_path = run_directory / "metrics.jsonl"
    startup = event_metrics(metric_path, "startup")
    model_and_data_seconds = startup["perf/startup_seconds"]
    export_seconds = (
        event_metrics(metric_path, "export")["perf/export_seconds"] if export else 0.0
    )
    training_seconds = max(
        lifecycle_seconds - model_and_data_seconds - export_seconds,
        0.0,
    )
    evaluation_history = _representax_evaluation_history(run_directory)
    initial = evaluation_history[0]["metrics"]
    final = evaluation_history[-1]["metrics"]
    initial_seconds = evaluation_history[0]["evaluation_seconds"]
    initial_compile_seconds = evaluation_history[0]["evaluation_compilation_seconds"]
    final_seconds = evaluation_history[-1]["evaluation_seconds"]
    final_compile_seconds = evaluation_history[-1]["evaluation_compilation_seconds"]
    evaluation_during_training_seconds = sum(
        item["evaluation_seconds"] for item in evaluation_history
    )
    training_compute_seconds = training_seconds - evaluation_during_training_seconds
    compilation_and_first_use_seconds, compiled_signature_count = (
        _training_compile_summary(run_directory)
    )
    steady_state_seconds, steady_state_step_count, steady_state_examples = (
        _training_steady_state_summary(run_directory)
    )
    device_memory = jax.devices()[0].memory_stats() or {}
    parameter_count = sum(
        int(value.size)
        for value in jax.tree.leaves(result.state.model)
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.inexact)
    )
    exported_bundle = None
    if result.inference_bundle is not None:
        _, restored_job = load_inference_bundle(result.inference_bundle)
        if restored_job.name != job.name:
            raise ValueError("reloaded inference bundle contains the wrong job")
        exported_bundle = str(result.inference_bundle)
    return {
        "schema_version": "representax-dense-retrieval-worker-v1",
        "framework": "representax",
        "jax_version": jax.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "steps": steps,
        "maximum_length": maximum_length,
        "sequence_length_buckets": list(sequence_length_buckets),
        "packing": packing,
        "packing_query_shape": packing_query_shape,
        "packing_document_shape": packing_document_shape,
        "cache_chunk_size": cache_chunk_size,
        "query_cache_chunk_size": query_cache_chunk_size,
        "document_cache_chunk_size": document_cache_chunk_size,
        "loss_row_chunk_size": loss_row_chunk_size,
        "grad_cache_implementation": grad_cache_implementation,
        "seed": seed,
        "world_size": world_size,
        "mixed_precision": mixed_precision,
        "resumed": result.resumed,
        "completed_iterations": result.completed_iterations,
        "exported_bundle": exported_bundle,
        "export_seconds": export_seconds,
        "model_and_data_seconds": model_and_data_seconds,
        "training_seconds": training_seconds,
        "training_compute_seconds": training_compute_seconds,
        "compilation_and_first_use_seconds": compilation_and_first_use_seconds,
        "compiled_signature_count": compiled_signature_count,
        "steady_state_seconds": steady_state_seconds,
        "steady_state_step_count": steady_state_step_count,
        "steady_state_examples": steady_state_examples,
        "amortized_examples_per_second": batch_size * steps / training_compute_seconds,
        "steady_state_examples_per_second": (
            steady_state_examples / steady_state_seconds
        ),
        "final_train_loss": _final_training_loss(run_directory),
        "initial_evaluation": initial,
        "final_evaluation": final,
        "initial_evaluation_seconds": initial_seconds,
        "initial_evaluation_compilation_seconds": initial_compile_seconds,
        "final_evaluation_seconds": final_seconds,
        "final_evaluation_compilation_seconds": final_compile_seconds,
        "evaluation_history": evaluation_history,
        "maximum_host_rss_bytes": _maximum_host_rss_bytes(),
        "jax_allocator_bytes_limit": device_memory.get("bytes_limit"),
        "jax_allocator_bytes_in_use": device_memory.get("bytes_in_use"),
        "jax_allocator_peak_bytes_in_use": device_memory.get("peak_bytes_in_use"),
        "jax_allocator_peak_pool_bytes": device_memory.get("peak_pool_bytes"),
    }


def _pad_features(features: dict[str, Any], length: int, pad_token_id: int) -> Any:
    import torch

    padded = {}
    for name, value in features.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            padded[name] = value
            continue
        if value.shape[1] > length:
            raise ValueError(f"tokenized {name} exceeds fixed length {length}")
        fill = pad_token_id if name.endswith("input_ids") else 0
        output = torch.full(
            (value.shape[0], length),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        output[:, : value.shape[1]] = value
        padded[name] = output
    return padded


def _oracle_retrieval_metrics(
    model: Any,
    data: RetrievalEvaluationData,
    batch_size: int,
    maximum_length: int,
) -> tuple[dict[str, float], float]:
    import numpy as np
    import torch

    from representax.evaluation import information_retrieval_metrics

    def encode(rows: Sequence[tuple[int, str]]) -> np.ndarray:
        outputs = []
        for start in range(0, len(rows), batch_size):
            texts = [text for _, text in rows[start : start + batch_size]]
            features = _pad_features(
                model.tokenize(texts),
                maximum_length,
                model.tokenizer.pad_token_id,
            )
            features = {
                name: (
                    value.to(model.device) if isinstance(value, torch.Tensor) else value
                )
                for name, value in features.items()
            }
            with torch.no_grad():
                output = model(features)["sentence_embedding"]
            outputs.append(output.float().cpu().numpy())
        return np.concatenate(outputs)

    started = time.perf_counter()
    was_training = model.training
    model.eval()
    queries = encode(data.queries)
    documents = encode(data.documents)
    model.train(was_training)
    queries /= np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    documents /= np.maximum(np.linalg.norm(documents, axis=1, keepdims=True), 1e-12)
    scores = queries @ documents.T
    ranked_indices = np.argsort(-scores, axis=1, kind="stable")[:, :100]
    document_ids = np.asarray([identifier for identifier, _ in data.documents])
    ranked_document_ids = document_ids[ranked_indices]
    raw = information_retrieval_metrics(
        ranked_document_ids,
        np.asarray([identifier for identifier, _ in data.queries]),
        data.relevant_documents,
        accuracy_at_k=(1, 3, 5, 10),
        precision_recall_at_k=(1, 3, 5, 10, 100),
        mrr_at_k=(10,),
        ndcg_at_k=(10,),
        map_at_k=(100,),
    )
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    metrics = {
        f"valid/{EVALUATION_SUBSET}/cosine_{name}": float(value)
        for name, value in raw.items()
    }
    return metrics, duration


def _representax_offline_metrics(
    model: Any,
    processor: Any,
    data: RetrievalEvaluationData,
    *,
    batch_size: int,
    iteration: int,
) -> dict[str, Any]:
    from experiments.preflights.dense_retrieval import (
        RetrievalEvaluationCollator,
        evaluation_rows,
    )

    from representax.config import PrecisionConfig
    from representax.evaluation import InformationRetrievalEvaluator
    from representax.precision import resolve_precision_policy
    from representax.train.evaluation import EvaluationRunner

    rows = evaluation_rows(data.queries, data.documents, batch_size=batch_size)
    collate = RetrievalEvaluationCollator(processor)
    batches = tuple(
        collate(rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    )
    evaluator = InformationRetrievalEvaluator(
        name=EVALUATION_SUBSET,
        relevant_documents=data.relevant_documents,
        score_functions=("cosine",),
        main_score_function="cosine",
        accuracy_at_k=(1, 3, 5, 10),
        precision_recall_at_k=(1, 3, 5, 10, 100),
        mrr_at_k=(10,),
        ndcg_at_k=(10,),
        map_at_k=(100,),
    )
    result = EvaluationRunner(
        evaluator,
        precision=resolve_precision_policy(PrecisionConfig.bfloat16_mixed()),
    ).run(model, batches, iteration=iteration)
    return {
        "metrics": {name: float(value) for name, value in result.metrics.items()},
        "examples": result.examples,
        "batches": result.batches,
        "duration_seconds": result.duration_seconds,
        "compilation_seconds": result.compilation_seconds,
    }


def _offline_evaluate(arguments: argparse.Namespace) -> None:
    import gc

    import jax

    artifact = arguments.artifact.expanduser().resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(f"exported artifact does not exist: {artifact}")
    if arguments.artifact_kind == "representax":
        from representax import load_inference_bundle
        from representax.train.job import load_model

        model, job = load_inference_bundle(artifact)
        initial_model, processor = load_model(
            job.model,
            key=jax.random.key(0),
            activation_rematerialization=job.training.activation_rematerialization,
        )
        del initial_model
        gc.collect()
    else:
        from representax.models import SentenceEncoder

        model, processor = SentenceEncoder.load_from_hf(
            artifact,
            local_files_only=True,
            parameter_dtype="float32",
            compute_dtype="bfloat16",
            sequence_length_buckets=(arguments.maximum_length,),
        )
    if processor is None:
        raise RuntimeError("exported dense model did not provide a processor")
    data = _evaluation_data(arguments.data_directory.expanduser().resolve())
    evaluation = _representax_offline_metrics(
        model,
        processor,
        data,
        batch_size=arguments.evaluation_batch_size,
        iteration=arguments.iteration,
    )
    result = {
        "schema_version": "representax-dense-retrieval-offline-evaluation-v1",
        "status": "accepted",
        "evaluated_by": "representax",
        "artifact_kind": arguments.artifact_kind,
        "artifact": str(artifact),
        "artifact_sha256": _directory_sha256(artifact),
        "data_manifest": str((arguments.data_directory / "manifest.json").resolve()),
        "data_manifest_sha256": _sha256(arguments.data_directory / "manifest.json"),
        "maximum_length": arguments.maximum_length,
        "evaluation_batch_size": arguments.evaluation_batch_size,
        "iteration": arguments.iteration,
        **evaluation,
    }
    _atomic_json(arguments.output.expanduser().resolve(), result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def _sentence_transformers_report(
    spec: ModelSpec,
    *,
    data_directory: Path,
    run_directory: Path,
    batch_size: int,
    steps: int,
    maximum_length: int,
    cache_chunk_size: int | None,
    evaluation_batch_size: int,
    evaluation_every_steps: int | None,
    data_threads: int,
    prefetch_buffer_size: int,
    persistent_workers: bool,
    torch_compile: bool,
    torch_compile_backend: str,
    fixed_query_length: int | None,
    fixed_document_length: int | None,
    seed: int,
    world_size: int,
    mixed_precision: bool,
    telemetry: bool,
    checkpoint_every: int | None,
    stop_after: int | None,
    resume: bool,
    export: bool,
) -> dict[str, Any] | None:
    import datasets
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.data_collator import (
        SentenceTransformerDataCollator,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )
    from transformers import TrainerCallback

    from benchmarks.samplers import sequential_sentence_transformers_batches

    if sentence_transformers.__version__ != ORACLE_VERSION:
        raise RuntimeError(f"expected sentence-transformers=={ORACLE_VERSION}")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"expected transformers=={TRANSFORMERS_VERSION}")
    if any(value is not None for value in (checkpoint_every, stop_after)):
        raise ValueError("reference preflight does not use checkpoint pause controls")
    if resume:
        raise ValueError("reference worker does not use Representax resume controls")
    if batch_size % world_size:
        raise ValueError("global batch size must be divisible by world size")
    process_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = process_world_size > 1
    if process_world_size != world_size:
        raise ValueError(
            f"launched world size {process_world_size} differs from {world_size}"
        )
    is_main = rank == 0
    if data_threads < 0:
        raise ValueError("Sentence Transformers data threads cannot be negative")
    if data_threads == 0 and persistent_workers:
        raise ValueError("persistent workers require at least one data thread")
    if data_threads > 0 and prefetch_buffer_size <= 0:
        raise ValueError("Sentence Transformers prefetch buffer must be positive")
    if (fixed_query_length is None) != (fixed_document_length is None):
        raise ValueError("fixed query and document lengths must be specified together")
    if fixed_query_length is not None and (
        fixed_query_length <= 0
        or fixed_document_length is None
        or fixed_document_length <= 0
        or max(fixed_query_length, fixed_document_length) > maximum_length
    ):
        raise ValueError("fixed text lengths must be within the maximum length")
    checkpoint = _checkpoint(spec)
    evaluation_data = _evaluation_data(data_directory)
    training_path = _artifact_path(
        data_directory,
        TRAIN_DATASET_ID,
        TRAIN_DATASET_FILE,
    )
    load_started = time.perf_counter()
    model = SentenceTransformer(str(checkpoint), local_files_only=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.max_seq_length = maximum_length
    required_rows = batch_size * steps
    training_data = datasets.Dataset(_training_table(training_path, required_rows))
    model_and_data_seconds = time.perf_counter() - load_started

    loss = (
        MultipleNegativesRankingLoss(model, scale=20.0)
        if cache_chunk_size is None
        else CachedMultipleNegativesRankingLoss(
            model,
            scale=20.0,
            mini_batch_size=cache_chunk_size,
            gather_across_devices=world_size > 1,
            show_progress_bar=False,
        )
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=batch_size // world_size,
        max_steps=steps,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.06,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=mixed_precision,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=data_threads,
        dataloader_prefetch_factor=(prefetch_buffer_size if data_threads > 0 else None),
        dataloader_persistent_workers=persistent_workers,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=True,
        torch_compile=torch_compile,
        torch_compile_backend=(torch_compile_backend if torch_compile else None),
        batch_sampler=sequential_sentence_transformers_batches,
        seed=seed,
        data_seed=seed,
    )

    class RecordingTrainer(SentenceTransformerTrainer):
        final_train_loss: Any = None

        def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
            result = super().compute_loss(*args, **kwargs)
            loss = result[0] if isinstance(result, tuple) else result
            self.final_train_loss = loss.detach()
            return result

    initial, initial_seconds = _oracle_retrieval_metrics(
        model,
        evaluation_data,
        evaluation_batch_size,
        maximum_length,
    )
    evaluation_history = [
        {
            "updates": 0,
            "metrics": initial,
            "final_train_loss": None,
            "evaluation_seconds": initial_seconds,
            "training_elapsed_seconds": 0.0,
        }
    ]
    training_started_at: float | None = None
    evaluation_during_training_seconds = 0.0

    class PeriodicRetrievalCallback(TrainerCallback):
        trainer: RecordingTrainer | None = None

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            nonlocal evaluation_during_training_seconds
            if evaluation_every_steps is None:
                return control
            updates = int(state.global_step)
            if updates % evaluation_every_steps != 0 and updates != steps:
                return control
            if self.trainer is None or self.trainer.final_train_loss is None:
                raise AssertionError("trainer loss is unavailable at evaluation")
            if training_started_at is None:
                raise AssertionError("training timer is unavailable at evaluation")
            elapsed = (
                time.perf_counter()
                - training_started_at
                - evaluation_during_training_seconds
            )
            if is_main:
                was_training = model.training
                metrics, duration = _oracle_retrieval_metrics(
                    model,
                    evaluation_data,
                    evaluation_batch_size,
                    maximum_length,
                )
                model.train(was_training)
                evaluation_during_training_seconds += duration
                evaluation_history.append(
                    {
                        "updates": updates,
                        "metrics": metrics,
                        "final_train_loss": float(self.trainer.final_train_loss),
                        "evaluation_seconds": duration,
                        "training_elapsed_seconds": elapsed,
                    }
                )
            if distributed:
                torch.distributed.barrier()
            return control

    callback = PeriodicRetrievalCallback()
    data_collator = SentenceTransformerDataCollator(
        preprocess_fn=model.tokenize,
        valid_label_columns=[],
    )
    if fixed_query_length is not None:
        import torch.nn.functional as functional

        dynamic_collator = data_collator
        pad_token_id = int(model.tokenizer.pad_token_id or 0)

        def fixed_shape_collator(features: list[dict[str, Any]]) -> dict[str, Any]:
            batch = dynamic_collator(features)
            for column, target_length in (
                ("query", fixed_query_length),
                ("positive", fixed_document_length),
            ):
                assert target_length is not None
                for suffix in ("input_ids", "attention_mask", "token_type_ids"):
                    name = f"{column}_{suffix}"
                    if name not in batch:
                        continue
                    value = batch[name]
                    padding = target_length - int(value.shape[1])
                    if padding < 0:
                        raise ValueError(
                            f"{column} length {value.shape[1]} exceeds fixed "
                            f"length {target_length}"
                        )
                    if padding:
                        batch[name] = functional.pad(
                            value,
                            (0, padding),
                            value=(pad_token_id if suffix == "input_ids" else 0),
                        )
            return batch

        # SentenceTransformers reads collator metadata while constructing the loader.
        fixed_shape_collator.__dict__.update(vars(dynamic_collator))
        data_collator = fixed_shape_collator

    trainer = RecordingTrainer(
        model=model,
        args=arguments,
        train_dataset=training_data,
        loss=loss,
        data_collator=data_collator,
    )
    callback.trainer = trainer
    metrics_stream = None
    accelerator_monitor = None
    if telemetry and is_main:
        import threading

        from representax.train.accelerator import AcceleratorMonitor

        run_directory.mkdir(parents=True, exist_ok=True)
        metrics_stream = (run_directory / "metrics.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        metrics_lock = threading.Lock()
        current_iteration = {"value": 0}

        def publish_reference_metric(
            event: str,
            iteration: int,
            values: Mapping[str, float | int],
        ) -> None:
            if metrics_stream is None:  # pragma: no cover - closure invariant
                raise AssertionError("reference metrics stream is unavailable")
            row = {
                "schema_version": "representax-reference-metrics-v1",
                "timestamp": datetime.now().astimezone().isoformat(),
                "event": event,
                "iteration": iteration,
                "metrics": dict(values),
            }
            with metrics_lock:
                metrics_stream.write(json.dumps(row, sort_keys=True) + "\n")

        publish_reference_metric(
            "startup",
            0,
            {"perf/model_and_data_seconds": model_and_data_seconds},
        )

        class ReferenceTimingCallback(TrainerCallback):
            started: float | None = None

            def on_train_begin(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> Any:
                torch.cuda.synchronize()
                self.started = time.perf_counter()
                return control

            def on_step_end(
                self, args: Any, state: Any, control: Any, **kwargs: Any
            ) -> Any:
                if self.started is None:
                    raise AssertionError("reference step timer did not start")
                iteration = int(state.global_step)
                torch.cuda.synchronize()
                completed_at = time.perf_counter()
                duration = completed_at - self.started
                self.started = completed_at
                current_iteration["value"] = iteration
                if trainer.final_train_loss is None:
                    raise AssertionError("reference trainer loss is unavailable")
                publish_reference_metric(
                    "training_step",
                    iteration,
                    {
                        "perf/examples": batch_size,
                        "perf/step_seconds": duration,
                        "perf/examples_per_second": batch_size / duration,
                        "train/loss": float(trainer.final_train_loss),
                    },
                )
                return control

        trainer.add_callback(ReferenceTimingCallback())
        accelerator_monitor = AcceleratorMonitor(
            lambda values: publish_reference_metric(
                "accelerator", current_iteration["value"], values
            )
        )
        accelerator_monitor.start()
    trainer.add_callback(callback)
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    training_started_at = training_started
    try:
        output = trainer.train()
        torch.cuda.synchronize()
    finally:
        if accelerator_monitor is not None:
            accelerator_monitor.close()
        if metrics_stream is not None:
            metrics_stream.close()
    training_seconds = time.perf_counter() - training_started
    if trainer.final_train_loss is None:  # pragma: no cover - trainer invariant
        raise AssertionError("Sentence Transformers did not compute a training loss")
    final_train_loss = float(trainer.final_train_loss)
    training_compute_seconds = training_seconds - evaluation_during_training_seconds
    if not is_main:
        return None
    reference_steady_state = (
        _reference_steady_state_summary(run_directory) if telemetry else None
    )
    if evaluation_every_steps is None:
        final, final_seconds = _oracle_retrieval_metrics(
            model,
            evaluation_data,
            evaluation_batch_size,
            maximum_length,
        )
        evaluation_history = []
    else:
        if evaluation_history[-1]["updates"] != steps:
            raise AssertionError("final periodic retrieval evaluation is missing")
        final = evaluation_history[-1]["metrics"]
        final_seconds = evaluation_history[-1]["evaluation_seconds"]
    exported_checkpoint = None
    export_seconds = 0.0
    if export:
        export_started = time.perf_counter()
        export_path = run_directory / "final-model"
        model.save_pretrained(str(export_path), safe_serialization=True)
        torch.cuda.synchronize()
        export_seconds = time.perf_counter() - export_started
        exported_checkpoint = str(export_path.resolve())
    return {
        "schema_version": "representax-dense-retrieval-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "parameter_count": parameter_count,
        "batch_size": batch_size,
        "steps": steps,
        "maximum_length": maximum_length,
        "cache_chunk_size": cache_chunk_size,
        "data_threads": data_threads,
        "prefetch_buffer_size": (prefetch_buffer_size if data_threads > 0 else None),
        "persistent_workers": persistent_workers,
        "torch_compile": torch_compile,
        "torch_compile_backend": (torch_compile_backend if torch_compile else None),
        "fixed_query_length": fixed_query_length,
        "fixed_document_length": fixed_document_length,
        "optimizer_implementation": str(arguments.optim),
        "seed": seed,
        "world_size": world_size,
        "mixed_precision": mixed_precision,
        "exported_checkpoint": exported_checkpoint,
        "export_seconds": export_seconds,
        "model_and_data_seconds": model_and_data_seconds,
        "training_seconds": training_seconds,
        "training_compute_seconds": training_compute_seconds,
        "amortized_examples_per_second": (
            batch_size * steps / training_compute_seconds
        ),
        "steady_state_seconds": (
            None if reference_steady_state is None else reference_steady_state[0]
        ),
        "steady_state_step_count": (
            None if reference_steady_state is None else reference_steady_state[1]
        ),
        "steady_state_examples": (
            None if reference_steady_state is None else reference_steady_state[2]
        ),
        "steady_state_examples_per_second": (
            None
            if reference_steady_state is None
            else reference_steady_state[2] / reference_steady_state[0]
        ),
        "trainer_metrics": output.metrics,
        "final_train_loss": final_train_loss,
        "initial_evaluation": initial,
        "final_evaluation": final,
        "initial_evaluation_seconds": initial_seconds,
        "final_evaluation_seconds": final_seconds,
        "evaluation_history": evaluation_history,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_host_rss_bytes": _maximum_host_rss_bytes(),
    }


def _worker(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
    kwargs = {
        "data_directory": arguments.data_directory.resolve(),
        "run_directory": arguments.run_directory.resolve(),
        "batch_size": arguments.batch_size,
        "steps": arguments.steps,
        "maximum_length": arguments.maximum_length,
        "cache_chunk_size": _framework_cache_chunk_size(arguments, arguments.framework),
        "evaluation_batch_size": arguments.evaluation_batch_size,
        "evaluation_every_steps": arguments.evaluation_every_steps,
        "seed": arguments.seed,
        "world_size": arguments.world_size,
        "mixed_precision": arguments.mixed_precision,
        "telemetry": arguments.telemetry,
        "checkpoint_every": arguments.checkpoint_every,
        "stop_after": arguments.stop_after,
        "resume": arguments.resume,
        "export": arguments.export,
    }
    if arguments.framework == "representax":
        packing_query_shape = (
            None
            if arguments.packing_query_shape is None
            else tuple(arguments.packing_query_shape)
        )
        packing_document_shape = (
            None
            if arguments.packing_document_shape is None
            else tuple(arguments.packing_document_shape)
        )
        if (packing_query_shape is None) != (packing_document_shape is None):
            raise ValueError(
                "packing query and document shapes must be specified together"
            )
        if packing_query_shape is not None and not arguments.packing:
            raise ValueError("fixed packing shapes require --packing")
        buckets = tuple(arguments.sequence_length_bucket or (arguments.maximum_length,))
        report = _representax_report(
            spec,
            sequence_length_buckets=buckets,
            query_cache_chunk_size=arguments.representax_query_cache_chunk_size,
            document_cache_chunk_size=(
                arguments.representax_document_cache_chunk_size
            ),
            loss_row_chunk_size=arguments.representax_loss_row_chunk_size,
            grad_cache_implementation=arguments.grad_cache_implementation,
            packing=arguments.packing,
            packing_query_shape=packing_query_shape,
            packing_document_shape=packing_document_shape,
            data_threads=arguments.data_threads,
            prefetch_buffer_size=arguments.prefetch_buffer_size,
            **kwargs,
        )
    else:
        if arguments.packing:
            raise ValueError("packing is a Representax worker option")
        if arguments.sequence_length_bucket:
            raise ValueError("sequence length buckets are a Representax worker option")
        report = _sentence_transformers_report(
            spec,
            data_threads=arguments.sentence_transformers_data_threads,
            prefetch_buffer_size=(arguments.sentence_transformers_prefetch_buffer_size),
            persistent_workers=(arguments.sentence_transformers_persistent_workers),
            torch_compile=arguments.sentence_transformers_torch_compile,
            torch_compile_backend=(
                arguments.sentence_transformers_torch_compile_backend
            ),
            fixed_query_length=arguments.sentence_transformers_query_length,
            fixed_document_length=arguments.sentence_transformers_document_length,
            **kwargs,
        )
    if report is not None:
        if arguments.framework == "representax":
            _atomic_json(arguments.report, report)
        else:
            write_reference_result(
                arguments.report,
                report,
                reference="sentence-transformers",
            )


def _device_stats(index: int) -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = completed.stdout.strip().splitlines()[0].split(",")
    return int(values[0].strip()), int(values[1].strip())


@dataclass(slots=True)
class _RunningProcess:
    process: subprocess.Popen[Any]
    stream: TextIO
    started: float
    gpu: int
    log: str
    peak_device_mib: int = 0
    utilization_samples: list[int] = field(default_factory=list)
    finished: float | None = None


def _run_processes(
    commands: Sequence[tuple[str, int, list[str], Path]],
) -> dict[str, Any]:
    running: dict[str, _RunningProcess] = {}
    for name, gpu, command, log_path in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("x", encoding="utf-8")
        environment = dict(os.environ)
        environment.pop("LD_LIBRARY_PATH", None)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_COMPILATION_CACHE_DIR": os.environ.get(
                    "REPRESENTAX_JAX_CACHE_DIR",
                    str(log_path.parent.parent / "jax-cache"),
                ),
                "TORCHINDUCTOR_CACHE_DIR": os.environ.get(
                    "REPRESENTAX_TORCHINDUCTOR_CACHE_DIR",
                    str(log_path.parent.parent / "torchinductor-cache"),
                ),
            }
        )
        if name == "representax":
            environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
            environment.update(
                {
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                }
            )
        started = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        running[name] = _RunningProcess(
            process=process,
            stream=stream,
            started=started,
            gpu=gpu,
            log=str(log_path),
        )
    while any(item.finished is None for item in running.values()):
        for item in running.values():
            if item.finished is not None:
                continue
            if item.process.poll() is None:
                memory_mib, utilization = _device_stats(item.gpu)
                item.peak_device_mib = max(
                    item.peak_device_mib,
                    memory_mib,
                )
                item.utilization_samples.append(utilization)
            else:
                item.finished = time.perf_counter()
        time.sleep(0.2)
    reports = {}
    for name, item in running.items():
        item.stream.close()
        return_code = item.process.returncode
        if item.finished is None:  # pragma: no cover - loop invariant
            raise AssertionError("process result is missing its finish time")
        reports[name] = {
            "return_code": return_code,
            "process_wall_seconds": item.finished - item.started,
            "process_peak_device_bytes": item.peak_device_mib * 1024 * 1024,
            "process_mean_gpu_utilization_percent": (
                statistics.mean(item.utilization_samples)
                if item.utilization_samples
                else 0.0
            ),
            "process_peak_gpu_utilization_percent": max(
                item.utilization_samples,
                default=0,
            ),
            "log": item.log,
        }
        if return_code:
            raise RuntimeError(
                f"{name} failed with exit code {return_code}; see {item.log}"
            )
    return reports


def _initial_metric_parity(
    native: Mapping[str, float],
    oracle: Mapping[str, float],
    *,
    tolerance: float,
) -> float:
    if native.keys() != oracle.keys():
        raise ValueError(
            "initial retrieval metric names differ: "
            f"native={sorted(native)}, oracle={sorted(oracle)}"
        )
    differences = {
        name: abs(native[name] - oracle[name])
        for name in native
        if abs(native[name] - oracle[name]) > tolerance
    }
    if differences:
        raise ValueError(f"initial retrieval metrics differ: {differences}")
    return max(abs(native[name] - oracle[name]) for name in native)


def _three_run_interval(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError("the accepted aggregate requires exactly three runs")
    mean = statistics.mean(values)
    half_width = (
        THREE_RUN_95_PERCENT_T_CRITICAL * statistics.stdev(values) / len(values) ** 0.5
    )
    return {
        "values": list(values),
        "mean": mean,
        "confidence_level": 0.95,
        "confidence_interval": [mean - half_width, mean + half_width],
        "minimum": min(values),
        "maximum": max(values),
    }


def _aggregate(arguments: argparse.Namespace) -> None:
    paths = [path.expanduser().resolve() for path in arguments.summary]
    if len(paths) != 3:
        raise ValueError("provide exactly three paired summary files")
    summaries = [json.loads(path.read_text()) for path in paths]
    contracts = []
    source_commits = []
    seeds = []
    for summary in summaries:
        if summary.get("schema_version") != "representax-dense-retrieval-comparison-v1":
            raise ValueError("aggregate input is not a dense-retrieval comparison")
        contract = dict(summary["contract"])
        seeds.append(int(contract.pop("seed")))
        source_commits.append(dict(contract.pop("source_commits", {})))
        contracts.append(contract)
        if not summary["comparison"]["initial_metric_parity"]:
            raise ValueError("aggregate input did not pass initial metric parity")
    if len(set(seeds)) != 3:
        raise ValueError(f"aggregate seeds must be unique: {seeds}")
    if contracts[1:] != contracts[:-1]:
        raise ValueError("aggregate scientific and execution contracts differ")

    ordered = sorted(zip(seeds, paths, summaries, source_commits, strict=True))
    metrics = {
        "sustained_training_speedup": lambda row: row["comparison"][
            "sustained_training_speedup"
        ],
        "representax_sustained_examples_per_second": lambda row: row["comparison"][
            "representax_sustained_examples_per_second"
        ],
        "sentence_transformers_sustained_examples_per_second": lambda row: row[
            "comparison"
        ]["sentence_transformers_sustained_examples_per_second"],
        "amortized_training_speedup": lambda row: row["comparison"][
            "amortized_training_speedup"
        ],
        "representax_amortized_examples_per_second": lambda row: row["representax"][
            "amortized_examples_per_second"
        ],
        "sentence_transformers_amortized_examples_per_second": lambda row: row[
            "sentence_transformers"
        ]["amortized_examples_per_second"],
        "representax_final_ndcg@10": lambda row: row["comparison"][
            "representax_final_ndcg@10"
        ],
        "sentence_transformers_final_ndcg@10": lambda row: row["comparison"][
            "sentence_transformers_final_ndcg@10"
        ],
        "final_ndcg@10_difference": lambda row: row["comparison"][
            "final_ndcg@10_difference"
        ],
        "compilation_break_even_steps": lambda row: row["comparison"][
            "compilation_break_even_steps"
        ],
    }
    aggregated_metrics = {}
    for name, select in metrics.items():
        values = [select(summary) for _, _, summary, _ in ordered]
        available = [value is not None for value in values]
        if not any(available):
            continue
        if not all(available):
            raise ValueError(f"aggregate metric availability differs: {name}")
        aggregated_metrics[name] = _three_run_interval(
            [float(value) for value in values]
        )

    aggregate = {
        "schema_version": "representax-dense-retrieval-three-run-aggregate-v1",
        "contract": contracts[0],
        "seeds": [seed for seed, _, _, _ in ordered],
        "inputs": [
            {
                "seed": seed,
                "path": str(path),
                "sha256": _sha256(path),
                "source_commits": commits,
            }
            for seed, path, _, commits in ordered
        ],
        "all_initial_metrics_match": True,
        "metrics": aggregated_metrics,
    }
    _atomic_json(arguments.output.expanduser().resolve(), aggregate)
    print(json.dumps(aggregate["metrics"], indent=2, sort_keys=True))


def _shared_worker_flags(arguments: argparse.Namespace) -> list[str]:
    flags = []
    if arguments.mixed_precision:
        flags.append("--mixed-precision")
    if arguments.telemetry:
        flags.append("--telemetry")
    if getattr(arguments, "export", False):
        flags.append("--export")
    return flags


def _framework_cache_chunk_size(
    arguments: argparse.Namespace,
    framework: str,
) -> int | None:
    override = getattr(
        arguments,
        framework.replace("-", "_") + "_cache_chunk_size",
        None,
    )
    return arguments.cache_chunk_size if override is None else override


def _representax_worker_flags(arguments: argparse.Namespace) -> list[str]:
    flags = [
        flag
        for bucket in arguments.sequence_length_bucket or ()
        for flag in ("--sequence-length-bucket", str(bucket))
    ]
    implementation = getattr(arguments, "grad_cache_implementation", "rematerialized")
    if implementation != "rematerialized":
        flags.extend(("--grad-cache-implementation", implementation))
    for name in (
        "query_cache_chunk_size",
        "document_cache_chunk_size",
        "loss_row_chunk_size",
    ):
        value = getattr(arguments, f"representax_{name}", None)
        if value is not None:
            flags.extend((f"--representax-{name.replace('_', '-')}", str(value)))
    return flags


def _sentence_transformers_worker_flags(arguments: argparse.Namespace) -> list[str]:
    flags = [
        "--sentence-transformers-data-threads",
        str(arguments.sentence_transformers_data_threads),
        "--sentence-transformers-prefetch-buffer-size",
        str(arguments.sentence_transformers_prefetch_buffer_size),
        "--sentence-transformers-torch-compile-backend",
        arguments.sentence_transformers_torch_compile_backend,
    ]
    if arguments.sentence_transformers_persistent_workers:
        flags.append("--sentence-transformers-persistent-workers")
    if arguments.sentence_transformers_torch_compile:
        flags.append("--sentence-transformers-torch-compile")
    query_length = getattr(arguments, "sentence_transformers_query_length", None)
    document_length = getattr(
        arguments,
        "sentence_transformers_document_length",
        None,
    )
    if query_length is not None:
        flags.extend(
            (
                "--sentence-transformers-query-length",
                str(query_length),
                "--sentence-transformers-document-length",
                str(document_length),
            )
        )
    return flags


def _pair(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
    source_commits = {
        "representax": _git_head(),
        "sentence-transformers": reference_source("sentence-transformers").commit,
    }
    if arguments.world_size != 1:
        raise ValueError(
            "pair launches one isolated GPU per framework; use workers for "
            "distributed comparisons"
        )
    if (
        any(
            value is not None
            for value in (arguments.checkpoint_every, arguments.stop_after)
        )
        or arguments.resume
    ):
        raise ValueError(
            "pair does not apply Representax-only checkpoint or resume controls"
        )
    result_directory = arguments.result_directory.expanduser().resolve()
    result_directory.mkdir(parents=True, exist_ok=False)
    reports = {
        name: result_directory / f"{name}.json"
        for name in ("representax", "sentence-transformers")
    }
    common = [
        "--model",
        spec.name,
        "--checkpoint",
        str(_checkpoint(spec)),
        "--data-directory",
        str(arguments.data_directory.resolve()),
        "--batch-size",
        str(arguments.batch_size),
        "--steps",
        str(arguments.steps),
        "--maximum-length",
        str(arguments.maximum_length),
        "--evaluation-batch-size",
        str(arguments.evaluation_batch_size),
        "--data-threads",
        str(arguments.data_threads),
        "--prefetch-buffer-size",
        str(arguments.prefetch_buffer_size),
        "--seed",
        str(arguments.seed),
    ]
    if arguments.evaluation_every_steps is not None:
        common.extend(
            ("--evaluation-every-steps", str(arguments.evaluation_every_steps))
        )
    if arguments.cache_chunk_size is not None:
        common.extend(("--cache-chunk-size", str(arguments.cache_chunk_size)))
    if arguments.representax_cache_chunk_size is not None:
        common.extend(
            (
                "--representax-cache-chunk-size",
                str(arguments.representax_cache_chunk_size),
            )
        )
    if arguments.sentence_transformers_cache_chunk_size is not None:
        common.extend(
            (
                "--sentence-transformers-cache-chunk-size",
                str(arguments.sentence_transformers_cache_chunk_size),
            )
        )
    common.extend(_sentence_transformers_worker_flags(arguments))
    common.extend(_shared_worker_flags(arguments))
    representax_only = _representax_worker_flags(arguments)
    commands = [
        (
            name,
            gpu,
            [
                sys.executable,
                "-m",
                "benchmarks.dense_retrieval",
                "worker",
                "--framework",
                name,
                *common,
                *(representax_only if name == "representax" else ()),
                "--run-directory",
                str(result_directory / "runs" / name),
                "--report",
                str(reports[name]),
            ],
            result_directory / "logs" / f"{name}.log",
        )
        for name, gpu in (
            ("representax", arguments.representax_gpu),
            ("sentence-transformers", arguments.sentence_transformers_gpu),
        )
    ]
    process = _run_processes(commands)
    native = json.loads(reports["representax"].read_text())
    oracle = json.loads(reports["sentence-transformers"].read_text())
    if native["parameter_count"] != oracle["parameter_count"]:
        raise RuntimeError(
            "paired model parameter counts differ: "
            f"Representax={native['parameter_count']}, "
            f"Sentence Transformers={oracle['parameter_count']}"
        )
    if native["mixed_precision"] != oracle["mixed_precision"]:
        raise ValueError("paired workers used different precision policies")
    maximum_initial_metric_difference = _initial_metric_parity(
        native["initial_evaluation"],
        oracle["initial_evaluation"],
        tolerance=spec.initial_metric_tolerance,
    )
    metric = f"valid/{EVALUATION_SUBSET}/cosine_ndcg@10"
    shared_offline_evaluation = None
    final_native_metrics = native["final_evaluation"]
    final_oracle_metrics = oracle["final_evaluation"]
    if arguments.export:
        artifacts = {
            "representax": native["exported_bundle"],
            "sentence-transformers": oracle["exported_checkpoint"],
        }
        if any(value is None for value in artifacts.values()):
            raise RuntimeError("paired export did not publish both final artifacts")
        offline_directory = result_directory / "offline-evaluation"
        offline_reports = {
            name: offline_directory / f"{name}.json" for name in artifacts
        }
        offline_commands = [
            (
                f"{name}-offline",
                gpu,
                [
                    sys.executable,
                    "-m",
                    "benchmarks.dense_retrieval",
                    "offline-evaluate",
                    "--artifact-kind",
                    name,
                    "--artifact",
                    str(artifacts[name]),
                    "--data-directory",
                    str(arguments.data_directory.resolve()),
                    "--maximum-length",
                    str(arguments.maximum_length),
                    "--evaluation-batch-size",
                    str(arguments.evaluation_batch_size),
                    "--iteration",
                    str(arguments.steps),
                    "--output",
                    str(offline_reports[name]),
                ],
                result_directory / "logs" / f"{name}-offline.log",
            )
            for name, gpu in (
                ("representax", arguments.representax_gpu),
                ("sentence-transformers", arguments.sentence_transformers_gpu),
            )
        ]
        offline_processes = _run_processes(offline_commands)
        offline = {
            name: json.loads(path.read_text()) for name, path in offline_reports.items()
        }
        final_native_metrics = offline["representax"]["metrics"]
        final_oracle_metrics = offline["sentence-transformers"]["metrics"]
        if final_native_metrics.keys() != final_oracle_metrics.keys():
            raise RuntimeError(
                "shared offline evaluator emitted different metric names"
            )
        shared_offline_evaluation = {
            "quality_denominator": "representax-offline-evaluator",
            "representax": {
                **offline["representax"],
                **offline_processes["representax-offline"],
            },
            "sentence_transformers": {
                **offline["sentence-transformers"],
                **offline_processes["sentence-transformers-offline"],
            },
        }
    (
        native_steady_seconds,
        oracle_steady_seconds,
        paired_steady_steps,
        paired_steady_examples,
        paired_steady_start,
        paired_steady_end,
    ) = _paired_steady_state_summary(
        result_directory / "runs" / "representax",
        result_directory / "runs" / "sentence-transformers",
    )
    native_sustained_rate = paired_steady_examples / native_steady_seconds
    oracle_rate = paired_steady_examples / oracle_steady_seconds
    native_step_seconds = native_steady_seconds / paired_steady_steps
    oracle_step_seconds = oracle_steady_seconds / paired_steady_steps
    seconds_saved_per_step = oracle_step_seconds - native_step_seconds
    compilation_break_even_steps = (
        native["compilation_and_first_use_seconds"] / seconds_saved_per_step
        if seconds_saved_per_step > 0
        else None
    )
    summary = {
        "schema_version": "representax-dense-retrieval-comparison-v1",
        "contract": {
            "source_commits": source_commits,
            "model": spec.name,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "checkpoint_model_sha256": _sha256(_checkpoint(spec) / "model.safetensors"),
            "data": json.loads(
                (arguments.data_directory / "manifest.json").read_text()
            ),
            "batch_size": arguments.batch_size,
            "steps": arguments.steps,
            "maximum_length": arguments.maximum_length,
            "cache_chunk_size": {
                "representax": native["cache_chunk_size"],
                "sentence_transformers": oracle["cache_chunk_size"],
            },
            "seed": arguments.seed,
            "evaluation_every_steps": arguments.evaluation_every_steps,
            "representax_data_threads": arguments.data_threads,
            "representax_prefetch_buffer_size": arguments.prefetch_buffer_size,
            "sentence_transformers_data_threads": oracle["data_threads"],
            "sentence_transformers_prefetch_buffer_size": oracle[
                "prefetch_buffer_size"
            ],
            "sentence_transformers_persistent_workers": oracle["persistent_workers"],
            "sentence_transformers_torch_compile": oracle["torch_compile"],
            "sentence_transformers_torch_compile_backend": oracle[
                "torch_compile_backend"
            ],
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "precision": (
                "bfloat16-compute-float32-master"
                if native["mixed_precision"]
                else "float32"
            ),
            "loss": "cosine MNR scale=20 asymmetric",
            "initial_metric_absolute_tolerance": spec.initial_metric_tolerance,
            "parameter_count": native["parameter_count"],
        },
        "representax": {**native, **process["representax"]},
        "sentence_transformers": {
            **oracle,
            **process["sentence-transformers"],
        },
        "shared_offline_evaluation": shared_offline_evaluation,
        "comparison": {
            "amortized_training_speedup": (
                native["amortized_examples_per_second"]
                / oracle["amortized_examples_per_second"]
            ),
            "sustained_training_speedup": native_sustained_rate / oracle_rate,
            "representax_sustained_examples_per_second": native_sustained_rate,
            "sentence_transformers_sustained_examples_per_second": oracle_rate,
            "paired_steady_state_steps": paired_steady_steps,
            "paired_steady_state_iterations": [
                paired_steady_start,
                paired_steady_end,
            ],
            "compilation_break_even_steps": compilation_break_even_steps,
            "initial_metric_parity": True,
            "parameter_count_match": True,
            "maximum_initial_metric_absolute_difference": (
                maximum_initial_metric_difference
            ),
            "quality_denominator": (
                "representax-offline-evaluator"
                if shared_offline_evaluation is not None
                else "framework-worker-diagnostic"
            ),
            "representax_final_ndcg@10": final_native_metrics[metric],
            "sentence_transformers_final_ndcg@10": final_oracle_metrics[metric],
            "final_ndcg@10_difference": (
                final_native_metrics[metric] - final_oracle_metrics[metric]
            ),
        },
    }
    _atomic_json(result_directory / "summary.json", summary)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


def _time_to_quality_summary(
    summary: Mapping[str, Any],
    *,
    quality_target: float,
) -> dict[str, Any]:
    if summary.get("schema_version") != "representax-dense-retrieval-comparison-v1":
        raise ValueError("time-to-quality input is not a paired comparison")
    metric = f"valid/{EVALUATION_SUBSET}/cosine_ndcg@10"
    native_history = summary["representax"]["evaluation_history"]
    oracle_history = summary["sentence_transformers"]["evaluation_history"]
    if not native_history or not oracle_history:
        raise ValueError("time-to-quality requires in-trajectory evaluation history")
    native_updates = [int(item["updates"]) for item in native_history]
    oracle_updates = [int(item["updates"]) for item in oracle_history]
    if native_updates != oracle_updates:
        raise ValueError(
            "time-to-quality evaluation checkpoints differ: "
            f"native={native_updates}, oracle={oracle_updates}"
        )
    points: list[dict[str, Any]] = []
    for native, oracle in zip(native_history, oracle_history, strict=True):
        updates = int(native["updates"])
        points.append(
            {
                "updates": updates,
                "representax": {
                    "ndcg@10": float(native["metrics"][metric]),
                    "train_loss": native["final_train_loss"],
                    "training_seconds": float(native["training_elapsed_seconds"]),
                },
                "sentence_transformers": {
                    "ndcg@10": float(oracle["metrics"][metric]),
                    "train_loss": oracle["final_train_loss"],
                    "training_seconds": float(oracle["training_elapsed_seconds"]),
                },
            }
        )

    def first_crossing(framework: str) -> dict[str, float | int] | None:
        for point in points:
            values = point[framework]
            if values["ndcg@10"] >= quality_target:
                return {
                    "updates": point["updates"],
                    "training_seconds": values["training_seconds"],
                    "ndcg@10": values["ndcg@10"],
                }
        return None

    return {
        "schema_version": "representax-dense-retrieval-time-to-quality-v1",
        "contract": summary["contract"],
        "quality_metric": metric,
        "quality_target": quality_target,
        "first_observed_crossing": {
            "representax": first_crossing("representax"),
            "sentence_transformers": first_crossing("sentence_transformers"),
        },
        "points": points,
    }


def _curve(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
    result_directory = arguments.result_directory.expanduser().resolve()
    result_directory.mkdir(parents=True, exist_ok=False)
    checkpoints = sorted(set(arguments.checkpoint_step))
    if len(checkpoints) != len(arguments.checkpoint_step) or any(
        value <= 0 for value in checkpoints
    ):
        raise ValueError("checkpoint steps must be unique positive integers")
    interval = checkpoints[0]
    maximum = checkpoints[-1]
    if checkpoints != list(range(interval, maximum + 1, interval)):
        raise ValueError(
            "in-trajectory checkpoints must be uniform multiples of the first step"
        )
    pair_directory = result_directory / "pair"
    command = [
        sys.executable,
        "-m",
        "benchmarks.dense_retrieval",
        "pair",
        "--model",
        spec.name,
        "--checkpoint",
        str(_checkpoint(spec)),
        "--data-directory",
        str(arguments.data_directory.resolve()),
        "--batch-size",
        str(arguments.batch_size),
        "--steps",
        str(maximum),
        "--maximum-length",
        str(arguments.maximum_length),
        "--evaluation-batch-size",
        str(arguments.evaluation_batch_size),
        "--evaluation-every-steps",
        str(interval),
        "--data-threads",
        str(arguments.data_threads),
        "--prefetch-buffer-size",
        str(arguments.prefetch_buffer_size),
        "--seed",
        str(arguments.seed),
        "--result-directory",
        str(pair_directory),
        "--representax-gpu",
        str(arguments.representax_gpu),
        "--sentence-transformers-gpu",
        str(arguments.sentence_transformers_gpu),
    ]
    if arguments.cache_chunk_size is not None:
        command.extend(("--cache-chunk-size", str(arguments.cache_chunk_size)))
    if arguments.representax_cache_chunk_size is not None:
        command.extend(
            (
                "--representax-cache-chunk-size",
                str(arguments.representax_cache_chunk_size),
            )
        )
    if arguments.sentence_transformers_cache_chunk_size is not None:
        command.extend(
            (
                "--sentence-transformers-cache-chunk-size",
                str(arguments.sentence_transformers_cache_chunk_size),
            )
        )
    command.extend(_representax_worker_flags(arguments))
    command.extend(_sentence_transformers_worker_flags(arguments))
    command.extend(_shared_worker_flags(arguments))
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    summary = json.loads((pair_directory / "summary.json").read_text())
    result = _time_to_quality_summary(
        summary,
        quality_target=arguments.quality_target,
    )
    _atomic_json(result_directory / "time-to-quality.json", result)
    print(json.dumps(result["first_observed_crossing"], indent=2, sort_keys=True))


def _worker_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_steps: bool = True,
) -> None:
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    if include_steps:
        parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--maximum-length", type=int, default=128)
    parser.add_argument("--sequence-length-bucket", type=int, action="append")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--packing-query-shape", type=int, nargs=2)
    parser.add_argument("--packing-document-shape", type=int, nargs=2)
    parser.add_argument("--cache-chunk-size", type=int)
    parser.add_argument("--representax-cache-chunk-size", type=int)
    parser.add_argument("--sentence-transformers-cache-chunk-size", type=int)
    parser.add_argument("--representax-query-cache-chunk-size", type=int)
    parser.add_argument("--representax-document-cache-chunk-size", type=int)
    parser.add_argument("--representax-loss-row-chunk-size", type=int)
    parser.add_argument(
        "--grad-cache-implementation",
        choices=("rematerialized", "custom_vjp"),
        default="rematerialized",
        help="Representax GradCache backward schedule",
    )
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-every-steps", type=int)
    parser.add_argument("--data-threads", type=int, default=4)
    parser.add_argument("--prefetch-buffer-size", type=int, default=8)
    parser.add_argument("--sentence-transformers-data-threads", type=int, default=0)
    parser.add_argument(
        "--sentence-transformers-prefetch-buffer-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--sentence-transformers-persistent-workers",
        action="store_true",
    )
    parser.add_argument(
        "--sentence-transformers-torch-compile",
        action="store_true",
    )
    parser.add_argument(
        "--sentence-transformers-torch-compile-backend",
        default="inductor",
    )
    parser.add_argument("--sentence-transformers-query-length", type=int)
    parser.add_argument("--sentence-transformers-document-length", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data-directory", type=Path, required=True)
    worker = commands.add_parser("worker")
    _worker_arguments(worker)
    worker.add_argument(
        "--framework",
        choices=("representax", "sentence-transformers"),
        required=True,
    )
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    pair = commands.add_parser("pair")
    _worker_arguments(pair)
    pair.add_argument("--result-directory", type=Path, required=True)
    pair.add_argument("--representax-gpu", type=int, default=4)
    pair.add_argument("--sentence-transformers-gpu", type=int, default=5)
    curve = commands.add_parser("curve")
    _worker_arguments(curve, include_steps=False)
    curve.add_argument("--checkpoint-step", type=int, action="append", required=True)
    curve.add_argument("--quality-target", type=float, default=0.4)
    curve.add_argument("--result-directory", type=Path, required=True)
    curve.add_argument("--representax-gpu", type=int, default=4)
    curve.add_argument("--sentence-transformers-gpu", type=int, default=5)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--summary", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    offline = commands.add_parser("offline-evaluate")
    offline.add_argument(
        "--artifact-kind",
        choices=("representax", "sentence-transformers"),
        required=True,
    )
    offline.add_argument("--artifact", type=Path, required=True)
    offline.add_argument("--data-directory", type=Path, required=True)
    offline.add_argument("--maximum-length", type=int, required=True)
    offline.add_argument("--evaluation-batch-size", type=int, required=True)
    offline.add_argument("--iteration", type=int, required=True)
    offline.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        _prepare_data(arguments.data_directory)
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "pair":
        _pair(arguments)
    elif arguments.command == "curve":
        _curve(arguments)
    elif arguments.command == "aggregate":
        _aggregate(arguments)
    else:
        _offline_evaluate(arguments)


if __name__ == "__main__":
    main()
