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
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

TRAIN_DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
TRAIN_DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
TRAIN_DATASET_FILE = "triplet-all/train-00000-of-00039.parquet"
EVALUATION_DATASET_ID = "sentence-transformers/NanoBEIR-en"
EVALUATION_DATASET_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
EVALUATION_SUBSET = "NanoMSMARCO"
ORACLE_VERSION = "5.6.1"
TRANSFORMERS_VERSION = "5.3.0"
THREE_RUN_95_PERCENT_T_CRITICAL = 4.302652729911275


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One native dense family with an identical Sentence Transformers oracle."""

    name: str
    model_id: str
    revision: str
    native_target: str = "representax.integrations.load_sentence_transformer_encoder"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


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


def _padded_chunks(
    rows: Sequence[tuple[int, str]],
    batch_size: int,
) -> Iterator[tuple[tuple[int, ...], tuple[str, ...], tuple[bool, ...]]]:
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        padding = batch_size - len(chunk)
        yield (
            tuple(row[0] for row in chunk) + (-1,) * padding,
            tuple(row[1] for row in chunk) + ("",) * padding,
            (True,) * len(chunk) + (False,) * padding,
        )


def _native_evaluation_batches(
    data: RetrievalEvaluationData,
    collator: Any,
    batch_size: int,
) -> Iterable[Any]:
    import jax.numpy as jnp

    from representax.evaluation import retrieval_evaluation_batch

    for kind, rows in (("query", data.queries), ("document", data.documents)):
        for ids, texts, valid in _padded_chunks(rows, batch_size):
            yield retrieval_evaluation_batch(
                collator(texts),
                jnp.asarray(ids, dtype=jnp.int32),
                kind=kind,
                valid=jnp.asarray(valid),
            )


def _native_retrieval_runner(data: RetrievalEvaluationData) -> Any:
    from representax.evaluation import InformationRetrievalEvaluator
    from representax.train import EvaluationRunner

    return EvaluationRunner(
        InformationRetrievalEvaluator(
            relevant_documents=data.relevant_documents,
            name=EVALUATION_SUBSET,
            score_functions=("cosine",),
            main_score_function="cosine",
            accuracy_at_k=(1, 3, 5, 10),
            precision_recall_at_k=(1, 3, 5, 10, 100),
            mrr_at_k=(10,),
            ndcg_at_k=(10,),
            map_at_k=(100,),
        )
    )


def _native_retrieval_metrics(
    runner: Any,
    model: Any,
    data: RetrievalEvaluationData,
    collator: Any,
    batch_size: int,
) -> tuple[dict[str, float], float, float]:
    result = runner.run(
        model,
        _native_evaluation_batches(data, collator, batch_size),
    )
    return dict(result.metrics), result.duration_seconds, result.compilation_seconds


def _training_compile_seconds(run_directory: Path) -> float:
    with (run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            value = row["metrics"].get("perf/compilation_and_first_step_seconds")
            if value is not None:
                return float(value)
    raise ValueError("Representax run did not report first executable use")


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
    cache_chunk_size: int | None,
    evaluation_batch_size: int,
    evaluation_every_steps: int | None,
    data_threads: int,
    prefetch_buffer_size: int,
    seed: int,
) -> dict[str, Any]:
    import jax

    from representax.config import (
        BatchConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        EvaluatorConfig,
        ExportConfig,
        GradCacheConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.integrations import SentenceTextCollator
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig
    from representax.train import build_job_runtime, run_training

    checkpoint = _checkpoint(spec)
    evaluation_data = _evaluation_data(data_directory)
    evaluation_collator = SentenceTextCollator(
        checkpoint,
        maximum_length=maximum_length,
    )
    training_path = _artifact_path(
        data_directory,
        TRAIN_DATASET_ID,
        TRAIN_DATASET_FILE,
    )
    training_data = DataConfig(
        distribution=mix(source(str(training_path), map=identity), shuffle=False),
        collate=ComponentConfig(
            target="representax.integrations.RetrievalPairCollator",
            parameters={
                "checkpoint": str(checkpoint),
                "maximum_length": maximum_length,
            },
        ),
        drop_remainder=True,
        num_threads=data_threads,
        prefetch_buffer_size=prefetch_buffer_size,
    )
    job = JobConfig(
        name=f"matched-dense-retrieval-{spec.name}",
        model=ModelConfig(
            target=spec.native_target,
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": spec.revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "float32",
            },
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=20.0, symmetric=False),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "learning_rate": 2e-5,
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=training_data,
        training=TrainingConfig(
            global_batch_size=batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=batch_size),
            grad_cache=(
                None
                if cache_chunk_size is None
                else GradCacheConfig(micro_batch_size=cache_chunk_size)
            ),
            activation_rematerialization="none",
            donate_buffers=True,
        ),
        logging=LoggingConfig(console_every=steps),
        evaluation=(
            None
            if evaluation_every_steps is None
            else EvaluationConfig(
                data=training_data,
                batch_size=evaluation_batch_size,
                evaluators=(EvaluatorConfig(name=EVALUATION_SUBSET),),
                every_steps=evaluation_every_steps,
                on_start=True,
                on_end=True,
                primary_metric=(f"valid/{EVALUATION_SUBSET}/cosine_ndcg@10"),
                primary_metric_mode="max",
                save_best=False,
            )
        ),
        export=ExportConfig(enabled=False),
    )

    load_started = time.perf_counter()
    runtime = build_job_runtime(job)
    model_and_data_seconds = time.perf_counter() - load_started
    evaluation_runner = _native_retrieval_runner(evaluation_data)
    periodic_evaluation = evaluation_every_steps is not None
    if periodic_evaluation:
        initial = None
        initial_seconds = None
        initial_compile_seconds = None
    else:
        initial, initial_seconds, initial_compile_seconds = _native_retrieval_metrics(
            evaluation_runner,
            runtime.state.model,
            evaluation_data,
            evaluation_collator,
            evaluation_batch_size,
        )
    training_started = time.perf_counter()
    result = run_training(
        state=runtime.state,
        step=runtime.step,
        batches=runtime.batches,
        job=job,
        run_directory=run_directory,
        place_batch=runtime.place_batch,
        evaluation_runners=(evaluation_runner,) if periodic_evaluation else (),
        evaluation_batches=(
            (
                lambda: _native_evaluation_batches(
                    evaluation_data,
                    evaluation_collator,
                    evaluation_batch_size,
                )
            )
            if periodic_evaluation
            else None
        ),
    )
    jax.block_until_ready(result.state)
    training_seconds = time.perf_counter() - training_started
    if periodic_evaluation:
        evaluation_history = _representax_evaluation_history(run_directory)
        initial = evaluation_history[0]["metrics"]
        final = evaluation_history[-1]["metrics"]
        initial_seconds = evaluation_history[0]["evaluation_seconds"]
        initial_compile_seconds = evaluation_history[0][
            "evaluation_compilation_seconds"
        ]
        final_seconds = evaluation_history[-1]["evaluation_seconds"]
        final_compile_seconds = evaluation_history[-1]["evaluation_compilation_seconds"]
        evaluation_during_training_seconds = sum(
            item["evaluation_seconds"] for item in evaluation_history
        )
    else:
        final, final_seconds, final_compile_seconds = _native_retrieval_metrics(
            evaluation_runner,
            result.state.model,
            evaluation_data,
            evaluation_collator,
            evaluation_batch_size,
        )
        evaluation_history = []
        evaluation_during_training_seconds = 0.0
    training_compute_seconds = training_seconds - evaluation_during_training_seconds
    compilation_and_first_step_seconds = _training_compile_seconds(run_directory)
    sustained_seconds = max(
        training_compute_seconds - compilation_and_first_step_seconds,
        0.0,
    )
    return {
        "schema_version": "representax-dense-retrieval-worker-v1",
        "framework": "representax",
        "jax_version": jax.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "batch_size": batch_size,
        "steps": steps,
        "maximum_length": maximum_length,
        "cache_chunk_size": cache_chunk_size,
        "seed": seed,
        "model_and_data_seconds": model_and_data_seconds,
        "training_seconds": training_seconds,
        "training_compute_seconds": training_compute_seconds,
        "compilation_and_first_step_seconds": compilation_and_first_step_seconds,
        "sustained_seconds_after_first_step": sustained_seconds,
        "amortized_examples_per_second": batch_size * steps / training_compute_seconds,
        "sustained_examples_per_second_after_first_step": (
            batch_size * max(steps - 1, 0) / sustained_seconds
            if sustained_seconds > 0
            else None
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
    seed: int,
) -> dict[str, Any]:
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
    del data_threads, prefetch_buffer_size
    checkpoint = _checkpoint(spec)
    evaluation_data = _evaluation_data(data_directory)
    training_path = _artifact_path(
        data_directory,
        TRAIN_DATASET_ID,
        TRAIN_DATASET_FILE,
    )
    load_started = time.perf_counter()
    model = SentenceTransformer(str(checkpoint), local_files_only=True)
    model.max_seq_length = maximum_length
    required_rows = batch_size * steps
    training_data = datasets.Dataset(_training_table(training_path, required_rows))
    model_and_data_seconds = time.perf_counter() - load_started

    class FixedLengthCollator(SentenceTransformerDataCollator):
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            return _pad_features(
                super().__call__(features),
                maximum_length,
                model.tokenizer.pad_token_id,
            )

    loss = (
        MultipleNegativesRankingLoss(model, scale=20.0)
        if cache_chunk_size is None
        else CachedMultipleNegativesRankingLoss(
            model,
            scale=20.0,
            mini_batch_size=cache_chunk_size,
            show_progress_bar=False,
        )
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=batch_size,
        max_steps=steps,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
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
            return control

    callback = PeriodicRetrievalCallback()
    trainer = RecordingTrainer(
        model=model,
        args=arguments,
        train_dataset=training_data,
        loss=loss,
        data_collator=FixedLengthCollator(
            preprocess_fn=model.tokenize,
            valid_label_columns=[],
        ),
    )
    callback.trainer = trainer
    trainer.add_callback(callback)
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    training_started_at = training_started
    output = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    if trainer.final_train_loss is None:  # pragma: no cover - trainer invariant
        raise AssertionError("Sentence Transformers did not compute a training loss")
    final_train_loss = float(trainer.final_train_loss)
    training_compute_seconds = training_seconds - evaluation_during_training_seconds
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
    return {
        "schema_version": "representax-dense-retrieval-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "batch_size": batch_size,
        "steps": steps,
        "maximum_length": maximum_length,
        "cache_chunk_size": cache_chunk_size,
        "seed": seed,
        "model_and_data_seconds": model_and_data_seconds,
        "training_seconds": training_seconds,
        "training_compute_seconds": training_compute_seconds,
        "amortized_examples_per_second": (
            batch_size * steps / training_compute_seconds
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
        "cache_chunk_size": arguments.cache_chunk_size,
        "evaluation_batch_size": arguments.evaluation_batch_size,
        "evaluation_every_steps": arguments.evaluation_every_steps,
        "data_threads": arguments.data_threads,
        "prefetch_buffer_size": arguments.prefetch_buffer_size,
        "seed": arguments.seed,
    }
    report = (
        _representax_report(spec, **kwargs)
        if arguments.framework == "representax"
        else _sentence_transformers_report(spec, **kwargs)
    )
    _atomic_json(arguments.report, report)


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
                "JAX_COMPILATION_CACHE_DIR": str(log_path.parent.parent / "jax-cache"),
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
    seeds = []
    for summary in summaries:
        if summary.get("schema_version") != "representax-dense-retrieval-comparison-v1":
            raise ValueError("aggregate input is not a dense-retrieval comparison")
        contract = dict(summary["contract"])
        seeds.append(int(contract.pop("seed")))
        contracts.append(contract)
        if not summary["comparison"]["initial_metric_parity"]:
            raise ValueError("aggregate input did not pass initial metric parity")
    if len(set(seeds)) != 3:
        raise ValueError(f"aggregate seeds must be unique: {seeds}")
    if contracts[1:] != contracts[:-1]:
        raise ValueError("aggregate scientific and execution contracts differ")

    ordered = sorted(zip(seeds, paths, summaries, strict=True))
    metrics = {
        "sustained_training_speedup": lambda row: row["comparison"][
            "sustained_training_speedup"
        ],
        "representax_sustained_examples_per_second": lambda row: row["representax"][
            "sustained_examples_per_second_after_first_step"
        ],
        "sentence_transformers_examples_per_second": lambda row: row[
            "sentence_transformers"
        ]["amortized_examples_per_second"],
        "final_ndcg@10_difference": lambda row: row["comparison"][
            "final_ndcg@10_difference"
        ],
        "compilation_break_even_steps": lambda row: row["comparison"][
            "compilation_break_even_steps"
        ],
    }
    aggregate = {
        "schema_version": "representax-dense-retrieval-three-run-aggregate-v1",
        "contract": contracts[0],
        "seeds": [seed for seed, _, _ in ordered],
        "inputs": [
            {
                "seed": seed,
                "path": str(path),
                "sha256": _sha256(path),
            }
            for seed, path, _ in ordered
        ],
        "all_initial_metrics_match": True,
        "metrics": {
            name: _three_run_interval(
                [float(select(summary)) for _, _, summary in ordered]
            )
            for name, select in metrics.items()
        },
    }
    _atomic_json(arguments.output.expanduser().resolve(), aggregate)
    print(json.dumps(aggregate["metrics"], indent=2, sort_keys=True))


def _pair(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
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
    maximum_initial_metric_difference = _initial_metric_parity(
        native["initial_evaluation"],
        oracle["initial_evaluation"],
        tolerance=spec.initial_metric_tolerance,
    )
    metric = f"valid/{EVALUATION_SUBSET}/cosine_ndcg@10"
    native_sustained_rate = native["sustained_examples_per_second_after_first_step"]
    oracle_rate = oracle["amortized_examples_per_second"]
    native_step_seconds = native["sustained_seconds_after_first_step"] / max(
        arguments.steps - 1,
        1,
    )
    oracle_step_seconds = oracle["training_compute_seconds"] / arguments.steps
    seconds_saved_per_step = oracle_step_seconds - native_step_seconds
    compilation_break_even_steps = (
        native["compilation_and_first_step_seconds"] / seconds_saved_per_step
        if seconds_saved_per_step > 0
        else None
    )
    summary = {
        "schema_version": "representax-dense-retrieval-comparison-v1",
        "contract": {
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
            "cache_chunk_size": arguments.cache_chunk_size,
            "seed": arguments.seed,
            "evaluation_every_steps": arguments.evaluation_every_steps,
            "representax_data_threads": arguments.data_threads,
            "representax_prefetch_buffer_size": arguments.prefetch_buffer_size,
            "optimizer": "AdamW",
            "learning_rate": 2e-5,
            "precision": "float32",
            "loss": "cosine MNR scale=20 asymmetric",
            "initial_metric_absolute_tolerance": spec.initial_metric_tolerance,
        },
        "representax": {**native, **process["representax"]},
        "sentence_transformers": {
            **oracle,
            **process["sentence-transformers"],
        },
        "comparison": {
            "amortized_training_speedup": (
                native["amortized_examples_per_second"] / oracle_rate
            ),
            "sustained_training_speedup": native_sustained_rate / oracle_rate,
            "compilation_break_even_steps": compilation_break_even_steps,
            "initial_metric_parity": True,
            "maximum_initial_metric_absolute_difference": (
                maximum_initial_metric_difference
            ),
            "representax_final_ndcg@10": native["final_evaluation"][metric],
            "sentence_transformers_final_ndcg@10": oracle["final_evaluation"][metric],
            "final_ndcg@10_difference": (
                native["final_evaluation"][metric] - oracle["final_evaluation"][metric]
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
    parser.add_argument("--cache-chunk-size", type=int)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--evaluation-every-steps", type=int)
    parser.add_argument("--data-threads", type=int, default=4)
    parser.add_argument("--prefetch-buffer-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)


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
    else:
        _aggregate(arguments)


if __name__ == "__main__":
    main()
