"""Bounded GTE-ModernColBERT preflight against PyLate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import statistics
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from experiments.preflights.provenance import reference_source, write_reference_result

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
TEXT_MANIFEST = ROOT / "benchmarks/configs/paper-text-reward-v1.json"
FRAMEWORKS = ("representax", "pylate")
GRAD_CACHE_MICRO_BATCH = 8
EVALUATION_BATCH_SIZE = 256
QUERY_SCORE_BATCH = 10
DOCUMENT_SCORE_BATCH = 64
QUERY_BUCKETS = (16, 32)
DOCUMENT_BUCKETS = (32, 64, 128, 256)
_ENCODING_PHASES = (
    "preprocess_seconds",
    "placement_seconds",
    "encoder_seconds",
    "host_seconds",
)


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    training_dataset: Mapping[str, Any]
    evaluation_dataset: Mapping[str, Any]
    global_batch_size: int
    maximum_query_length: int
    maximum_document_length: int
    reference_version: str


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the immutable late-interaction row from the paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(TEXT_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "late-interaction"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "late-interaction"
    )
    if campaign_row["frameworks"] != ["representax", "pylate"]:
        raise ValueError("unexpected late-interaction frameworks")
    if panel_row["reference"] != "pylate":
        raise ValueError("unexpected late-interaction reference")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the late-interaction reference requires a release")
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        training_dataset=panel["datasets"][panel_row["train"][0]],
        evaluation_dataset=panel["datasets"][panel_row["evaluate"][0]],
        global_batch_size=int(campaign_row["global_batch"]),
        maximum_query_length=int(campaign_row["maximum_query_length"]),
        maximum_document_length=int(campaign_row["maximum_document_length"]),
        reference_version=reference.release,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def select_training_rows(
    rows: Iterable[Mapping[str, Any]], *, count: int
) -> tuple[dict[str, Any], ...]:
    """Take the first deterministic query-positive pairs from MS MARCO."""

    selected = []
    for source_index, row in enumerate(rows):
        query = str(row["query"])
        positive = str(row["positive"])
        if not query or not positive:
            continue
        selected.append(
            {"source_index": source_index, "query": query, "positive": positive}
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"MS MARCO source contains only {len(selected)} usable rows")
    return tuple(selected)


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    file = parquet.ParquetFile(path)
    for batch in file.iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def _single_parquet(directory: Path) -> Path:
    paths = tuple(sorted(directory.glob("*.parquet")))
    if len(paths) != 1:
        raise ValueError(f"expected one parquet under {directory}, found {len(paths)}")
    return paths[0]


def prepare_data(
    output: Path,
    *,
    training_parquet: Path,
    nanobeir_directory: Path,
    training_rows: int,
) -> dict[str, Any]:
    """Materialize deterministic train and complete NanoMSMARCO transfer views."""

    contract = frozen_contract()
    if training_rows < contract.global_batch_size:
        raise ValueError("training rows must contain at least one frozen global batch")
    output.mkdir(parents=True, exist_ok=False)
    training = select_training_rows(
        _parquet_rows(training_parquet), count=training_rows
    )
    train_path = output / "train.jsonl"
    _write_jsonl(train_path, training)

    query_source = _single_parquet(nanobeir_directory / "queries")
    corpus_source = _single_parquet(nanobeir_directory / "corpus")
    qrels_source = _single_parquet(nanobeir_directory / "qrels")
    queries = tuple(
        {"identifier": int(row["_id"]), "text": str(row["text"])}
        for row in _parquet_rows(query_source)
    )
    corpus = tuple(
        {"identifier": int(row["_id"]), "text": str(row["text"])}
        for row in _parquet_rows(corpus_source)
    )
    relevant: dict[int, list[int]] = {}
    for row in _parquet_rows(qrels_source):
        relevant.setdefault(int(row["query-id"]), []).append(int(row["corpus-id"]))
    query_ids = {row["identifier"] for row in queries}
    corpus_ids = {row["identifier"] for row in corpus}
    if set(relevant) != query_ids:
        raise ValueError("NanoMSMARCO qrels do not cover exactly the frozen queries")
    if any(not set(values) <= corpus_ids for values in relevant.values()):
        raise ValueError("NanoMSMARCO qrels refer to missing corpus rows")
    query_path = output / "queries.jsonl"
    corpus_path = output / "corpus.jsonl"
    qrels_path = output / "qrels.json"
    _write_jsonl(query_path, queries)
    _write_jsonl(corpus_path, corpus)
    _write_json(
        qrels_path,
        {str(query): sorted(documents) for query, documents in relevant.items()},
    )

    manifest = {
        "schema_version": "representax-late-interaction-preflight-data-v1",
        "contract": asdict(contract),
        "training": {
            "source": str(training_parquet.resolve()),
            "source_sha256": _sha256(training_parquet),
            "rows": len(training),
            "path": train_path.name,
            "sha256": _sha256(train_path),
        },
        "evaluation": {
            "dataset": "NanoMSMARCO",
            "queries": len(queries),
            "documents": len(corpus),
            "qrels": sum(len(values) for values in relevant.values()),
            "query_path": query_path.name,
            "query_sha256": _sha256(query_path),
            "corpus_path": corpus_path.name,
            "corpus_sha256": _sha256(corpus_path),
            "qrels_path": qrels_path.name,
            "qrels_sha256": _sha256(qrels_path),
            "source_sha256": {
                "queries": _sha256(query_source),
                "corpus": _sha256(corpus_source),
                "qrels": _sha256(qrels_source),
            },
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _representax_job(
    *,
    checkpoint: Path,
    data_directory: Path,
    steps: int,
    seed: int,
    lifecycle: bool = True,
    static_shapes: bool = False,
) -> Any:
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        ExportConfig,
        GradCacheConfig,
        HuggingFaceExportConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.late_interaction import (
        LateInteractionConfig,
        LateInteractionContrastiveConfig,
    )

    contract = frozen_contract()
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    required_rows = contract.global_batch_size * steps
    manifest_path = data_directory / "manifest.json"
    training_rows = (
        int(_document(manifest_path)["training"]["rows"])
        if manifest_path.is_file()
        else required_rows
    )
    repeats = max(1, (required_rows + training_rows - 1) // training_rows)
    sources = tuple(
        source(
            str(data_directory / "train.jsonl"),
            map=identity,
            name=f"ms-marco-{index}",
        )
        for index in range(repeats)
    )
    return JobConfig(
        name=(
            "paper-preflight-late-interaction"
            if lifecycle
            else "paper-profile-late-interaction"
        ),
        model=ModelConfig(
            target=(
                "representax.models.late_interaction:"
                "LateInteractionTextEncoder.load_from_hf"
            ),
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "rematerialization": "none",
                "query_sequence_length_buckets": list(
                    (max(QUERY_BUCKETS),) if static_shapes else QUERY_BUCKETS
                ),
                "document_sequence_length_buckets": list(
                    (max(DOCUMENT_BUCKETS),) if static_shapes else DOCUMENT_BUCKETS
                ),
            },
        ),
        task=LateInteractionConfig(),
        loss=LateInteractionContrastiveConfig(
            temperature=0.02, symmetric=False, negative_scope="global"
        ),
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
                    "peak_value": 3e-6,
                    "warmup_steps": 1,
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(*sources, shuffle=False),
            collate=ComponentConfig(
                target="representax.tasks.retrieval:RetrievalCollator",
                parameters={"document_field": "positive"},
            ),
            drop_remainder=True,
            num_threads=0,
            prefetch_buffer_size=1,
        ),
        training=TrainingConfig(
            global_batch_size=contract.global_batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=contract.global_batch_size),
            grad_cache=GradCacheConfig(
                micro_batch_size=GRAD_CACHE_MICRO_BATCH,
                loss_row_chunk_size=GRAD_CACHE_MICRO_BATCH,
            ),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=(
            CheckpointConfig(
                every=steps // 2, keep=2, save_final=True, asynchronous=True
            )
            if lifecycle
            else None
        ),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=lifecycle),
        export=ExportConfig(
            enabled=lifecycle,
            selection="final",
            huggingface=(
                HuggingFaceExportConfig(
                    source_checkpoint=str(checkpoint),
                    adapter=ComponentConfig(
                        target=(
                            "representax.integrations.late_interaction:"
                            "LateInteractionCheckpointAdapter"
                        ),
                        parameters={
                            "model_id": contract.model_id,
                            "revision": contract.model_revision,
                        },
                    ),
                    verify_reload=True,
                )
                if lifecycle
                else None
            ),
        ),
    )


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return _read_jsonl(path)


def representax_steady_state(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, float]:
    """Measure only completed updates that did not compile a fresh executable."""

    seconds = []
    compilation = 0.0
    for row in rows:
        metrics = row["metrics"]
        if "perf/compilation_and_first_step_seconds" in metrics:
            compilation += float(metrics["perf/compilation_and_first_step_seconds"])
            continue
        duration = metrics.get("perf/step_seconds")
        if duration is not None and float(duration) > 0:
            seconds.append(float(duration))
    if not seconds:
        raise ValueError("run emitted no post-compilation step durations")
    return {
        "measured_steps": float(len(seconds)),
        "compilation_and_first_step_seconds": compilation,
        "median_step_seconds": statistics.median(seconds),
        "examples_per_second": batch_size * len(seconds) / sum(seconds),
    }


def _write_flat_index(
    directory: Path, identifiers: Sequence[int], embeddings: Sequence[np.ndarray]
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    lengths = np.asarray([len(value) for value in embeddings], dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths)))
    tokens = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    np.save(directory / "tokens.npy", tokens, allow_pickle=False)
    np.save(directory / "offsets.npy", offsets, allow_pickle=False)
    _write_json(directory / "identifiers.json", list(identifiers))
    files = tuple(path for path in directory.iterdir() if path.is_file())
    return {
        "kind": "exact-flat-multivector",
        "documents": len(identifiers),
        "tokens": int(len(tokens)),
        "embedding_dimension": int(tokens.shape[-1]),
        "write_seconds": time.perf_counter() - started,
        "size_bytes": sum(path.stat().st_size for path in files),
        "files": {path.name: _sha256(path) for path in files},
    }


def _pad_embeddings(
    values: Sequence[np.ndarray], *, length: int
) -> tuple[np.ndarray, np.ndarray]:
    if not values or any(value.ndim != 2 for value in values):
        raise ValueError("token embeddings must be non-empty rank-two arrays")
    dimension = values[0].shape[1]
    if any(value.shape[1] != dimension or len(value) > length for value in values):
        raise ValueError("token embeddings disagree with the fixed score shape")
    padded = np.zeros((len(values), length, dimension), dtype=np.float32)
    valid = np.zeros((len(values), length), dtype=np.bool_)
    for index, value in enumerate(values):
        padded[index, : len(value)] = value
        valid[index, : len(value)] = True
    return padded, valid


def _padded_block(
    values: Sequence[np.ndarray], *, start: int, count: int, length: int
) -> tuple[np.ndarray, np.ndarray, int]:
    block = list(values[start : start + count])
    actual = len(block)
    if not block:
        raise ValueError("cannot pad an empty score block")
    empty = np.zeros((1, block[0].shape[1]), dtype=np.float32)
    block.extend(empty.copy() for _ in range(count - actual))
    padded, valid = _pad_embeddings(block, length=length)
    return padded, valid, actual


def _retrieval_metrics(
    scores: np.ndarray,
    query_ids: Sequence[int],
    document_ids: Sequence[int],
    relevant: Mapping[int, frozenset[int]],
) -> dict[str, float]:
    from representax.evaluation.retrieval import information_retrieval_metrics

    maximum = min(100, len(document_ids))
    top = np.argsort(-scores, axis=1)[:, :maximum]
    ranked = np.asarray(document_ids, dtype=np.int64)[top]
    return {
        f"valid/nanobeir-msmarco/maxsim_{name}": float(value)
        for name, value in information_retrieval_metrics(
            ranked,
            np.asarray(query_ids, dtype=np.int64),
            relevant,
            accuracy_at_k=(1, 5, 10),
            precision_recall_at_k=(1, 5, 10),
            mrr_at_k=(10,),
            ndcg_at_k=(10,),
            map_at_k=(100,),
        ).items()
    }


def _evaluation_data(
    data_directory: Path,
) -> tuple[
    tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[int, frozenset[int]]
]:
    queries = _read_jsonl(data_directory / "queries.jsonl")
    corpus = _read_jsonl(data_directory / "corpus.jsonl")
    relevant = {
        int(query): frozenset(int(document) for document in documents)
        for query, documents in _document(data_directory / "qrels.json").items()
    }
    return queries, corpus, relevant


def _representax_encode(
    model: Any,
    processor: Any,
    texts: Sequence[str],
    *,
    route: Any,
    sequence_lengths: Sequence[int],
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    import equinox as eqx
    import jax

    from representax.core import encode_late_interaction

    @eqx.filter_jit
    def compiled(candidate: Any, batch: Any) -> Any:
        return encode_late_interaction(candidate, batch, route=route)

    outputs: list[np.ndarray | None] = [None] * len(texts)
    timings: list[dict[str, float | int]] = []
    order = sorted(range(len(texts)), key=lambda index: (-len(texts[index]), index))
    for start in range(0, len(order), EVALUATION_BATCH_SIZE):
        indices = order[start : start + EVALUATION_BATCH_SIZE]
        block = [texts[index] for index in indices]
        actual = len(block)
        block.extend("" for _ in range(EVALUATION_BATCH_SIZE - actual))

        began = time.perf_counter()
        batch = processor(tuple(block), route=route)
        preprocess_seconds = time.perf_counter() - began

        began = time.perf_counter()
        placed = jax.device_put(batch)
        jax.block_until_ready(placed)
        placement_seconds = time.perf_counter() - began

        began = time.perf_counter()
        encoded = compiled(model, placed)
        jax.block_until_ready(encoded)
        encoder_seconds = time.perf_counter() - began

        began = time.perf_counter()
        values, valid = jax.device_get((encoded.values, encoded.valid))
        values = np.asarray(values, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        host_seconds = time.perf_counter() - began
        sequence_length = values.shape[1]
        if sequence_length not in sequence_lengths:
            raise RuntimeError("Representax encoded outside the frozen sequence shape")
        timings.append(
            {
                "sequence_length": sequence_length,
                "examples": actual,
                "preprocess_seconds": preprocess_seconds,
                "placement_seconds": placement_seconds,
                "encoder_seconds": encoder_seconds,
                "host_seconds": host_seconds,
            }
        )
        for batch_index, output_index in enumerate(indices):
            outputs[output_index] = values[batch_index, valid[batch_index]]
    if any(value is None for value in outputs):
        raise RuntimeError("Representax encoding did not restore every sorted row")
    return tuple(value for value in outputs if value is not None), _encoding_timings(
        timings, compiled=True
    )


def _encoding_timings(
    rows: Sequence[Mapping[str, float | int]], *, compiled: bool
) -> dict[str, Any]:
    if not rows:
        raise ValueError("encoding timing requires at least one batch")
    cold_indices: set[int] = set()
    first_by_shape: dict[int, float] = {}
    if compiled:
        for index, row in enumerate(rows):
            length = int(row["sequence_length"])
            if length not in first_by_shape:
                first_by_shape[length] = float(row["encoder_seconds"])
                cold_indices.add(index)
    else:
        first_by_shape[int(rows[0]["sequence_length"])] = float(
            rows[0]["encoder_seconds"]
        )
        cold_indices.add(0)
    warm = [row for index, row in enumerate(rows) if index not in cold_indices]
    warm_examples = sum(int(row["examples"]) for row in warm)
    warm_encoder = sum(float(row["encoder_seconds"]) for row in warm)
    warm_end_to_end = sum(
        sum(float(row[name]) for name in _ENCODING_PHASES) for row in warm
    )
    totals = {name: sum(float(row[name]) for row in rows) for name in _ENCODING_PHASES}
    first_eager_seconds = next(iter(first_by_shape.values()))
    return {
        "batch_size": EVALUATION_BATCH_SIZE,
        "batches": len(rows),
        "compiled_sequence_lengths": sorted(first_by_shape),
        "compilation_and_first_batch_seconds": (
            sum(first_by_shape.values()) if compiled else 0.0
        ),
        "compilation_and_first_batch_seconds_by_length": (
            {str(length): value for length, value in first_by_shape.items()}
            if compiled
            else {}
        ),
        "first_eager_batch_seconds": 0.0 if compiled else first_eager_seconds,
        "warm_batches": len(warm),
        "warm_examples": warm_examples,
        "warm_encoder_seconds": warm_encoder,
        "warm_end_to_end_seconds": warm_end_to_end,
        "warm_encoder_examples_per_second": (
            0.0 if not warm_encoder else warm_examples / warm_encoder
        ),
        "warm_end_to_end_examples_per_second": (
            0.0 if not warm_end_to_end else warm_examples / warm_end_to_end
        ),
        "examples_per_second_after_first_batch": (
            0.0 if not warm_encoder else warm_examples / warm_encoder
        ),
        "total_encoding_seconds": sum(totals.values()),
        "phase_totals_seconds": totals,
        "batch_timings": [dict(row) for row in rows],
    }


def _representax_maxsim(
    queries: Sequence[np.ndarray], documents: Sequence[np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    from representax.core import LateInteractionRepresentation
    from representax.tasks.late_interaction.scoring import maxsim_scores

    contract = frozen_contract()

    @jax.jit
    def score(
        query_values: Any, query_valid: Any, document_values: Any, document_valid: Any
    ) -> Any:
        return maxsim_scores(
            LateInteractionRepresentation(query_values, query_valid),
            LateInteractionRepresentation(document_values, document_valid),
            document_chunk_size=DOCUMENT_SCORE_BATCH,
        )

    document_values, document_valid = _pad_embeddings(
        documents, length=contract.maximum_document_length
    )
    document_values_device = jnp.asarray(document_values, dtype=jnp.bfloat16)
    document_valid_device = jnp.asarray(document_valid)
    result = np.empty((len(queries), len(documents)), dtype=np.float32)
    compile_seconds = None
    warm_seconds = []
    for query_start in range(0, len(queries), QUERY_SCORE_BATCH):
        q_values, q_valid, q_actual = _padded_block(
            queries,
            start=query_start,
            count=QUERY_SCORE_BATCH,
            length=contract.maximum_query_length,
        )
        began = time.perf_counter()
        block = score(
            jnp.asarray(q_values, dtype=jnp.bfloat16),
            jnp.asarray(q_valid),
            document_values_device,
            document_valid_device,
        )
        block.block_until_ready()
        duration = time.perf_counter() - began
        if compile_seconds is None:
            compile_seconds = duration
        else:
            warm_seconds.append(duration)
        result[query_start : query_start + q_actual] = np.asarray(block)[:q_actual]
    warm_comparisons = max(
        0,
        len(queries) * len(documents)
        - min(QUERY_SCORE_BATCH, len(queries)) * len(documents),
    )
    return result, {
        "backend": "jax-exact-maxsim",
        "compilation_and_first_tile_seconds": float(compile_seconds or 0.0),
        "warm_score_seconds": sum(warm_seconds),
        "warm_query_document_comparisons": float(warm_comparisons),
        "warm_query_document_comparisons_per_second": (
            warm_comparisons / sum(warm_seconds) if warm_seconds else 0.0
        ),
    }


def _representax_evaluation(
    model: Any,
    processor: Any,
    data_directory: Path,
    *,
    index_directory: Path | None,
) -> dict[str, Any]:
    from representax.config import PrecisionConfig
    from representax.core import Route
    from representax.precision import model_for_compute, resolve_precision_policy

    queries, corpus, relevant = _evaluation_data(data_directory)
    compute_model = model_for_compute(
        model, resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    )
    query_embeddings, query_encoding = _representax_encode(
        compute_model,
        processor,
        [str(row["text"]) for row in queries],
        route=Route.QUERY,
        sequence_lengths=QUERY_BUCKETS,
    )
    document_embeddings, document_encoding = _representax_encode(
        compute_model,
        processor,
        [str(row["text"]) for row in corpus],
        route=Route.DOCUMENT,
        sequence_lengths=DOCUMENT_BUCKETS,
    )
    index = (
        None
        if index_directory is None
        else _write_flat_index(
            index_directory,
            [int(row["identifier"]) for row in corpus],
            document_embeddings,
        )
    )
    scores, scoring = _representax_maxsim(query_embeddings, document_embeddings)
    return {
        "metrics": _retrieval_metrics(
            scores,
            [int(row["identifier"]) for row in queries],
            [int(row["identifier"]) for row in corpus],
            relevant,
        ),
        "query_encoding": query_encoding,
        "document_encoding": document_encoding,
        "index": index,
        "scoring": scoring,
        "probe_query_embedding": query_embeddings[0].tolist(),
    }


def _representax_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    resume_existing: bool = False,
) -> dict[str, Any]:
    import jax

    from representax import load_inference_bundle
    from representax.integrations.late_interaction import (
        load_late_interaction_text_model,
    )
    from representax.train import run_job

    contract = frozen_contract()
    initial_model, processor = load_late_interaction_text_model(
        checkpoint,
        revision=contract.model_revision,
        local_files_only=True,
        parameter_dtype=jax.numpy.float32,
        compute_dtype=jax.numpy.bfloat16,
        rematerialization="none",
        query_sequence_length_buckets=QUERY_BUCKETS,
        document_sequence_length_buckets=DOCUMENT_BUCKETS,
    )
    initial_evaluation = _representax_evaluation(
        initial_model, processor, data_directory, index_directory=None
    )
    del initial_model
    gc.collect()
    jax.clear_caches()

    job = _representax_job(
        checkpoint=checkpoint, data_directory=data_directory, steps=steps, seed=seed
    )
    started = time.perf_counter()
    if resume_existing:
        completed = run_job(job, run_directory, resume=True)
    else:
        paused = run_job(job, run_directory, stop_after=steps // 2)
        if paused.completed_iterations != steps // 2:
            raise RuntimeError("Representax did not stop at the resumable midpoint")
        del paused
        gc.collect()
        jax.clear_caches()
        completed = run_job(job, run_directory, resume=True)
    jax.block_until_ready(completed.state)
    elapsed = time.perf_counter() - started
    if not completed.resumed or completed.completed_iterations != steps:
        raise RuntimeError("Representax did not resume to the requested update count")
    if completed.inference_bundle is None:
        raise RuntimeError("Representax did not export an inference bundle")

    final_evaluation = _representax_evaluation(
        completed.state.model,
        processor,
        data_directory,
        index_directory=run_directory / "flat-index",
    )
    probe_text = str(_read_jsonl(data_directory / "queries.jsonl")[0]["text"])
    from representax.core import Route, encode_late_interaction

    probe = processor((probe_text,), route=Route.QUERY)
    expected = encode_late_interaction(
        cast(Any, completed.state.model), probe, route=Route.QUERY
    )
    restored, restored_job = load_inference_bundle(completed.inference_bundle)
    actual = encode_late_interaction(cast(Any, restored), probe, route=Route.QUERY)
    jax.block_until_ready((expected, actual))
    reload_difference = float(
        np.max(np.abs(np.asarray(expected.values) - np.asarray(actual.values)))
    )
    if reload_difference != 0.0 or restored_job.name != job.name:
        raise RuntimeError("native inference reload changed late-interaction output")

    rows = _metric_rows(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    if len(training) != steps:
        raise RuntimeError("Representax metric stream is missing training updates")
    update_norms = [
        float(row["metrics"]["train/update_global_norm"]) for row in training
    ]
    if not all(np.isfinite(update_norms)) or not any(
        value > 0 for value in update_norms
    ):
        raise RuntimeError("Representax produced no finite nonzero update")
    bundle_manifest = _document(completed.inference_bundle / "manifest.json")
    return {
        "schema_version": "representax-late-interaction-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "precision": "bfloat16-compute-float32-master",
        "elapsed_seconds": elapsed,
        "steady_state": representax_steady_state(training, contract.global_batch_size),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_loss": float(training[-1]["metrics"]["train/loss"]),
        "resumed": completed.resumed,
        "checkpoint_iterations": [steps // 2, steps],
        "inference_bundle": str(completed.inference_bundle),
        "huggingface_export": str(completed.inference_bundle / "huggingface"),
        "huggingface_verified": bundle_manifest["huggingface"]["verified"],
        "native_reload_maximum_absolute_difference": reload_difference,
        "probe_text": probe_text,
        "probe_query_embedding": final_evaluation["probe_query_embedding"],
        "peak_device_bytes": int(
            (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        ),
        "device": jax.devices()[0].device_kind,
    }


class _StepTimer:
    """Small Trainer callback factory kept independent of optional torch imports."""

    @staticmethod
    def callback(*, stop_after: int | None = None) -> Any:
        import torch
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def __init__(self) -> None:
                self.started = 0.0
                self.durations: list[tuple[int, float]] = []

            def on_step_begin(
                self, args: Any, state: Any, control: Any, **_: Any
            ) -> None:
                torch.cuda.synchronize()
                self.started = time.perf_counter()

            def on_step_end(
                self, args: Any, state: Any, control: Any, **_: Any
            ) -> None:
                torch.cuda.synchronize()
                self.durations.append(
                    (int(state.global_step), time.perf_counter() - self.started)
                )
                if stop_after is not None and state.global_step >= stop_after:
                    control.should_training_stop = True

        return Callback()


def _pylate_encode(
    model: Any, texts: Sequence[str], *, is_query: bool
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    import torch
    from sentence_transformers.util import batch_to_device

    model.eval()
    outputs: list[np.ndarray | None] = [None] * len(texts)
    timings: list[dict[str, float | int]] = []
    order = sorted(range(len(texts)), key=lambda index: (-len(texts[index]), index))
    for start in range(0, len(order), EVALUATION_BATCH_SIZE):
        indices = order[start : start + EVALUATION_BATCH_SIZE]
        block = [texts[index] for index in indices]
        actual = len(block)
        block.extend("" for _ in range(EVALUATION_BATCH_SIZE - actual))

        began = time.perf_counter()
        features = model.tokenize(block, is_query=is_query, pad=True)
        preprocess_seconds = time.perf_counter() - began

        began = time.perf_counter()
        features = batch_to_device(features, model.device)
        torch.cuda.synchronize()
        placement_seconds = time.perf_counter() - began

        began = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.forward(input=features)
            if is_query and model.do_query_expansion:
                masks = torch.ones_like(encoded["input_ids"], dtype=torch.bool)
            elif is_query:
                masks = encoded["attention_mask"].bool()
            else:
                masks = torch.logical_and(
                    model.skiplist_mask(
                        input_ids=features["input_ids"], skiplist=model.skiplist
                    ),
                    encoded["attention_mask"],
                )
            values = [
                torch.nn.functional.normalize(
                    encoded["token_embeddings"][index, mask], p=2, dim=1
                )
                for index, mask in enumerate(masks)
            ]
        torch.cuda.synchronize()
        encoder_seconds = time.perf_counter() - began

        began = time.perf_counter()
        host_values = [
            np.asarray(value.float().cpu(), dtype=np.float32)
            for value in values[:actual]
        ]
        host_seconds = time.perf_counter() - began
        sequence_length = int(features["input_ids"].shape[1])
        timings.append(
            {
                "sequence_length": sequence_length,
                "examples": actual,
                "preprocess_seconds": preprocess_seconds,
                "placement_seconds": placement_seconds,
                "encoder_seconds": encoder_seconds,
                "host_seconds": host_seconds,
            }
        )
        for output_index, value in zip(indices, host_values, strict=True):
            outputs[output_index] = value
    if any(value is None for value in outputs):
        raise RuntimeError("PyLate encoding did not restore every sorted row")
    return tuple(value for value in outputs if value is not None), _encoding_timings(
        timings, compiled=False
    )


def _pylate_maxsim(
    queries: Sequence[np.ndarray], documents: Sequence[np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    colbert_scores = importlib.import_module("pylate.scores").colbert_scores

    contract = frozen_contract()
    result = np.empty((len(queries), len(documents)), dtype=np.float32)
    durations = []
    for query_start in range(0, len(queries), QUERY_SCORE_BATCH):
        q_values, q_valid, q_actual = _padded_block(
            queries,
            start=query_start,
            count=QUERY_SCORE_BATCH,
            length=contract.maximum_query_length,
        )
        for document_start in range(0, len(documents), DOCUMENT_SCORE_BATCH):
            d_values, d_valid, d_actual = _padded_block(
                documents,
                start=document_start,
                count=DOCUMENT_SCORE_BATCH,
                length=contract.maximum_document_length,
            )
            began = time.perf_counter()
            block = colbert_scores(
                torch.as_tensor(q_values, device="cuda"),
                torch.as_tensor(d_values, device="cuda"),
                queries_mask=torch.as_tensor(q_valid, device="cuda"),
                documents_mask=torch.as_tensor(d_valid, device="cuda"),
                backend="torch",
            )
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - began)
            result[
                query_start : query_start + q_actual,
                document_start : document_start + d_actual,
            ] = block[:q_actual, :d_actual].float().cpu().numpy()
    warm_comparisons = max(
        0,
        len(queries) * len(documents)
        - min(QUERY_SCORE_BATCH, len(queries))
        * min(DOCUMENT_SCORE_BATCH, len(documents)),
    )
    return result, {
        "backend": "pylate-torch-exact-maxsim",
        "compilation_and_first_tile_seconds": 0.0,
        "first_eager_tile_seconds": durations[0],
        "warm_score_seconds": sum(durations[1:]),
        "warm_query_document_comparisons": float(warm_comparisons),
        "warm_query_document_comparisons_per_second": (
            warm_comparisons / sum(durations[1:]) if len(durations) > 1 else 0.0
        ),
    }


def _pylate_evaluation(
    model: Any,
    data_directory: Path,
    *,
    index_directory: Path | None,
) -> dict[str, Any]:
    queries, corpus, relevant = _evaluation_data(data_directory)
    query_embeddings, query_encoding = _pylate_encode(
        model, [str(row["text"]) for row in queries], is_query=True
    )
    document_embeddings, document_encoding = _pylate_encode(
        model, [str(row["text"]) for row in corpus], is_query=False
    )
    index = (
        None
        if index_directory is None
        else _write_flat_index(
            index_directory,
            [int(row["identifier"]) for row in corpus],
            document_embeddings,
        )
    )
    scores, scoring = _pylate_maxsim(query_embeddings, document_embeddings)
    return {
        "metrics": _retrieval_metrics(
            scores,
            [int(row["identifier"]) for row in queries],
            [int(row["identifier"]) for row in corpus],
            relevant,
        ),
        "query_encoding": query_encoding,
        "document_encoding": document_encoding,
        "index": index,
        "scoring": scoring,
        "probe_query_embedding": query_embeddings[0].tolist(),
    }


def _reference_arguments(
    output: Path,
    *,
    max_steps: int,
    save_steps: int,
    seed: int,
    save: bool = True,
) -> Any:
    from sentence_transformers import SentenceTransformerTrainingArguments

    contract = frozen_contract()
    return SentenceTransformerTrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=contract.global_batch_size,
        max_steps=max_steps,
        learning_rate=3e-6,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        save_strategy="steps" if save else "no",
        save_steps=save_steps,
        save_total_limit=2 if save else None,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=seed,
        data_seed=seed,
    )


def _reference_dataset(path: Path, *, rows: int) -> Any:
    import datasets

    source = datasets.Dataset.from_json(str(path))
    repeats = max(1, (rows + len(source) - 1) // len(source))
    dataset = datasets.concatenate_datasets([source] * repeats).select(range(rows))
    return dataset.select_columns(["query", "positive"])


def _pylate_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    from importlib.metadata import version

    import torch
    from sentence_transformers import SentenceTransformerTrainer

    losses = importlib.import_module("pylate.losses")
    models = importlib.import_module("pylate.models")
    utils = importlib.import_module("pylate.utils")

    contract = frozen_contract()
    if version("pylate") != contract.reference_version:
        raise RuntimeError(
            f"expected pylate=={contract.reference_version}, found {version('pylate')}"
        )
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    required_rows = contract.global_batch_size * steps
    train = _reference_dataset(data_directory / "train.jsonl", rows=required_rows)

    def load_model(path: Path) -> Any:
        model = models.ColBERT(
            model_name_or_path=str(path), device="cuda", local_files_only=True
        )
        model.query_length = contract.maximum_query_length
        model.document_length = contract.maximum_document_length
        return model

    model = load_model(checkpoint)
    initial_evaluation = _pylate_evaluation(model, data_directory, index_directory=None)
    midpoint = steps // 2
    checkpoint_root = run_directory / "checkpoints"
    first_timer = _StepTimer.callback(stop_after=midpoint)
    first_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        score_mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        temperature=0.02,
    )
    first_trainer = SentenceTransformerTrainer(
        model=model,
        args=_reference_arguments(
            checkpoint_root, max_steps=steps, save_steps=midpoint, seed=seed
        ),
        train_dataset=train,
        loss=first_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[first_timer],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first_output = first_trainer.train()
    torch.cuda.synchronize()
    if first_trainer.state.global_step != midpoint:
        raise RuntimeError("PyLate did not stop at the requested midpoint")
    midpoint_checkpoint = checkpoint_root / f"checkpoint-{midpoint}"
    if not midpoint_checkpoint.is_dir():
        raise RuntimeError("PyLate did not save the resumable midpoint checkpoint")
    first_history = tuple(first_trainer.state.log_history)
    del first_trainer, first_loss, model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model(checkpoint)
    second_timer = _StepTimer.callback()
    second_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        score_mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        temperature=0.02,
    )
    second_trainer = SentenceTransformerTrainer(
        model=model,
        args=_reference_arguments(
            checkpoint_root, max_steps=steps, save_steps=midpoint, seed=seed
        ),
        train_dataset=train,
        loss=second_loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[second_timer],
    )
    second_output = second_trainer.train(
        resume_from_checkpoint=str(midpoint_checkpoint)
    )
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    second_history = tuple(second_trainer.state.log_history)
    final_checkpoint = checkpoint_root / f"checkpoint-{steps}"
    if not final_checkpoint.is_dir():
        raise RuntimeError("PyLate did not save the final resumable checkpoint")
    final_evaluation = _pylate_evaluation(
        model,
        data_directory,
        index_directory=run_directory / "flat-index",
    )
    export = run_directory / "final-model"
    second_trainer.save_model(str(export))
    probe_text = str(_read_jsonl(data_directory / "queries.jsonl")[0]["text"])
    del second_trainer, second_loss, model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint_model = load_model(final_checkpoint)
    expected = np.asarray(
        checkpoint_model.encode(
            [probe_text], is_query=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
    )
    restored = load_model(export)
    checkpoint_state = checkpoint_model.state_dict()
    restored_state = restored.state_dict()
    parameters_exact = set(checkpoint_state) == set(restored_state) and all(
        torch.equal(checkpoint_state[name], restored_state[name])
        for name in checkpoint_state
    )
    actual = np.asarray(
        restored.encode(
            [probe_text], is_query=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
    )
    reload_difference = float(np.max(np.abs(expected - actual)))
    if not parameters_exact or not np.array_equal(expected, actual):
        raise RuntimeError("PyLate export reload changed the probe embedding")

    durations = [*first_timer.durations, *second_timer.durations]
    warm = [
        duration
        for iteration, duration in durations
        if iteration not in {1, midpoint + 1}
    ]
    if not warm:
        raise RuntimeError("PyLate emitted no warmed optimizer-step timings")
    loss_rows = [
        float(row["loss"])
        for row in (*first_history, *second_history)
        if row.get("loss") is not None
    ]
    if not loss_rows:
        raise RuntimeError("PyLate trainer emitted no finite training losses")
    peak_device_bytes = int(torch.cuda.max_memory_allocated())
    return {
        "schema_version": "representax-late-interaction-worker-v1",
        "framework": "pylate",
        "framework_version": version("pylate"),
        "sentence_transformers_version": version("sentence-transformers"),
        "transformers_version": version("transformers"),
        "torch_version": torch.__version__,
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "precision": "pytorch-bfloat16-autocast-float32-master",
        "execution_mode": "eager",
        "compilation_seconds": 0.0,
        "training_seconds": training_seconds,
        "steady_state": {
            "measured_steps": len(warm),
            "median_step_seconds": statistics.median(warm),
            "examples_per_second": contract.global_batch_size * len(warm) / sum(warm),
        },
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_loss": loss_rows[-1],
        "training_metrics": {
            "first": first_output.metrics,
            "resumed": second_output.metrics,
        },
        "resumed": True,
        "checkpoint_iterations": [midpoint, steps],
        "inference_bundle": str(export),
        "reload_parameters_exact": parameters_exact,
        "reload_maximum_absolute_difference": reload_difference,
        "probe_text": probe_text,
        "probe_query_embedding": final_evaluation["probe_query_embedding"],
        "peak_device_bytes": peak_device_bytes,
        "device": torch.cuda.get_device_name(),
    }


def _representax_training_profile(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import jax

    from representax.train import run_job

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        lifecycle=False,
        static_shapes=True,
    )
    started = time.perf_counter()
    completed = run_job(job, run_directory)
    jax.block_until_ready(completed.state)
    elapsed = time.perf_counter() - started
    rows = _metric_rows(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    if len(training) != steps:
        raise RuntimeError("Representax timing profile is missing training updates")
    if completed.inference_bundle is not None:
        raise RuntimeError(
            "checkpoint-free timing profile unexpectedly exported a model"
        )
    return _representax_training_profile_report(
        run_directory=run_directory, steps=steps, elapsed_seconds=elapsed
    )


def _representax_training_profile_report(
    *, run_directory: Path, steps: int, elapsed_seconds: float | None = None
) -> dict[str, Any]:
    import jax

    contract = frozen_contract()
    rows = _metric_rows(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    if len(training) != steps:
        raise RuntimeError("Representax timing profile is missing training updates")
    if elapsed_seconds is None:
        startup = next(row for row in rows if row.get("event") == "startup")
        elapsed_seconds = float(startup["metrics"]["perf/startup_seconds"])
        elapsed_seconds += sum(
            float(
                row["metrics"][
                    "perf/step_seconds"
                    if "perf/step_seconds" in row["metrics"]
                    else "perf/compilation_and_first_step_seconds"
                ]
            )
            for row in training
        )
    return {
        "schema_version": "representax-late-interaction-training-profile-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "query_sequence_length": contract.maximum_query_length,
        "document_sequence_length": contract.maximum_document_length,
        "elapsed_seconds": elapsed_seconds,
        "steady_state": representax_steady_state(training, contract.global_batch_size),
        "step_timings": [
            {
                "iteration": int(row["iteration"]),
                "step_seconds": float(
                    row["metrics"][
                        "perf/step_seconds"
                        if "perf/step_seconds" in row["metrics"]
                        else "perf/compilation_and_first_step_seconds"
                    ]
                ),
                "compiled": (
                    "perf/compilation_and_first_step_seconds" in row["metrics"]
                ),
            }
            for row in training
        ],
        "checkpointing": False,
        "export": False,
        "device": jax.devices()[0].device_kind,
    }


def _pylate_training_profile(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    from importlib.metadata import version

    import torch
    from sentence_transformers import SentenceTransformerTrainer

    losses = importlib.import_module("pylate.losses")
    models = importlib.import_module("pylate.models")
    utils = importlib.import_module("pylate.utils")
    contract = frozen_contract()
    model = models.ColBERT(
        model_name_or_path=str(checkpoint), device="cuda", local_files_only=True
    )
    model.query_length = contract.maximum_query_length
    model.document_length = contract.maximum_document_length
    train = _reference_dataset(
        data_directory / "train.jsonl",
        rows=contract.global_batch_size * steps,
    )
    timer = _StepTimer.callback()
    loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        score_mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        temperature=0.02,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=_reference_arguments(
            run_directory,
            max_steps=steps,
            save_steps=steps + 1,
            seed=seed,
            save=False,
        ),
        train_dataset=train,
        loss=loss,
        data_collator=utils.ColBERTCollator(model.tokenize),
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    durations = [duration for _, duration in timer.durations]
    if len(durations) != steps or len(durations) < 3:
        raise RuntimeError("PyLate timing profile is missing training updates")
    warm = durations[1:]
    return {
        "schema_version": "representax-late-interaction-training-profile-v1",
        "framework": "pylate",
        "framework_version": version("pylate"),
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "query_sequence_length": contract.maximum_query_length,
        "document_sequence_length": contract.maximum_document_length,
        "elapsed_seconds": elapsed,
        "steady_state": {
            "measured_steps": len(warm),
            "median_step_seconds": statistics.median(warm),
            "examples_per_second": (contract.global_batch_size * len(warm) / sum(warm)),
        },
        "step_timings": [
            {
                "iteration": iteration,
                "step_seconds": duration,
                "compiled": False,
                "first_eager_step": iteration == 1,
            }
            for iteration, duration in timer.durations
        ],
        "checkpointing": False,
        "export": False,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "device": torch.cuda.get_device_name(),
    }


def _training_profile(arguments: argparse.Namespace) -> None:
    if arguments.steps < 4 or arguments.steps % 2:
        raise ValueError(
            "training profile requires an even step count of at least four"
        )
    if arguments.reuse_existing:
        if arguments.framework != "representax":
            raise ValueError("--reuse-existing currently requires Representax")
        report = _representax_training_profile_report(
            run_directory=arguments.run_directory,
            steps=arguments.steps,
        )
        _write_json(arguments.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    function = (
        _representax_training_profile
        if arguments.framework == "representax"
        else _pylate_training_profile
    )
    report = function(
        checkpoint=arguments.checkpoint,
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    _write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _encoding_profile(arguments: argparse.Namespace) -> None:
    contract = frozen_contract()
    corpus = _read_jsonl(arguments.data_directory / "corpus.jsonl")
    if arguments.examples <= 0:
        raise ValueError("encoding profile examples must be positive")
    if arguments.examples > len(corpus):
        raise ValueError("encoding profile exceeds the available evaluation corpus")
    texts = [str(row["text"]) for row in corpus[: arguments.examples]]
    if arguments.framework == "representax":
        import jax

        from representax.config import PrecisionConfig
        from representax.core import Route
        from representax.integrations.late_interaction import (
            load_late_interaction_text_model,
        )
        from representax.precision import model_for_compute, resolve_precision_policy

        model, processor = load_late_interaction_text_model(
            arguments.checkpoint,
            revision=contract.model_revision,
            local_files_only=True,
            parameter_dtype=jax.numpy.float32,
            compute_dtype=jax.numpy.bfloat16,
            rematerialization="none",
            query_sequence_length_buckets=(contract.maximum_query_length,),
            document_sequence_length_buckets=(contract.maximum_document_length,),
        )
        model = model_for_compute(
            model, resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
        )
        embeddings, timings = _representax_encode(
            model,
            processor,
            texts,
            route=Route.DOCUMENT,
            sequence_lengths=(contract.maximum_document_length,),
        )
    else:
        models = importlib.import_module("pylate.models")
        model = models.ColBERT(
            model_name_or_path=str(arguments.checkpoint),
            device="cuda",
            local_files_only=True,
        )
        model.query_length = contract.maximum_query_length
        model.document_length = contract.maximum_document_length
        embeddings, timings = _pylate_encode(model, texts, is_query=False)
    if not all(np.isfinite(value).all() for value in embeddings):
        raise RuntimeError("encoding profile produced non-finite embeddings")
    report = {
        "schema_version": "representax-late-interaction-encoding-profile-v1",
        "framework": arguments.framework,
        "examples": len(embeddings),
        "batch_size": EVALUATION_BATCH_SIZE,
        "sequence_length": contract.maximum_document_length,
        "embedding_dimension": int(embeddings[0].shape[-1]),
        "timings": timings,
    }
    _write_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _verify_export(report: Path, output: Path) -> None:
    models = importlib.import_module("pylate.models")

    values = _document(report)
    model = models.ColBERT(
        model_name_or_path=values["huggingface_export"],
        device="cuda",
        local_files_only=True,
    )
    actual = np.asarray(
        model.encode(
            [values["probe_text"]],
            is_query=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0],
        dtype=np.float32,
    )
    expected = np.asarray(values["probe_query_embedding"], dtype=np.float32)
    difference = float(np.max(np.abs(expected - actual)))
    if expected.shape != actual.shape or not np.allclose(
        expected, actual, rtol=3e-2, atol=3e-3
    ):
        raise RuntimeError("PyLate reload changed the Representax export")
    _write_json(
        output,
        {
            "framework": "pylate",
            "source": values["huggingface_export"],
            "maximum_absolute_difference": difference,
            "embedding_shape": list(actual.shape),
        },
    )


def _remeasure_representax(
    report: Path, checkpoint: Path, data_directory: Path, output: Path
) -> None:
    """Re-run only final retrieval systems measurements from a durable bundle."""

    import jax

    from representax import load_inference_bundle
    from representax.integrations.late_interaction import (
        load_late_interaction_text_model,
    )

    contract = frozen_contract()
    values = _document(report)
    model, _ = load_inference_bundle(values["inference_bundle"])
    _, processor = load_late_interaction_text_model(
        checkpoint,
        revision=contract.model_revision,
        local_files_only=True,
        parameter_dtype=jax.numpy.float32,
        compute_dtype=jax.numpy.bfloat16,
        rematerialization="none",
        query_sequence_length_buckets=QUERY_BUCKETS,
        document_sequence_length_buckets=DOCUMENT_BUCKETS,
    )
    evaluation = _representax_evaluation(
        model, processor, data_directory, index_directory=None
    )
    evaluation.pop("probe_query_embedding")
    _write_json(
        output,
        {
            "schema_version": "representax-late-interaction-remeasurement-v1",
            "source_report": str(report.resolve()),
            "source_inference_bundle": values["inference_bundle"],
            "measurement": evaluation,
        },
    )


def _remeasure_pylate(report: Path, data_directory: Path, output: Path) -> None:
    """Re-run final PyLate retrieval without writing another model or index."""

    models = importlib.import_module("pylate.models")
    values = _document(report)
    model = models.ColBERT(
        model_name_or_path=values["inference_bundle"],
        device="cuda",
        local_files_only=True,
    )
    contract = frozen_contract()
    model.query_length = contract.maximum_query_length
    model.document_length = contract.maximum_document_length
    evaluation = _pylate_evaluation(model, data_directory, index_directory=None)
    evaluation.pop("probe_query_embedding")
    _write_json(
        output,
        {
            "schema_version": "representax-late-interaction-remeasurement-v1",
            "source_report": str(report.resolve()),
            "source_inference_bundle": values["inference_bundle"],
            "measurement": evaluation,
        },
    )


def _run_process(
    command: Sequence[str], *, environment: Mapping[str, str], log: Path
) -> None:
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"preflight process failed with {result.returncode}: {log}")


def _worker(arguments: argparse.Namespace) -> None:
    function = (
        _representax_worker if arguments.framework == "representax" else _pylate_worker
    )
    parameters = dict(
        checkpoint=arguments.checkpoint,
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    if arguments.framework == "representax":
        parameters["resume_existing"] = arguments.resume_existing
    elif arguments.resume_existing:
        raise ValueError("--resume-existing is only available for Representax")
    report = function(**parameters)
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        report = write_reference_result(
            arguments.report,
            report,
            reference="pylate",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def _build_summary(
    *,
    data_directory: Path,
    representax_report: Path,
    pylate_report: Path,
    verification_report: Path,
    steps: int,
    seed: int,
    gpu: int,
    commands: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "representax-late-interaction-preflight-v1",
        "scope": "bounded-readiness-preflight-not-paper-result",
        "contract": {
            **asdict(frozen_contract()),
            "steps": steps,
            "seed": seed,
            "gpu": gpu,
            "data_manifest": _document(data_directory / "manifest.json"),
        },
        "commands": dict(commands or {}),
        "representax": _document(representax_report),
        "pylate": _document(pylate_report),
        "representax_pylate_reload": _document(verification_report),
    }


def _summarize_existing(arguments: argparse.Namespace) -> None:
    summary = _build_summary(
        data_directory=arguments.data_directory,
        representax_report=arguments.representax_report,
        pylate_report=arguments.pylate_report,
        verification_report=arguments.verification_report,
        steps=arguments.steps,
        seed=arguments.seed,
        gpu=arguments.gpu,
    )
    _write_json(arguments.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    interpreters = {
        "representax": arguments.representax_python,
        "pylate": arguments.reference_python,
    }
    commands = {}
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        command = [
            str(interpreters[framework]),
            "-m",
            "experiments.preflights.late_interaction",
            "worker",
            "--framework",
            framework,
            "--checkpoint",
            str(arguments.checkpoint),
            "--data-directory",
            str(arguments.data_directory),
            "--run-directory",
            str(output / framework),
            "--report",
            str(report),
            "--steps",
            str(arguments.steps),
            "--seed",
            str(arguments.seed),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
            "HF_HOME": str(arguments.hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
        if framework == "representax":
            environment.update(
                {
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                    "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
                }
            )
        else:
            environment["PYLATE_SCORES_BACKEND"] = "torch"
        commands[framework] = command
        _run_process(
            command,
            environment=environment,
            log=output / f"{framework}.log",
        )

    verification = output / "representax-pylate-reload.json"
    verify = [
        str(arguments.reference_python),
        "-m",
        "experiments.preflights.late_interaction",
        "verify-export",
        "--report",
        str(output / "representax.json"),
        "--output",
        str(verification),
    ]
    _run_process(
        verify,
        environment={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
            "HF_HOME": str(arguments.hf_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        log=output / "representax-pylate-reload.log",
    )
    summary = _build_summary(
        data_directory=arguments.data_directory,
        representax_report=output / "representax.json",
        pylate_report=output / "pylate.json",
        verification_report=verification,
        steps=arguments.steps,
        seed=arguments.seed,
        gpu=arguments.gpu,
        commands=commands,
    )
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--training-parquet", type=Path, required=True)
    prepare.add_argument("--nanobeir-directory", type=Path, required=True)
    prepare.add_argument("--training-rows", type=int, default=2048)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=4)
    worker.add_argument("--seed", type=int, default=7)
    worker.add_argument("--resume-existing", action="store_true")

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--representax-python", type=Path, required=True)
    pair.add_argument("--reference-python", type=Path, required=True)
    pair.add_argument("--hf-home", type=Path, default=Path("/raid/.cache/huggingface"))
    pair.add_argument("--steps", type=int, default=4)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--gpu", type=int, default=1)

    verify = subparsers.add_parser("verify-export")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    remeasure = subparsers.add_parser("remeasure-representax")
    remeasure.add_argument("--report", type=Path, required=True)
    remeasure.add_argument("--checkpoint", type=Path, required=True)
    remeasure.add_argument("--data-directory", type=Path, required=True)
    remeasure.add_argument("--output", type=Path, required=True)

    remeasure_pylate = subparsers.add_parser("remeasure-pylate")
    remeasure_pylate.add_argument("--report", type=Path, required=True)
    remeasure_pylate.add_argument("--data-directory", type=Path, required=True)
    remeasure_pylate.add_argument("--output", type=Path, required=True)

    encoding_profile = subparsers.add_parser("encoding-profile")
    encoding_profile.add_argument("--framework", choices=FRAMEWORKS, required=True)
    encoding_profile.add_argument("--checkpoint", type=Path, required=True)
    encoding_profile.add_argument("--data-directory", type=Path, required=True)
    encoding_profile.add_argument("--output", type=Path, required=True)
    encoding_profile.add_argument("--examples", type=int, default=1536)

    training_profile = subparsers.add_parser("training-profile")
    training_profile.add_argument("--framework", choices=FRAMEWORKS, required=True)
    training_profile.add_argument("--checkpoint", type=Path, required=True)
    training_profile.add_argument("--data-directory", type=Path, required=True)
    training_profile.add_argument("--run-directory", type=Path, required=True)
    training_profile.add_argument("--output", type=Path, required=True)
    training_profile.add_argument("--steps", type=int, default=10)
    training_profile.add_argument("--seed", type=int, default=7)
    training_profile.add_argument("--reuse-existing", action="store_true")

    summarize = subparsers.add_parser("summarize-existing")
    summarize.add_argument("--data-directory", type=Path, required=True)
    summarize.add_argument("--representax-report", type=Path, required=True)
    summarize.add_argument("--pylate-report", type=Path, required=True)
    summarize.add_argument("--verification-report", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--steps", type=int, default=4)
    summarize.add_argument("--seed", type=int, default=7)
    summarize.add_argument("--gpu", type=int, default=1)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    training_parquet=arguments.training_parquet,
                    nanobeir_directory=arguments.nanobeir_directory,
                    training_rows=arguments.training_rows,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "verify-export":
        _verify_export(arguments.report, arguments.output)
    elif arguments.command == "remeasure-representax":
        _remeasure_representax(
            arguments.report,
            arguments.checkpoint,
            arguments.data_directory,
            arguments.output,
        )
    elif arguments.command == "remeasure-pylate":
        _remeasure_pylate(
            arguments.report,
            arguments.data_directory,
            arguments.output,
        )
    elif arguments.command == "encoding-profile":
        _encoding_profile(arguments)
    elif arguments.command == "training-profile":
        _training_profile(arguments)
    elif arguments.command == "summarize-existing":
        _summarize_existing(arguments)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()
