"""Matched paper preflight for MiniLM cross-encoder reranking."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.preflights.provenance import reference_source, write_reference_result

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
TEXT_MANIFEST = ROOT / "benchmarks/configs/paper-text-reward-v1.json"
FRAMEWORKS = ("representax", "sentence-transformers")
MICRO_BATCH_SIZE = 16
SEQUENCE_BUCKETS = (64, 128, 256, 512)


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    training_dataset: Mapping[str, Any]
    evaluation_dataset: Mapping[str, Any]
    batch_size: int
    maximum_length: int
    reference_version: str


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the cross-encoder row frozen by the two paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(TEXT_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "cross-encoder-reranking"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "cross-encoder-reranking"
    )
    if campaign_row["frameworks"] != ["representax", "sentence-transformers"]:
        raise ValueError("unexpected cross-encoder frameworks")
    if panel_row["reference"] != "sentence-transformers":
        raise ValueError("unexpected cross-encoder reference")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the cross-encoder reference requires a release")
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        training_dataset=panel["datasets"][panel_row["train"][0]],
        evaluation_dataset=panel["datasets"][panel_row["evaluate"][0]],
        batch_size=int(campaign_row["global_batch"]),
        maximum_length=int(campaign_row["maximum_sequence_length"]),
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


def pointwise_training_rows(
    triples: Iterable[Mapping[str, Any]],
    *,
    query_count: int,
) -> tuple[dict[str, Any], ...]:
    """Take one positive and one hard negative for distinct MS MARCO queries."""

    rows = []
    queries = set()
    for source_index, triple in enumerate(triples):
        query = str(triple["query"])
        if query in queries:
            continue
        queries.add(query)
        rows.extend(
            (
                {
                    "source_index": source_index,
                    "query": query,
                    "document": str(triple["positive"]),
                    "label": 1.0,
                },
                {
                    "source_index": source_index,
                    "query": query,
                    "document": str(triple["negative"]),
                    "label": 0.0,
                },
            )
        )
        if len(queries) == query_count:
            break
    if len(queries) != query_count:
        raise ValueError(f"MS MARCO source contains only {len(queries)} unique queries")
    return tuple(rows)


def select_judged_candidates(
    qrels: Sequence[tuple[str, int]],
    *,
    count: int,
) -> tuple[tuple[str, int], ...]:
    """Choose a stable mixed-relevance TREC candidate set for one query."""

    if count < 2:
        raise ValueError("reranking candidate count must be at least two")
    ordered = sorted(qrels, key=lambda value: value[0])
    relevant = [value for value in ordered if value[1] > 0]
    nonrelevant = [value for value in ordered if value[1] <= 0]
    if not relevant or not nonrelevant:
        raise ValueError("TREC preflight candidates require both relevance classes")
    relevant_count = min(len(relevant), max(1, count // 4))
    selected = [*relevant[:relevant_count], *nonrelevant[: count - relevant_count]]
    if len(selected) < count:
        stop = relevant_count + count - len(selected)
        selected.extend(relevant[relevant_count:stop])
    if len(selected) != count:
        raise ValueError("TREC query has too few judged candidates")
    return tuple(sorted(selected, key=lambda value: value[0]))


def _training_triples(path: Path) -> Iterable[Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    source_index = 0
    for batch in parquet.ParquetFile(path).iter_batches(
        batch_size=8_192,
        columns=("query", "positive", "negative"),
    ):
        for row in batch.to_pylist():
            yield {**row, "source_index": source_index}
            source_index += 1


def _resolve_training_parquet(
    path: Path | None,
    *,
    cache_directory: Path,
    contract: FrozenContract,
) -> Path:
    if path is not None:
        return path.resolve()
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            contract.training_dataset["repo_id"],
            "triplet-all/train-00000-of-00039.parquet",
            repo_type="dataset",
            revision=contract.training_dataset["revision"],
            cache_dir=cache_directory,
        )
    ).resolve()


def prepare_data(
    output: Path,
    *,
    training_parquet: Path | None,
    cache_directory: Path,
    ir_datasets_home: Path,
    training_queries: int,
    evaluation_queries: int,
    candidates_per_query: int,
) -> dict[str, Any]:
    """Materialize deterministic MS MARCO and TREC DL preflight views."""

    contract = frozen_contract()
    output.mkdir(parents=True, exist_ok=False)
    parquet_path = _resolve_training_parquet(
        training_parquet,
        cache_directory=cache_directory,
        contract=contract,
    )
    training_rows = pointwise_training_rows(
        _training_triples(parquet_path),
        query_count=training_queries,
    )
    train_path = output / "train.jsonl"
    _write_jsonl(train_path, training_rows)

    os.environ["IR_DATASETS_HOME"] = str(ir_datasets_home.resolve())
    import ir_datasets

    if ir_datasets.__version__ != contract.evaluation_dataset["source_version"]:
        raise RuntimeError(
            "expected ir-datasets=="
            f"{contract.evaluation_dataset['source_version']}, found "
            f"{ir_datasets.__version__}"
        )
    dataset = ir_datasets.load(contract.evaluation_dataset["dataset_id"])
    queries = {row.query_id: row.text for row in dataset.queries_iter()}
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in dataset.qrels_iter():
        grouped[row.query_id].append((row.doc_id, int(row.relevance)))
    store = dataset.docs_store()
    evaluation_rows = []
    for query_id in sorted(grouped, key=lambda value: int(value)):
        try:
            selected = select_judged_candidates(
                grouped[query_id], count=candidates_per_query
            )
        except ValueError:
            continue
        documents = [store.get(document_id) for document_id, _ in selected]
        if any(document is None for document in documents):
            raise KeyError(
                f"TREC document store is missing a judged row for {query_id}"
            )
        evaluation_rows.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "document_ids": [document_id for document_id, _ in selected],
                "documents": [document.text for document in documents],
                "labels": [float(label) for _, label in selected],
            }
        )
        if len(evaluation_rows) == evaluation_queries:
            break
    if len(evaluation_rows) != evaluation_queries:
        raise ValueError("TREC DL source has too few eligible preflight queries")
    evaluation_path = output / "evaluation.jsonl"
    _write_jsonl(evaluation_path, evaluation_rows)

    manifest = {
        "schema_version": "representax-cross-encoder-preflight-data-v1",
        "contract": asdict(contract),
        "training": {
            "source": str(parquet_path),
            "source_sha256": _sha256(parquet_path),
            "rows": len(training_rows),
            "unique_queries": training_queries,
            "path": train_path.name,
            "sha256": _sha256(train_path),
        },
        "evaluation": {
            "query_count": evaluation_queries,
            "candidates_per_query": candidates_per_query,
            "selection": "up-to-one-quarter relevant, then nonrelevant, by doc id",
            "path": evaluation_path.name,
            "sha256": _sha256(evaluation_path),
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
) -> Any:
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        ExportConfig,
        HuggingFaceExportConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        RerankingEvaluatorConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.cross_encoder import (
        BinaryCrossEntropyConfig,
        PointwiseScoringConfig,
    )

    contract = frozen_contract()
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    if contract.batch_size % MICRO_BATCH_SIZE:
        raise ValueError("frozen batch must be divisible by the microbatch")

    def data(path: Path, *, evaluation: bool = False) -> DataConfig:
        collator = (
            ComponentConfig(
                target="representax.tasks.cross_encoder:ListwiseRankingCollator",
                parameters={
                    "documents_per_query": int(
                        _document(data_directory / "manifest.json")["evaluation"][
                            "candidates_per_query"
                        ]
                    )
                },
            )
            if evaluation
            else ComponentConfig(
                target="representax.tasks.cross_encoder:PointwiseCollator"
            )
        )
        return DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=collator,
            drop_remainder=not evaluation,
            num_threads=0,
            prefetch_buffer_size=2,
        )

    schedule = ComponentConfig(
        target="optax.warmup_cosine_decay_schedule",
        parameters={
            "init_value": 0.0,
            "peak_value": 2e-5,
            "warmup_steps": 1,
            "decay_steps": steps,
            "end_value": 0.0,
        },
    )
    return JobConfig(
        name="paper-preflight-cross-encoder-reranking",
        model=ModelConfig(
            target="representax.models.bert:BertScorer.load_from_hf",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": SEQUENCE_BUCKETS,
            },
        ),
        task=PointwiseScoringConfig(),
        loss=BinaryCrossEntropyConfig(),
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
            schedule=schedule,
            max_gradient_norm=1.0,
        ),
        data=data(data_directory / "train.jsonl"),
        training=TrainingConfig(
            global_batch_size=contract.batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(
                micro_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=contract.batch_size // MICRO_BATCH_SIZE,
            ),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(
            every=steps // 2,
            keep=2,
            save_final=True,
            asynchronous=True,
        ),
        logging=LoggingConfig(console_every=1, timing=True),
        evaluation=EvaluationConfig(
            data=data(data_directory / "evaluation.jsonl", evaluation=True),
            batch_size=1,
            evaluators=(
                RerankingEvaluatorConfig(name="trec_dl_2019", at_k=(1, 10, 32)),
            ),
            on_start=True,
            on_end=True,
            primary_metric="valid/trec_dl_2019/ndcg@10",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(
            selection="final",
            huggingface=HuggingFaceExportConfig(
                source_checkpoint=str(checkpoint),
                adapter=ComponentConfig(
                    target=("representax.models.bert:BertScorerCheckpointAdapter"),
                    parameters={"rematerialization": "none"},
                ),
                verify_reload=True,
            ),
        ),
    )


def _score_pairs(
    model: Any,
    processor: Any,
    pairs: Sequence[Sequence[str]],
) -> np.ndarray:
    import jax

    from representax.config import PrecisionConfig
    from representax.core import score_logits
    from representax.precision import precision_context, resolve_precision_policy

    batch = processor(tuple(tuple(pair) for pair in pairs))
    with precision_context(resolve_precision_policy(PrecisionConfig.bfloat16_mixed())):
        scores = score_logits(model, batch)
    jax.block_until_ready(scores)
    return np.asarray(scores).reshape(-1)


def _probe_pairs(data_directory: Path) -> tuple[tuple[str, str], ...]:
    row = _read_jsonl(data_directory / "evaluation.jsonl")[0]
    return tuple((row["query"], document) for document in row["documents"][:4])


def _cross_encoder_predict(
    model: Any,
    pairs: Sequence[tuple[str, str]],
) -> np.ndarray:
    """Call the reference across its inconsistent overloaded annotation."""

    return np.asarray(
        model.predict(
            list(pairs),
            batch_size=MICRO_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    ).reshape(-1)


def representax_steady_state(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, float]:
    """Derive warm throughput from completed steps after each compilation."""

    seconds = []
    for row in rows:
        metrics = row["metrics"]
        if "perf/compilation_and_first_step_seconds" in metrics:
            continue
        duration = metrics.get("perf/step_seconds")
        if duration is not None and float(duration) > 0:
            seconds.append(float(duration))
    if not seconds:
        raise ValueError("run emitted no post-compilation step durations")
    return {
        "measured_steps": float(len(seconds)),
        "median_step_seconds": statistics.median(seconds),
        "aggregate_examples_per_second": batch_size * len(seconds) / sum(seconds),
    }


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return _read_jsonl(path)


def _representax_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import jax

    from representax import load_inference_bundle
    from representax.models.bert import load_bert_scorer
    from representax.train import run_job

    contract = frozen_contract()
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
    )
    started = time.perf_counter()
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

    _, processor = load_bert_scorer(
        checkpoint,
        revision=contract.model_revision,
        local_files_only=True,
        compute_dtype=jax.numpy.bfloat16,
        sequence_length_buckets=SEQUENCE_BUCKETS,
        rematerialization="none",
    )
    pairs = list(_probe_pairs(data_directory))
    final_probe = _score_pairs(completed.state.model, processor, pairs)
    reloaded, restored_job = load_inference_bundle(completed.inference_bundle)
    reload_probe = _score_pairs(reloaded, processor, pairs)
    if not np.allclose(final_probe, reload_probe, rtol=2e-2, atol=2e-3):
        raise RuntimeError("Representax inference reload changed scorer logits")

    rows = _metric_rows(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    compile_seconds = sum(
        float(row["metrics"].get("perf/compilation_and_first_step_seconds", 0.0))
        for row in training
    )
    bundle_manifest = _document(completed.inference_bundle / "manifest.json")
    return {
        "schema_version": "representax-cross-encoder-worker-v1",
        "framework": "representax",
        "steps": steps,
        "batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "sequence_length_buckets": SEQUENCE_BUCKETS,
        "precision": "bfloat16-compute-float32-master",
        "elapsed_seconds": elapsed,
        "compilation_and_first_step_seconds": compile_seconds,
        "steady_state": representax_steady_state(training, contract.batch_size),
        "initial_evaluation": evaluations[0]["metrics"],
        "final_evaluation": evaluations[-1]["metrics"],
        "final_training": training[-1]["metrics"],
        "resumed": completed.resumed,
        "checkpoint_iterations": [steps // 2, steps],
        "inference_bundle": str(completed.inference_bundle),
        "huggingface_export": str(completed.inference_bundle / "huggingface"),
        "huggingface_verified": bundle_manifest["huggingface"]["verified"],
        "reload_job_name": restored_job.name,
        "reload_probe_maximum_absolute_difference": float(
            np.max(np.abs(final_probe - reload_probe))
        ),
        "probe_pairs": pairs,
        "probe_scores": final_probe.tolist(),
        "peak_device_bytes": int(
            (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        ),
    }


def _reference_metrics(model: Any, data_directory: Path) -> dict[str, float]:
    from representax.evaluation import ranking_metrics

    score_rows = []
    label_rows = []
    for row in _read_jsonl(data_directory / "evaluation.jsonl"):
        pairs = [(row["query"], document) for document in row["documents"]]
        scores = _cross_encoder_predict(model, pairs)
        score_rows.append(scores)
        label_rows.append(np.asarray(row["labels"], dtype=np.float32))
    scores = np.stack(score_rows)
    labels = np.stack(label_rows)
    valid = np.ones(labels.shape, dtype=bool)
    return {
        f"valid/trec_dl_2019/{name}": value
        for name, value in ranking_metrics(
            scores, labels, valid, at_k=(1, 10, labels.shape[1])
        ).items()
    }


def _reference_training_dataset(path: Path) -> Any:
    import datasets

    dataset = datasets.Dataset.from_json(str(path))
    required = ["query", "document", "label"]
    missing = set(required) - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"reference training data is missing columns: {sorted(missing)}"
        )
    return dataset.select_columns(required)


def _sentence_transformers_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import sentence_transformers
    import torch
    import transformers
    from benchmarks.samplers import sequential_sentence_transformers_batches
    from sentence_transformers import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    contract = frozen_contract()
    if sentence_transformers.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected sentence-transformers=={contract.reference_version}, "
            f"found {sentence_transformers.__version__}"
        )
    model = CrossEncoder(
        str(checkpoint),
        local_files_only=True,
        max_length=contract.maximum_length,
    )
    train = _reference_training_dataset(data_directory / "train.jsonl")
    initial_evaluation = _reference_metrics(model, data_directory)
    arguments = CrossEncoderTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=contract.batch_size // MICRO_BATCH_SIZE,
        max_steps=steps,
        learning_rate=2e-5,
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
        save_strategy="steps",
        save_steps=steps // 2,
        save_total_limit=2,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        batch_sampler=sequential_sentence_transformers_batches,
        seed=seed,
        data_seed=seed,
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=arguments,
        train_dataset=train,
        loss=BinaryCrossEntropyLoss(model),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    final_evaluation = _reference_metrics(model, data_directory)
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    pairs = _probe_pairs(data_directory)
    final_probe = _cross_encoder_predict(model, pairs)
    reloaded = CrossEncoder(str(export), local_files_only=True, device="cuda")
    trained_model = model.model
    restored_model = reloaded.model
    if trained_model is None or restored_model is None:
        raise RuntimeError("Sentence Transformers scorer has no underlying model")
    expected_state = trained_model.state_dict()
    reloaded_state = restored_model.state_dict()
    if set(expected_state) != set(reloaded_state) or not all(
        torch.equal(expected_state[name], reloaded_state[name])
        for name in expected_state
    ):
        raise RuntimeError("Sentence Transformers export changed model parameters")
    reload_probe = _cross_encoder_predict(reloaded, pairs)
    reload_difference = float(np.max(np.abs(final_probe - reload_probe)))
    if not np.all(np.isfinite(reload_probe)):
        raise RuntimeError("Sentence Transformers reload produced non-finite logits")
    checkpoints = sorted((run_directory / "checkpoints").glob("checkpoint-*"))
    return {
        "schema_version": "representax-cross-encoder-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "precision": "bfloat16-compute-float32-master",
        "training_seconds": training_seconds,
        "examples_per_second": contract.batch_size * steps / training_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "training_metrics": output.metrics,
        "checkpoint_directories": [str(path) for path in checkpoints],
        "inference_bundle": str(export),
        "reload_parameters_exact": True,
        "reload_probe_maximum_absolute_difference": reload_difference,
        "reload_probe_precision_note": (
            "in-memory trainer inference retains Accelerate BF16 autocast; "
            "the standalone reload evaluates the same exact FP32 parameters"
        ),
        "probe_pairs": pairs,
        "probe_scores": final_probe.tolist(),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _verify_huggingface_export(report: Path, output: Path) -> None:
    from sentence_transformers import CrossEncoder

    values = _document(report)
    model = CrossEncoder(
        values["huggingface_export"],
        local_files_only=True,
        device="cpu",
    )
    pairs = [(str(pair[0]), str(pair[1])) for pair in values["probe_pairs"]]
    scores = _cross_encoder_predict(model, pairs)
    native = np.asarray(values["probe_scores"])
    difference = float(np.max(np.abs(scores - native)))
    if not np.allclose(scores, native, rtol=3e-2, atol=3e-2):
        raise RuntimeError("Transformers reload changed exported scorer logits")
    _write_json(
        output,
        {
            "framework": "sentence-transformers",
            "source": values["huggingface_export"],
            "maximum_absolute_difference": difference,
            "scores": scores.tolist(),
        },
    )


def _worker(arguments: argparse.Namespace) -> None:
    report = (
        _representax_worker(
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
        if arguments.framework == "representax"
        else _sentence_transformers_worker(
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
    )
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        write_reference_result(
            arguments.report,
            report,
            reference="sentence-transformers",
        )


def _run_process(
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    log: Path,
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


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        log = output / f"{framework}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.preflights.cross_encoder",
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
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
            }
        )
        if framework == "representax":
            environment.update(
                {
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                }
            )
        _run_process(command, environment=environment, log=log)
        reports[framework] = _document(report)

    verification = output / "representax-huggingface-reload.json"
    verify_environment = os.environ.copy()
    verify_environment.update(
        {"CUDA_VISIBLE_DEVICES": "", "TOKENIZERS_PARALLELISM": "false"}
    )
    _run_process(
        (
            sys.executable,
            "-m",
            "experiments.preflights.cross_encoder",
            "verify-export",
            "--report",
            str(output / "representax.json"),
            "--output",
            str(verification),
        ),
        environment=verify_environment,
        log=output / "representax-huggingface-reload.log",
    )
    summary = {
        "schema_version": "representax-cross-encoder-preflight-v1",
        "contract": {
            **asdict(frozen_contract()),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "gpu": arguments.gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
        },
        **reports,
        "representax_huggingface_reload": _document(verification),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--training-parquet", type=Path)
    prepare.add_argument(
        "--cache-directory", type=Path, default=Path("/raid/.cache/huggingface")
    )
    prepare.add_argument(
        "--ir-datasets-home", type=Path, default=Path("/raid/.cache/ir_datasets")
    )
    prepare.add_argument("--training-queries", type=int, default=512)
    prepare.add_argument("--evaluation-queries", type=int, default=8)
    prepare.add_argument("--candidates-per-query", type=int, default=32)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=8)
    worker.add_argument("--seed", type=int, default=7)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=8)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--gpu", type=int, default=0)

    verify = subparsers.add_parser("verify-export")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    training_parquet=arguments.training_parquet,
                    cache_directory=arguments.cache_directory,
                    ir_datasets_home=arguments.ir_datasets_home,
                    training_queries=arguments.training_queries,
                    evaluation_queries=arguments.evaluation_queries,
                    candidates_per_query=arguments.candidates_per_query,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "verify-export":
        _verify_huggingface_export(arguments.report, arguments.output)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()
