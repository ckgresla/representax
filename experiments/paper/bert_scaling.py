"""Bounded dense-retrieval acceptance for the controlled BERT scaling ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LADDER_MANIFEST = ROOT / "benchmarks/configs/bert-scaling-v1.json"
DEFAULT_ARTIFACT_ROOT = Path(
    "/home/ckg/representax-artifacts/bert-scaling-ladder-msmarco-20260831-v3"
)
DEFAULT_NQ_SOURCE = Path(
    "/home/ckg/representax-artifacts/nq-miracl-acceptance-20260831/nq/data"
)
TRAIN_DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
TRAIN_DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
TRAIN_DATASET_FILE = "triplet-all/train-00000-of-00039.parquet"
EVALUATION_DATASET_ID = "sentence-transformers/NanoBEIR-en"
EVALUATION_DATASET_REVISION = "beb106fbcfaa599c508c667041bf8c85fd78736b"
EVALUATION_SUBSET = "NanoMSMARCO"
NQ_DATASET_ID = "mteb/nq"
NQ_DATASET_REVISION = "b84726e65fd226125cf7c0cbeeb5c214d49e8187"
TOKENIZER_ID = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
SIZE_ORDER = ("bert-30m", "bert-110m", "bert-500m", "bert-1b", "bert-4b")
GPU_ASSIGNMENTS = {
    "bert-30m": (2,),
    "bert-110m": (2,),
    "bert-500m": (2, 3),
    "bert-1b": (2, 3),
    "bert-4b": (2, 3, 4, 5),
}
SHARDED_SIZES = frozenset({"bert-500m", "bert-1b", "bert-4b"})
TRAINING_ROWS = 8
TRAINING_BATCH_SIZE = 2
EVALUATION_QUERIES = 4
EVALUATION_DOCUMENTS = 16
EVALUATION_BATCH_SIZE = 4
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_STEPS = 3
HBM_ADMISSION_FRACTION = 0.85
ACTIVE_UPDATE_BYTES_PER_PARAMETER = 18
PRECISION_LABEL = "fp32-master-bf16-compute"
PRECISION_RECIPE = {
    "model_parameter_dtype": "float32",
    "model_compute_dtype": "bfloat16",
    "training_policy": "bfloat16_mixed",
    "configured_evaluation_policy": "bfloat16_mixed",
    "exported_evaluation_policy": "bfloat16_mixed",
}


@dataclass(frozen=True, slots=True)
class LadderEntry:
    """One validated architecture in the frozen scaling manifest."""

    name: str
    target_parameters: int
    expected_parameters: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    type_vocab_size: int
    hidden_activation: str
    hidden_dropout_probability: float
    attention_dropout_probability: float
    norm_epsilon: float
    initializer_range: float
    pad_token_id: int
    admission_gate: str | None = None

    def config_values(self) -> dict[str, Any]:
        """Return exactly the values accepted by native ``BertConfig``."""

        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "hidden_activation": self.hidden_activation,
            "hidden_dropout_probability": self.hidden_dropout_probability,
            "attention_dropout_probability": self.attention_dropout_probability,
            "norm_epsilon": self.norm_epsilon,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
        }


@dataclass(frozen=True, slots=True)
class EvaluationData:
    """One bounded query/corpus/qrels view with remapped integer IDs."""

    queries: tuple[tuple[int, str], ...]
    documents: tuple[tuple[int, str], ...]
    relevant_documents: Mapping[int, frozenset[int]]
    query_source_ids: tuple[str, ...] = ()
    document_source_ids: tuple[str, ...] = ()


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_ladder(path: Path = LADDER_MANIFEST) -> tuple[LadderEntry, ...]:
    """Validate the shared manifest and materialize its native configurations."""

    from scripts.validate_bert_scaling import validate_manifest

    document = _document(path)
    validated = validate_manifest(document)
    shared = document["shared"]
    entries = []
    for size, record in zip(document["sizes"], validated, strict=True):
        entry = LadderEntry(
            name=str(size["name"]),
            target_parameters=int(size["target_parameters"]),
            expected_parameters=int(size["expected_parameters"]),
            vocab_size=int(shared["vocab_size"]),
            hidden_size=int(size["hidden_size"]),
            intermediate_size=int(size["intermediate_size"]),
            num_hidden_layers=int(size["num_hidden_layers"]),
            num_attention_heads=int(size["num_attention_heads"]),
            max_position_embeddings=int(shared["max_position_embeddings"]),
            type_vocab_size=int(shared["type_vocab_size"]),
            hidden_activation=str(shared["hidden_activation"]),
            hidden_dropout_probability=float(
                shared["hidden_dropout_probability"]
            ),
            attention_dropout_probability=float(
                shared["attention_dropout_probability"]
            ),
            norm_epsilon=float(shared["norm_epsilon"]),
            initializer_range=float(shared["initializer_range"]),
            pad_token_id=int(shared["pad_token_id"]),
            admission_gate=size.get("admission_gate"),
        )
        if record["parameters"] != entry.expected_parameters:
            raise ValueError(f"validated count drifted for {entry.name}")
        entries.append(entry)
    return tuple(entries)


def ladder_entry(name: str) -> LadderEntry:
    try:
        return next(entry for entry in load_ladder() if entry.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown BERT ladder size {name!r}") from error


def select_bounded_evaluation(
    queries: Mapping[str, str],
    documents: Mapping[str, str],
    qrels: Sequence[tuple[str, str]],
    *,
    query_count: int = EVALUATION_QUERIES,
    document_count: int = EVALUATION_DOCUMENTS,
) -> EvaluationData:
    """Select real qrel pairs, then add deterministic corpus distractors."""

    if query_count <= 0 or document_count < query_count:
        raise ValueError("evaluation bounds require at least one document per query")
    selected_pairs = []
    seen_queries = set()
    for query_id, document_id in qrels:
        if query_id in seen_queries:
            continue
        if query_id not in queries or document_id not in documents:
            continue
        selected_pairs.append((query_id, document_id))
        seen_queries.add(query_id)
        if len(selected_pairs) == query_count:
            break
    if len(selected_pairs) != query_count:
        raise ValueError(
            f"bounded evaluation found {len(selected_pairs)} qrel pairs; "
            f"expected {query_count}"
        )
    selected_document_ids = [document_id for _, document_id in selected_pairs]
    for document_id in documents:
        if document_id not in selected_document_ids:
            selected_document_ids.append(document_id)
        if len(selected_document_ids) == document_count:
            break
    if len(selected_document_ids) != document_count:
        raise ValueError("bounded evaluation corpus is unexpectedly short")
    remapped_documents = {
        source_id: 10_000 + index
        for index, source_id in enumerate(selected_document_ids)
    }
    return EvaluationData(
        queries=tuple(
            (index, queries[query_id])
            for index, (query_id, _) in enumerate(selected_pairs)
        ),
        documents=tuple(
            (remapped_documents[source_id], documents[source_id])
            for source_id in selected_document_ids
        ),
        relevant_documents={
            index: frozenset((remapped_documents[document_id],))
            for index, (_, document_id) in enumerate(selected_pairs)
        },
        query_source_ids=tuple(query_id for query_id, _ in selected_pairs),
        document_source_ids=tuple(selected_document_ids),
    )


def _serialize_evaluation(value: EvaluationData) -> dict[str, Any]:
    query_sources = value.query_source_ids or tuple(
        str(identifier) for identifier, _ in value.queries
    )
    document_sources = value.document_source_ids or tuple(
        str(identifier) for identifier, _ in value.documents
    )
    return {
        "queries": [
            {"id": identifier, "source_id": source_id, "text": text}
            for (identifier, text), source_id in zip(
                value.queries, query_sources, strict=True
            )
        ],
        "documents": [
            {"id": identifier, "source_id": source_id, "text": text}
            for (identifier, text), source_id in zip(
                value.documents, document_sources, strict=True
            )
        ],
        "relevant_documents": {
            str(query_id): sorted(documents)
            for query_id, documents in value.relevant_documents.items()
        },
    }


def _load_evaluation(path: Path) -> EvaluationData:
    value = _document(path)
    queries = tuple((int(row["id"]), str(row["text"])) for row in value["queries"])
    documents = tuple(
        (int(row["id"]), str(row["text"])) for row in value["documents"]
    )
    relevant = {
        int(query_id): frozenset(int(document) for document in document_ids)
        for query_id, document_ids in value["relevant_documents"].items()
    }
    query_ids = {identifier for identifier, _ in queries}
    document_ids = {identifier for identifier, _ in documents}
    if query_ids != set(relevant):
        raise ValueError("evaluation queries and qrels differ")
    if set().union(*relevant.values()) - document_ids:
        raise ValueError("evaluation qrels reference missing documents")
    return EvaluationData(
        queries=queries,
        documents=documents,
        relevant_documents=relevant,
        query_source_ids=tuple(
            str(row.get("source_id", row["id"])) for row in value["queries"]
        ),
        document_source_ids=tuple(
            str(row.get("source_id", row["id"])) for row in value["documents"]
        ),
    )


def _download_dataset_file(
    directory: Path,
    *,
    repo_id: str,
    revision: str,
    filename: str,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=directory / "sources" / repo_id,
        )
    ).resolve()


def select_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> tuple[dict[str, str], ...]:
    """Select the first distinct-query positive pairs in source order."""

    selected = []
    seen_queries = set()
    for row in rows:
        query = str(row["query"])
        positive = str(row["positive"])
        if not query or not positive or query in seen_queries:
            continue
        selected.append({"query": query, "positive": positive})
        seen_queries.add(query)
        if len(selected) == count:
            break
    return tuple(selected)


def _training_rows(path: Path, count: int) -> tuple[dict[str, str], ...]:
    import pyarrow.parquet as parquet

    source = parquet.ParquetFile(path, memory_map=True)
    selected: tuple[dict[str, str], ...] = ()
    candidates = []
    for index in range(source.num_row_groups):
        candidates.extend(
            source.read_row_group(index, columns=("query", "positive")).to_pylist()
        )
        selected = select_training_rows(candidates, count=count)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("pinned MS MARCO training slice is incomplete")
    return selected


def _prepare_msmarco(directory: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    training_path = _download_dataset_file(
        directory,
        repo_id=TRAIN_DATASET_ID,
        revision=TRAIN_DATASET_REVISION,
        filename=TRAIN_DATASET_FILE,
    )
    evaluation_paths = {
        kind: _download_dataset_file(
            directory,
            repo_id=EVALUATION_DATASET_ID,
            revision=EVALUATION_DATASET_REVISION,
            filename=f"{kind}/{EVALUATION_SUBSET}-00000-of-00001.parquet",
        )
        for kind in ("queries", "corpus", "qrels")
    }
    training = _training_rows(training_path, TRAINING_ROWS)
    _write_jsonl(directory / "train.jsonl", training)
    queries = {
        str(row["_id"]): str(row["text"])
        for row in parquet.read_table(evaluation_paths["queries"]).to_pylist()
    }
    documents = {
        str(row["_id"]): "\n".join(
            part for part in (str(row.get("title", "")), str(row["text"])) if part
        )
        for row in parquet.read_table(evaluation_paths["corpus"]).to_pylist()
    }
    qrels = tuple(
        (str(row["query-id"]), str(row["corpus-id"]))
        for row in parquet.read_table(evaluation_paths["qrels"]).to_pylist()
        if float(row.get("score", 1)) > 0
    )
    evaluation = select_bounded_evaluation(queries, documents, qrels)
    _write_json(
        directory / "msmarco-evaluation.json",
        _serialize_evaluation(evaluation),
    )
    files = {
        "training": training_path,
        **{f"evaluation_{kind}": path for kind, path in evaluation_paths.items()},
    }
    return {
        "training": {
            "repo_id": TRAIN_DATASET_ID,
            "revision": TRAIN_DATASET_REVISION,
            "filename": TRAIN_DATASET_FILE,
            "rows": TRAINING_ROWS,
            "materialized_path": str(directory / "train.jsonl"),
            "materialized_sha256": _sha256(directory / "train.jsonl"),
        },
        "evaluation": {
            "repo_id": EVALUATION_DATASET_ID,
            "revision": EVALUATION_DATASET_REVISION,
            "subset": EVALUATION_SUBSET,
            "queries": len(evaluation.queries),
            "documents": len(evaluation.documents),
            "path": str(directory / "msmarco-evaluation.json"),
            "sha256": _sha256(directory / "msmarco-evaluation.json"),
        },
        "source_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in files.items()
        },
    }


def _prepare_nq_confirmation(directory: Path, source: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    source_manifest = _document(source / "manifest.json")
    contract = source_manifest.get("contract", {})
    evaluation_datasets = {
        tuple(value) for value in contract.get("evaluation_datasets", ())
    }
    if (NQ_DATASET_ID, NQ_DATASET_REVISION) not in evaluation_datasets:
        raise ValueError("NQ confirmation source is not the pinned MTEB NQ artifact")
    source_evaluation = source / "evaluation.json"
    evaluation = _load_evaluation(source_evaluation)
    destination = directory / "nq-evaluation.json"
    shutil.copyfile(source_evaluation, destination)
    return {
        "repo_id": NQ_DATASET_ID,
        "revision": NQ_DATASET_REVISION,
        "queries": len(evaluation.queries),
        "documents": len(evaluation.documents),
        "path": str(destination),
        "sha256": _sha256(destination),
        "source_manifest": str(source / "manifest.json"),
        "source_manifest_sha256": _sha256(source / "manifest.json"),
    }


def prepare_inputs(
    output: Path,
    *,
    nq_source: Path = DEFAULT_NQ_SOURCE,
) -> dict[str, Any]:
    """Materialize pinned MS MARCO train/eval and bounded NQ confirmation."""

    from huggingface_hub import snapshot_download

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"input directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    msmarco = _prepare_msmarco(output)
    nq = _prepare_nq_confirmation(output, nq_source)
    tokenizer_path = output / "tokenizer"
    snapshot_download(
        TOKENIZER_ID,
        revision=TOKENIZER_REVISION,
        allow_patterns=(
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ),
        local_dir=tokenizer_path,
    )
    manifest = {
        "schema_version": "representax-bert-scaling-retrieval-inputs-v1",
        "msmarco": msmarco,
        "nq_confirmation": nq,
        "tokenizer": {
            "repo_id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "path": str(tokenizer_path),
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def make_bert_ladder_processor(
    tokenizer_path: str | Path,
    *,
    sequence_length: int,
) -> Any:
    """Build the pinned tokenizer boundary without allocating model weights."""

    from transformers import AutoTokenizer

    from representax.models import make_text_processor
    from representax.models.bert import BertEncoder

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return make_text_processor(
        tokenizer=tokenizer,
        batch_builder=BertEncoder.make_batch,
        sequence_length_buckets=(sequence_length,),
    )


def load_bert_ladder_model(
    size: str,
    tokenizer_path: str | Path,
    *,
    sequence_length: int,
    parameter_dtype: str = "bfloat16",
    compute_dtype: str = "bfloat16",
    rematerialization: str = "full",
    initialization_device: str = "default",
    key: Any,
) -> tuple[Any, Any]:
    """Initialize one native random-weight BERT and its pinned text processor."""

    import jax
    import jax.numpy as jnp

    from representax.models.bert import BertConfig, BertEncoder

    entry = ladder_entry(size)
    if sequence_length > entry.max_position_embeddings:
        raise ValueError("sequence length exceeds the frozen BERT position table")
    if initialization_device not in {"cpu", "default"}:
        raise ValueError("initialization_device must be 'cpu' or 'default'")
    config = BertConfig(**entry.config_values())
    cpu_device = (
        jax.devices("cpu")[0] if initialization_device == "cpu" else None
    )
    with jax.default_device(cpu_device):
        initialization_key = (
            jax.device_put(key, cpu_device) if cpu_device is not None else key
        )
        model = BertEncoder.init(
            config,
            key=initialization_key,
            parameter_dtype=jnp.dtype(parameter_dtype),
            compute_dtype=jnp.dtype(compute_dtype),
            rematerialization=rematerialization,
            model_id=f"representax/{size}",
            revision="bert-scaling-v1-random-init",
        )
    return model, make_bert_ladder_processor(
        tokenizer_path,
        sequence_length=sequence_length,
    )


def build_job(
    entry: LadderEntry,
    *,
    data_directory: Path,
    msmarco_evaluation: EvaluationData,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    steps: int = DEFAULT_STEPS,
) -> Any:
    """Build the canonical MS MARCO MNR lifecycle used by every ladder size."""

    if steps < 3:
        raise ValueError("acceptance needs compile, steady, and reload updates")
    if not 1 <= sequence_length <= entry.max_position_embeddings:
        raise ValueError("sequence length is outside the frozen BERT contract")
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        ExportConfig,
        FSDPConfig,
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

    sharded = entry.name in SHARDED_SIZES
    training_arguments: dict[str, Any] = {}
    if sharded:
        training_arguments.update(
            mesh=MeshConfig(axis_shapes=(2,), axis_names=("model",)),
            sharding=FSDPConfig(
                data_axis=None,
                parameter_axis="model",
                minimum_parameter_elements=2**18,
            ),
        )
    evaluation_source = "bert-ladder-evaluation://msmarco"
    return JobConfig(
        name=f"paper-bert-scaling-msmarco-{entry.name}",
        model=ModelConfig(
            target="experiments.paper.bert_scaling:load_bert_ladder_model",
            parameters={
                "size": entry.name,
                "tokenizer_path": str(data_directory / "tokenizer"),
                "sequence_length": sequence_length,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "rematerialization": "full",
                "initialization_device": "cpu" if sharded else "default",
            },
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=20.0, symmetric=False),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={
                    "learning_rate": 1e-5,
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data_directory / "train.jsonl"), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval:RetrievalCollator"
            ),
            drop_remainder=True,
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=TRAINING_BATCH_SIZE,
            max_steps=steps,
            seed=42,
            batch=BatchConfig(micro_batch_size=TRAINING_BATCH_SIZE),
            precision=PrecisionConfig.bfloat16_mixed(),
            activation_rematerialization="full",
            donate_buffers=True,
            **training_arguments,
        ),
        checkpointing=CheckpointConfig(
            every=steps - 1,
            keep=1,
            save_final=False,
            asynchronous=False,
        ),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        evaluation=EvaluationConfig(
            data=DataConfig(
                distribution=mix(
                    source(evaluation_source, map=identity),
                    shuffle=False,
                ),
                collate=ComponentConfig(
                    target=(
                        "experiments.paper.dense_retrieval:"
                        "RetrievalEvaluationCollator"
                    )
                ),
                drop_remainder=True,
                num_threads=0,
                prefetch_buffer_size=0,
            ),
            batch_size=EVALUATION_BATCH_SIZE,
            evaluators=(
                InformationRetrievalEvaluatorConfig(
                    name="msmarco-bounded",
                    relevant_documents=dict(msmarco_evaluation.relevant_documents),
                    score_functions=("cosine",),
                    main_score_function="cosine",
                    accuracy_at_k=(1, 4),
                    precision_recall_at_k=(1, 4, 10, 100),
                    mrr_at_k=(4,),
                    ndcg_at_k=(4, 10),
                    map_at_k=(4,),
                ),
            ),
            on_start=True,
            on_end=True,
            primary_metric="valid/msmarco-bounded/cosine_ndcg@10",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(enabled=True, selection="final"),
    )


def _visible_physical_gpus() -> tuple[int, ...]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        return ()
    try:
        return tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must contain physical GPU indices"
        ) from error


def _metric_rows(path: Path, event: str) -> tuple[dict[str, Any], ...]:
    return tuple(
        row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
        if row.get("event") == event
    )


def _parameter_count(model: Any) -> int:
    import equinox as eqx
    import jax

    return sum(
        int(value.size)
        for value in jax.tree.leaves(model)
        if eqx.is_array(value)
    )


def _model_digest(model: Any) -> str:
    """Hash every array leaf after gathering its complete logical value."""

    import equinox as eqx
    import jax
    import numpy as np

    digest = hashlib.sha256()
    for value in jax.tree.leaves(model):
        if not eqx.is_array(value):
            continue
        host = np.asarray(jax.device_get(value))
        digest.update(str(host.dtype).encode())
        digest.update(str(host.shape).encode())
        digest.update(host.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _device_memory() -> tuple[dict[str, Any], ...]:
    import jax

    output = []
    for device in jax.devices("gpu"):
        stats = device.memory_stats() or {}
        output.append(
            {
                "logical_id": device.id,
                "device_kind": device.device_kind,
                "bytes_in_use": stats.get("bytes_in_use"),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
                "bytes_limit": stats.get("bytes_limit"),
            }
        )
    return tuple(output)


def _timings(updates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    compile_rows = []
    steady_rows = []
    for row in updates:
        metrics = row["metrics"]
        if "perf/compilation_and_first_step_seconds" in metrics:
            compile_rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "seconds": float(
                        metrics["perf/compilation_and_first_step_seconds"]
                    ),
                }
            )
        elif "perf/step_seconds" in metrics:
            steady_rows.append(
                {
                    "iteration": int(row["iteration"]),
                    "seconds": float(metrics["perf/step_seconds"]),
                }
            )
    if not compile_rows:
        raise RuntimeError("run_job did not report compile-plus-first-update timing")
    if not steady_rows:
        raise RuntimeError("run_job did not report a steady update timing")
    return {
        "compile_and_first_update": compile_rows,
        "steady_updates": steady_rows,
        "steady_median_seconds": statistics.median(
            row["seconds"] for row in steady_rows
        ),
    }


def _evaluation_records(data: EvaluationData) -> tuple[Any, ...]:
    from experiments.paper.dense_retrieval import evaluation_rows

    return evaluation_rows(
        data.queries,
        data.documents,
        batch_size=EVALUATION_BATCH_SIZE,
    )


def _evaluation_batches(processor: Any, data: EvaluationData) -> tuple[Any, ...]:
    from experiments.paper.dense_retrieval import RetrievalEvaluationCollator

    rows = _evaluation_records(data)
    collate = RetrievalEvaluationCollator(processor)
    return tuple(
        collate(rows[start : start + EVALUATION_BATCH_SIZE])
        for start in range(0, len(rows), EVALUATION_BATCH_SIZE)
    )


def _canonical_evaluation(
    name: str,
    model: Any,
    processor: Any,
    data: EvaluationData,
    *,
    iteration: int,
    place_batch: Any = None,
) -> dict[str, float]:
    from representax.config import PrecisionConfig
    from representax.evaluation import InformationRetrievalEvaluator
    from representax.precision import resolve_precision_policy
    from representax.train.evaluation import EvaluationRunner

    evaluator = InformationRetrievalEvaluator(
        name=name,
        relevant_documents=data.relevant_documents,
        score_functions=("cosine",),
        main_score_function="cosine",
        accuracy_at_k=(1, 4),
        precision_recall_at_k=(1, 4, 10, 100),
        mrr_at_k=(4,),
        ndcg_at_k=(4, 10),
        map_at_k=(4,),
    )
    runner = EvaluationRunner(
        evaluator,
        precision=resolve_precision_policy(PrecisionConfig.bfloat16_mixed()),
    )
    arguments = {}
    if place_batch is not None:
        arguments["place_batch"] = place_batch
    result = runner.run(
        model,
        _evaluation_batches(processor, data),
        iteration=iteration,
        **arguments,
    )
    return {name: float(value) for name, value in result.metrics.items()}


def _evaluation_layout(
    entry: LadderEntry,
    model: Any,
    *,
    place_parameters: bool,
) -> tuple[Any, Any]:
    """Recreate the run_job evaluation placement for a native model."""

    import equinox as eqx
    import jax

    if entry.name not in SHARDED_SIZES:
        return model, jax.device_put

    from representax.train.sharding import fsdp_parameter_specs, place_model

    mesh = jax.make_mesh(
        (len(GPU_ASSIGNMENTS[entry.name]),),
        ("model",),
        axis_types=(jax.sharding.AxisType.Auto,),
        devices=jax.devices("gpu")[: len(GPU_ASSIGNMENTS[entry.name])],
    )
    if place_parameters:
        model = place_model(
            model,
            mesh,
            fsdp_parameter_specs(
                model,
                mesh,
                axis_name="model",
                minimum_elements=2**18,
            ),
        )
    batch_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(),
    )

    def place_batch(tree: Any) -> Any:
        return jax.tree.map(
            lambda value: (
                jax.device_put(value, batch_sharding)
                if eqx.is_array(value)
                else value
            ),
            tree,
            is_leaf=lambda value: value is None,
        )

    return model, place_batch


def _single_device_evaluation_layout(model: Any) -> tuple[Any, Any]:
    """Place a live or reloaded model on the canonical GPU-2 parity topology."""

    import equinox as eqx
    import jax

    device = jax.devices("gpu")[0]
    model = jax.tree.map(
        lambda value: jax.device_put(value, device) if eqx.is_array(value) else value,
        model,
    )
    return model, jax.device_put


def run_size(
    entry: LadderEntry,
    *,
    data_directory: Path,
    output: Path,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Run one dense-retrieval lifecycle entirely through ``run_job``."""

    import jax

    from experiments.paper.dense_retrieval import fixed_rows_resolver
    from representax import load_inference_bundle
    from representax.train import run_job

    expected_gpus = GPU_ASSIGNMENTS[entry.name]
    visible_gpus = _visible_physical_gpus()
    if visible_gpus != expected_gpus:
        raise RuntimeError(
            f"{entry.name} requires physical GPUs {expected_gpus}; "
            f"CUDA_VISIBLE_DEVICES exposes {visible_gpus}"
        )
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != len(
        expected_gpus
    ):
        raise RuntimeError(f"{entry.name} has an invalid JAX GPU topology")
    inputs = _document(data_directory / "manifest.json")
    msmarco = _load_evaluation(Path(inputs["msmarco"]["evaluation"]["path"]))
    nq = _load_evaluation(Path(inputs["nq_confirmation"]["path"]))
    processor = make_bert_ladder_processor(
        inputs["tokenizer"]["path"],
        sequence_length=sequence_length,
    )
    job = build_job(
        entry,
        data_directory=data_directory,
        msmarco_evaluation=msmarco,
        sequence_length=sequence_length,
        steps=steps,
    )
    output.mkdir(parents=True, exist_ok=False)
    run_directory = output / "run"
    resolvers = {
        "bert-ladder-evaluation": fixed_rows_resolver(
            _evaluation_records(msmarco)
        )
    }

    started = time.perf_counter()
    paused = run_job(
        job,
        run_directory,
        resolvers=resolvers,
        stop_after=steps - 1,
    )
    if paused.completed_iterations != steps - 1:
        raise RuntimeError("run_job did not stop at the reload checkpoint")
    paused_parameter_count = _parameter_count(paused.state.model)
    del paused
    gc.collect()

    completed = run_job(job, run_directory, resolvers=resolvers, resume=True)
    jax.block_until_ready(completed.state)
    if not completed.resumed or completed.completed_iterations != steps:
        raise RuntimeError("run_job did not reload and finish the lifecycle")
    if completed.inference_bundle is None:
        raise RuntimeError("run_job did not export a native inference bundle")
    final_parameter_count = _parameter_count(completed.state.model)
    inference_bundle = completed.inference_bundle
    if {paused_parameter_count, final_parameter_count} != {
        entry.expected_parameters
    }:
        raise RuntimeError("allocated parameter count differs from the frozen manifest")

    metrics_path = run_directory / "metrics.jsonl"
    updates = _metric_rows(metrics_path, "training_step")
    evaluations = _metric_rows(metrics_path, "evaluation")
    if tuple(int(row["iteration"]) for row in updates) != tuple(range(1, steps + 1)):
        raise RuntimeError("training metric stream is incomplete after reload")
    if len(evaluations) < 2 or int(evaluations[-1]["iteration"]) != steps:
        raise RuntimeError("held-out evaluation did not run on start and end")
    losses = tuple(float(row["metrics"]["train/loss"]) for row in updates)
    update_norms = tuple(
        float(row["metrics"]["train/update_global_norm"]) for row in updates
    )
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("BERT retrieval update produced a non-finite loss")
    if not all(
        bool(row["metrics"]["train/numeric_finite"])
        and not bool(row["metrics"]["train/skipped_update"])
        for row in updates
    ):
        raise RuntimeError("BERT retrieval update failed the numeric-finite gate")
    if not all(math.isfinite(norm) and norm > 0.0 for norm in update_norms):
        raise RuntimeError("BERT retrieval optimizer did not make a real update")
    final_metrics = {
        name: float(value)
        for name, value in evaluations[-1]["metrics"].items()
        if name.startswith("valid/")
    }
    live_model, live_place_batch = _evaluation_layout(
        entry,
        completed.state.model,
        place_parameters=False,
    )
    live_msmarco = _canonical_evaluation(
        "msmarco-bounded",
        live_model,
        processor,
        msmarco,
        iteration=steps,
        place_batch=live_place_batch,
    )
    live_parity_error = max(
        abs(live_msmarco[name] - final_metrics[name]) for name in final_metrics
    )
    live_model_digest = _model_digest(live_model)
    training_device_memory = _device_memory()
    parity_reference_model, parity_place_batch = _single_device_evaluation_layout(
        live_model
    )
    parity_reference_msmarco = _canonical_evaluation(
        "msmarco-bounded",
        parity_reference_model,
        processor,
        msmarco,
        iteration=steps,
        place_batch=parity_place_batch,
    )
    topology_metric_difference = max(
        abs(parity_reference_msmarco[name] - final_metrics[name])
        for name in final_metrics
    )
    del parity_reference_model
    checkpoint_manifests = sorted(
        path
        for path in (run_directory / "checkpoints").rglob("checkpoint.json")
        if (path.parent / "REPRESENTAX_COMPLETE").is_file()
    )
    if not checkpoint_manifests:
        raise RuntimeError("run_job did not leave a complete checkpoint manifest")
    if not (inference_bundle / "manifest.json").is_file() or not (
        inference_bundle / "REPRESENTAX_COMPLETE"
    ).is_file():
        raise RuntimeError("native inference bundle is incomplete")

    del completed, live_model
    gc.collect()
    jax.clear_caches()
    reloaded_model, reloaded_job = load_inference_bundle(inference_bundle)
    reloaded_parameter_count = _parameter_count(reloaded_model)
    reloaded_model, reloaded_place_batch = _single_device_evaluation_layout(
        reloaded_model
    )
    reloaded_model_digest = _model_digest(reloaded_model)
    exported_msmarco = _canonical_evaluation(
        "msmarco-bounded",
        reloaded_model,
        processor,
        msmarco,
        iteration=steps,
        place_batch=reloaded_place_batch,
    )
    exported_nq = _canonical_evaluation(
        "nq-bounded",
        reloaded_model,
        processor,
        nq,
        iteration=steps,
        place_batch=reloaded_place_batch,
    )
    if exported_msmarco.keys() != parity_reference_msmarco.keys():
        raise RuntimeError("exported MS MARCO evaluation metric set changed")
    parity_error = max(
        abs(exported_msmarco[name] - parity_reference_msmarco[name])
        for name in parity_reference_msmarco
    )
    if (
        live_parity_error > 1e-7
        or parity_error > 1e-7
        or live_model_digest != reloaded_model_digest
    ):
        _write_json(
            output / "failure.json",
            {
                "schema_version": "representax-bert-scaling-failure-v1",
                "status": "failed_incomplete",
                "size": entry.name,
                "phase": "native_export_reload_parity",
                "configured_final_msmarco": final_metrics,
                "live_final_msmarco": live_msmarco,
                "parity_reference_msmarco": parity_reference_msmarco,
                "exported_msmarco": exported_msmarco,
                "live_evaluation_max_absolute_error": live_parity_error,
                "configured_vs_parity_topology_max_absolute_difference": (
                    topology_metric_difference
                ),
                "export_reload_max_absolute_error": parity_error,
                "live_model_digest": live_model_digest,
                "reloaded_model_digest": reloaded_model_digest,
                "nq_confirmation_completed": True,
                "accepted_result_written": False,
            },
        )
        raise RuntimeError("native export/reload failed exact held-out parity")
    if reloaded_parameter_count != entry.expected_parameters:
        raise RuntimeError("native export/reload changed the parameter count")
    jax.block_until_ready(reloaded_model)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "schema_version": "representax-bert-scaling-retrieval-result-v1",
        "status": "accepted",
        "size": entry.name,
        "architecture": asdict(entry),
        "parameter_validation": {
            "manifest_expected": entry.expected_parameters,
            "before_reload": paused_parameter_count,
            "after_checkpoint_reload": final_parameter_count,
            "after_export_reload": reloaded_parameter_count,
            "live_model_digest": live_model_digest,
            "reloaded_model_digest": reloaded_model_digest,
        },
        "data": inputs,
        "training": {
            "entrypoint": "representax.train.run_job",
            "task": "retrieval",
            "loss": "mnr",
            "mnr_scale": 20.0,
            "mnr_symmetric": False,
            "steps": steps,
            "global_batch_size": TRAINING_BATCH_SIZE,
            "sequence_length": sequence_length,
            "precision": PRECISION_LABEL,
            "precision_recipe": dict(PRECISION_RECIPE),
            "optimizer": "adamw",
            "activation_rematerialization": "full",
            "sharding": (
                "fsdp-model-axis" if entry.name in SHARDED_SIZES else "single"
            ),
        },
        "devices": {
            "physical_gpu_indices": list(visible_gpus),
            "training_jax": list(training_device_memory),
            "jax": list(_device_memory()),
        },
        "lifecycle": {
            "updates": len(updates),
            "losses": list(losses),
            "update_global_norms": list(update_norms),
            "checkpoint_manifests": [
                str(path.relative_to(run_directory))
                for path in checkpoint_manifests
            ],
            "resumed": True,
            "inference_bundle": str(inference_bundle),
            "reloaded_job_name": reloaded_job.name,
        },
        "evaluation": {
            "configured_events": len(evaluations),
            "configured_final_msmarco": final_metrics,
            "live_final_msmarco": live_msmarco,
            "live_evaluation_max_absolute_error": live_parity_error,
            "parity_topology": {
                "physical_gpu_indices": [visible_gpus[0]],
                "purpose": "native-live-vs-reloaded-evaluation-parity",
            },
            "parity_reference_msmarco": parity_reference_msmarco,
            "configured_vs_parity_topology_max_absolute_difference": (
                topology_metric_difference
            ),
            "exported_msmarco": exported_msmarco,
            "export_reload_max_absolute_error": parity_error,
            "exported_nq_confirmation": exported_nq,
            "msmarco_queries": len(msmarco.queries),
            "msmarco_documents": len(msmarco.documents),
            "nq_queries": len(nq.queries),
            "nq_documents": len(nq.documents),
        },
        "timing": _timings(updates),
        "elapsed_seconds": elapsed_seconds,
    }
    _write_json(output / "result.json", result)
    return result


def four_b_feasibility(
    one_b_result: Mapping[str, Any] | None,
    *,
    device_total_bytes: Sequence[int],
) -> dict[str, Any]:
    """Apply the dense-retrieval 1B and HBM gates without allocating 4B."""

    one_b = ladder_entry("bert-1b")
    four_b = ladder_entry("bert-4b")
    device_count = len(GPU_ASSIGNMENTS["bert-4b"])
    if len(device_total_bytes) != device_count or any(
        value <= 0 for value in device_total_bytes
    ):
        raise ValueError("4B feasibility requires positive HBM totals for GPUs 2-5")
    planning_peak_per_device = math.ceil(
        four_b.expected_parameters
        * ACTIVE_UPDATE_BYTES_PER_PARAMETER
        / device_count
    )
    observed_peak = None
    projected_peak = None
    one_b_practical = bool(
        one_b_result is not None
        and one_b_result.get("status") == "accepted"
        and one_b_result.get("training", {}).get("task") == "retrieval"
        and one_b_result.get("training", {}).get("loss") == "mnr"
        and one_b_result.get("training", {}).get("precision")
        == PRECISION_LABEL
        and one_b_result.get("training", {}).get("precision_recipe")
        == PRECISION_RECIPE
        and one_b_result.get("evaluation", {}).get(
            "export_reload_max_absolute_error"
        )
        is not None
        and one_b_result.get("evaluation", {}).get("exported_nq_confirmation")
    )
    if one_b_practical:
        observed = [
            row.get("peak_bytes_in_use")
            for row in one_b_result["devices"]["training_jax"]
        ]
        if observed and all(isinstance(value, int) and value > 0 for value in observed):
            observed_peak = max(observed)
            projected_peak = math.ceil(
                observed_peak
                * four_b.expected_parameters
                / one_b.expected_parameters
                * len(GPU_ASSIGNMENTS["bert-1b"])
                / device_count
            )
    estimated_peak = max(
        value
        for value in (planning_peak_per_device, projected_peak)
        if value is not None
    )
    admission_limit = math.floor(min(device_total_bytes) * HBM_ADMISSION_FRACTION)
    reasons = []
    if not one_b_practical:
        reasons.append("bert-1b-dense-retrieval-pilot-not-accepted")
    if estimated_peak > admission_limit:
        reasons.append("projected-active-update-hbm-exceeds-85-percent-reserve")
    admitted = not reasons
    return {
        "schema_version": "representax-bert-4b-retrieval-feasibility-v1",
        "size": "bert-4b",
        "status": (
            "admitted_pending_four_gpu_canary" if admitted else "not_run"
        ),
        "decision": "pending_four_gpu_canary" if admitted else "skip",
        "admission_gate": four_b.admission_gate,
        "one_b_dense_retrieval_practical": one_b_practical,
        "physical_gpu_indices": list(GPU_ASSIGNMENTS["bert-4b"]),
        "required_canary": {
            "updates": 1,
            "compiled": True,
            "devices": 4,
            "pending_gpu_indices": [4, 5],
        },
        "device_total_bytes": list(device_total_bytes),
        "hbm_admission_fraction": HBM_ADMISSION_FRACTION,
        "admission_limit_bytes_per_device": admission_limit,
        "planning": {
            "active_update_bytes_per_parameter": ACTIVE_UPDATE_BYTES_PER_PARAMETER,
            "estimated_peak_bytes_per_device": planning_peak_per_device,
        },
        "one_b_observed_peak_bytes_per_device": observed_peak,
        "one_b_observed_device_count": len(GPU_ASSIGNMENTS["bert-1b"]),
        "four_b_projected_peak_bytes_per_device": projected_peak,
        "effective_estimated_peak_bytes_per_device": estimated_peak,
        "reasons": reasons,
    }


def _gpu_memory_totals(indices: Sequence[int]) -> tuple[int, ...]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    totals = {}
    for line in process.stdout.splitlines():
        index, memory_mib = (part.strip() for part in line.split(",", 1))
        totals[int(index)] = int(memory_mib) * 1024**2
    return tuple(totals[index] for index in indices)


def _worker_command(
    entry: LadderEntry,
    *,
    data_directory: Path,
    output: Path,
    sequence_length: int,
    steps: int,
) -> subprocess.CompletedProcess[str]:
    gpus = GPU_ASSIGNMENTS[entry.name]
    environment = dict(os.environ)
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=",".join(str(index) for index in gpus),
        JAX_DEFAULT_MATMUL_PRECISION="highest",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    python_path = [str(ROOT / "src"), str(ROOT)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.paper.bert_scaling",
            "worker",
            "--size",
            entry.name,
            "--data-directory",
            str(data_directory),
            "--output",
            str(output),
            "--sequence-length",
            str(sequence_length),
            "--steps",
            str(steps),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )


def run_sweep(
    artifact_root: Path,
    *,
    nq_source: Path = DEFAULT_NQ_SOURCE,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    steps: int = DEFAULT_STEPS,
) -> dict[str, Any]:
    """Run mandatory retrieval sizes sequentially and conditionally admit 4B."""

    inputs = artifact_root / "inputs"
    runs = artifact_root / "runs"
    if artifact_root.exists():
        raise FileExistsError(f"artifact root already exists: {artifact_root}")
    artifact_root.mkdir(parents=True)
    prepare_inputs(inputs, nq_source=nq_source)
    entries = {entry.name: entry for entry in load_ladder()}
    results: dict[str, Any] = {}
    for name in SIZE_ORDER[:-1]:
        output = runs / name
        process = _worker_command(
            entries[name],
            data_directory=inputs,
            output=output,
            sequence_length=sequence_length,
            steps=steps,
        )
        if process.returncode == 0:
            results[name] = _document(output / "result.json")
        else:
            failure = {
                "schema_version": "representax-bert-scaling-retrieval-failure-v1",
                "status": "failed",
                "size": name,
                "returncode": process.returncode,
                "stdout_tail": process.stdout[-8_000:],
                "stderr_tail": process.stderr[-8_000:],
            }
            _write_json(output / "failure.json", failure)
            results[name] = failure
            if name == "bert-1b":
                break
    feasibility = four_b_feasibility(
        results.get("bert-1b"),
        device_total_bytes=_gpu_memory_totals(GPU_ASSIGNMENTS["bert-4b"]),
    )
    _write_json(runs / "bert-4b" / "feasibility.json", feasibility)
    results["bert-4b"] = feasibility
    summary = {
        "schema_version": "representax-bert-scaling-retrieval-campaign-v1",
        "artifact_root": str(artifact_root),
        "manifest": str(LADDER_MANIFEST),
        "sequence_length": sequence_length,
        "steps": steps,
        "inputs": _document(inputs / "manifest.json"),
        "results": results,
    }
    _write_json(artifact_root / "summary.json", summary)
    return summary


def summarize_existing(
    result_paths: Sequence[Path],
    *,
    feasibility_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze selected retrieval workers and the 4B gate into one artifact."""

    results = {}
    for path in result_paths:
        result = _document(path)
        name = str(result.get("size"))
        if name in results:
            raise ValueError(f"duplicate selected result for {name}")
        if result.get("status") != "accepted":
            raise ValueError(f"selected result is not accepted: {path}")
        training = result.get("training", {})
        evaluation = result.get("evaluation", {})
        parameters = result.get("parameter_validation", {})
        if (
            training.get("precision") != PRECISION_LABEL
            or training.get("precision_recipe") != PRECISION_RECIPE
        ):
            raise ValueError(f"selected result has a stale precision recipe: {path}")
        if (
            evaluation.get("live_evaluation_max_absolute_error") != 0.0
            or evaluation.get("export_reload_max_absolute_error") != 0.0
            or not evaluation.get("exported_nq_confirmation")
        ):
            raise ValueError(f"selected result failed evaluator parity: {path}")
        if parameters.get("live_model_digest") != parameters.get(
            "reloaded_model_digest"
        ):
            raise ValueError(f"selected result failed native parameter parity: {path}")
        results[name] = result
    required = set(SIZE_ORDER[:-1])
    if set(results) != required:
        raise ValueError(
            f"selected results must be exactly {sorted(required)}; "
            f"received {sorted(results)}"
        )
    feasibility = _document(feasibility_path)
    if feasibility.get("size") != "bert-4b":
        raise ValueError("conditional feasibility record must describe bert-4b")
    summary = {
        "schema_version": "representax-bert-scaling-retrieval-campaign-v1",
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__)),
        },
        "ladder_manifest": {
            "path": str(LADDER_MANIFEST),
            "sha256": _sha256(LADDER_MANIFEST),
        },
        "inputs": results["bert-30m"]["data"],
        "accepted_sizes": [name for name in SIZE_ORDER if name in results],
        "results": {name: results[name] for name in SIZE_ORDER if name in results},
        "bert-4b": feasibility,
    }
    _write_json(output, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate frozen configs")
    validate.add_argument("--manifest", type=Path, default=LADDER_MANIFEST)

    prepare = commands.add_parser("prepare", help="materialize pinned retrieval data")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--nq-source", type=Path, default=DEFAULT_NQ_SOURCE)

    worker = commands.add_parser("worker", help="run one isolated ladder size")
    worker.add_argument("--size", choices=SIZE_ORDER, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument(
        "--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH
    )
    worker.add_argument("--steps", type=int, default=DEFAULT_STEPS)

    gate = commands.add_parser("gate-4b", help="record conditional 4B feasibility")
    gate.add_argument("--one-b-result", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)

    summarize = commands.add_parser(
        "summarize", help="freeze selected worker and feasibility results"
    )
    summarize.add_argument("--result", type=Path, action="append", required=True)
    summarize.add_argument("--four-b-feasibility", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    sweep = commands.add_parser("sweep", help="run mandatory sizes and gate 4B")
    sweep.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    sweep.add_argument("--nq-source", type=Path, default=DEFAULT_NQ_SOURCE)
    sweep.add_argument(
        "--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH
    )
    sweep.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "validate":
        print(
            json.dumps(
                [asdict(entry) for entry in load_ladder(arguments.manifest)],
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_inputs(arguments.output, nq_source=arguments.nq_source),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if arguments.command == "worker":
        result = run_size(
            ladder_entry(arguments.size),
            data_directory=arguments.data_directory,
            output=arguments.output,
            sequence_length=arguments.sequence_length,
            steps=arguments.steps,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.command == "gate-4b":
        result = four_b_feasibility(
            _document(arguments.one_b_result),
            device_total_bytes=_gpu_memory_totals(GPU_ASSIGNMENTS["bert-4b"]),
        )
        _write_json(arguments.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.command == "summarize":
        result = summarize_existing(
            arguments.result,
            feasibility_path=arguments.four_b_feasibility,
            output=arguments.output,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.command == "sweep":
        print(
            json.dumps(
                run_sweep(
                    arguments.artifact_root,
                    nq_source=arguments.nq_source,
                    sequence_length=arguments.sequence_length,
                    steps=arguments.steps,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
