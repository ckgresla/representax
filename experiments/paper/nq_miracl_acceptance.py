"""Bounded real-data acceptance for the NQ and MIRACL paper rows."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.request import Request, urlopen

DEFAULT_ARTIFACT_ROOT = Path(
    "/home/ckg/representax-artifacts/nq-miracl-acceptance-20260831"
)
LANGUAGES = ("ar", "en", "hi", "ja")
TRAINING_ROWS = 8
EVALUATION_QUERIES = 4
TRAINING_BATCH_SIZE = 4
EVALUATION_BATCH_SIZE = 4
STEPS = 2
MAXIMUM_LENGTH = 64
NQ_CORPUS_PREFIX_BYTES = 4 * 1024 * 1024
MIRACL_CORPUS_PREFIX_BYTES = 16 * 1024 * 1024
MIRACL_MAX_DOCUMENTS = 100_000

RowName = Literal["nq", "miracl"]


@dataclass(frozen=True, slots=True)
class RowContract:
    """Immutable external identity and bounded execution contract for one row."""

    name: RowName
    workload: str
    model_id: str
    model_revision: str
    model_target: str
    training_datasets: tuple[tuple[str, str], ...]
    evaluation_datasets: tuple[tuple[str, str], ...]
    languages: tuple[str, ...] = ()


CONTRACTS: dict[RowName, RowContract] = {
    "nq": RowContract(
        name="nq",
        workload="dense-natural-questions",
        model_id="sentence-transformers/all-mpnet-base-v2",
        model_revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        model_target="representax.models:SentenceEncoder.load_from_hf",
        training_datasets=(
            (
                "sentence-transformers/natural-questions",
                "f9e894e1081e206e577b4eaa9ee6de2b06ae6f17",
            ),
        ),
        evaluation_datasets=(
            ("mteb/nq", "b84726e65fd226125cf7c0cbeeb5c214d49e8187"),
        ),
    ),
    "miracl": RowContract(
        name="miracl",
        workload="dense-multilingual",
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        model_revision="12949877f0092093f366c6450340011320152a05",
        model_target=(
            "experiments.paper.nq_miracl_acceptance:load_miracl_text_encoder"
        ),
        training_datasets=(
            ("miracl/miracl", "5be20db9509754dadad47689368639fcec739c00"),
            (
                "miracl/miracl-corpus",
                "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            ),
        ),
        evaluation_datasets=(
            ("miracl/miracl", "5be20db9509754dadad47689368639fcec739c00"),
            (
                "miracl/miracl-corpus",
                "d921ec7e349ce0d28daf30b2da9da5ee698bef0d",
            ),
        ),
        languages=LANGUAGES,
    ),
}


@dataclass(frozen=True, slots=True)
class EvaluationData:
    queries: tuple[tuple[int, str], ...]
    documents: tuple[tuple[int, str], ...]
    relevant_documents: Mapping[int, frozenset[int]]
    query_metadata: tuple[Mapping[str, str], ...]
    document_metadata: tuple[Mapping[str, str], ...]


def frozen_contract(row: RowName) -> RowContract:
    """Return one paper-manifest-aligned row contract."""

    try:
        return CONTRACTS[row]
    except KeyError as error:  # pragma: no cover - argparse closes this path
        raise ValueError(f"unknown acceptance row {row!r}") from error


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _download(
    repo_id: str,
    revision: str,
    filename: str,
    *,
    cache_directory: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            cache_dir=cache_directory,
        )
    ).resolve()


def _dataset_url(repo_id: str, revision: str, filename: str) -> str:
    from huggingface_hub import hf_hub_url

    return hf_hub_url(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=filename,
    )


def _bounded_response(url: str, limit: int) -> tuple[bytes, Mapping[str, str]]:
    request = Request(url, headers={"Range": f"bytes=0-{limit - 1}"})
    with urlopen(request, timeout=120) as response:
        value = response.read(limit + 1)
        headers = {name.lower(): value for name, value in response.headers.items()}
    if len(value) > limit:
        raise ValueError(f"bounded source exceeded {limit} bytes: {url}")
    return value, headers


def _jsonl_prefix(value: bytes) -> tuple[dict[str, Any], ...]:
    lines = value.splitlines()
    if value and not value.endswith((b"\n", b"\r")):
        lines = lines[:-1]
    return tuple(json.loads(line) for line in lines if line.strip())


def _qrels_tsv(path: Path) -> tuple[tuple[str, str], ...]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for index, row in enumerate(reader):
            if index == 0 and row[:2] == ["query-id", "corpus-id"]:
                continue
            if len(row) == 3:
                query_id, document_id, score = row
            elif len(row) == 4:
                query_id, _, document_id, score = row
            else:
                raise ValueError(f"unexpected qrels row in {path}: {row!r}")
            if int(score) > 0:
                rows.append((query_id, document_id))
    return tuple(rows)


def _topics_tsv(path: Path) -> dict[str, str]:
    topics = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream, delimiter="\t"):
            if len(row) != 2:
                raise ValueError(f"unexpected topic row in {path}: {row!r}")
            topics[row[0]] = row[1]
    return topics


def select_nq_evaluation(
    queries: Mapping[str, str],
    corpus: Mapping[str, str],
    qrels: Sequence[tuple[str, str]],
    *,
    count: int = EVALUATION_QUERIES,
) -> EvaluationData:
    """Select the first real NQ queries whose positives are in the bounded prefix."""

    selected: list[tuple[str, str, str, str]] = []
    seen_queries = set()
    for source_query_id, source_document_id in qrels:
        if source_query_id in seen_queries:
            continue
        query = queries.get(source_query_id)
        document = corpus.get(source_document_id)
        if query is None or document is None:
            continue
        selected.append((source_query_id, query, source_document_id, document))
        seen_queries.add(source_query_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(
            f"NQ prefix yielded {len(selected)} evaluable queries; expected {count}"
        )
    return _evaluation_data(selected, languages=("en",) * count)


def _evaluation_data(
    rows: Sequence[tuple[str, str, str, str]],
    *,
    languages: Sequence[str],
) -> EvaluationData:
    if len(rows) != len(languages):
        raise ValueError("evaluation language metadata must match query rows")
    queries = []
    documents = []
    relevant_documents = {}
    query_metadata = []
    document_metadata = []
    paired = zip(rows, languages, strict=True)
    for index, (
        (query_source, query, document_source, document),
        language,
    ) in enumerate(paired):
        query_id = index
        document_id = 10_000 + index
        queries.append((query_id, query))
        documents.append((document_id, document))
        relevant_documents[query_id] = frozenset((document_id,))
        query_metadata.append(
            {"source_id": query_source, "language": language}
        )
        document_metadata.append(
            {"source_id": document_source, "language": language}
        )
    return EvaluationData(
        queries=tuple(queries),
        documents=tuple(documents),
        relevant_documents=relevant_documents,
        query_metadata=tuple(query_metadata),
        document_metadata=tuple(document_metadata),
    )


def _serialize_evaluation(value: EvaluationData) -> dict[str, Any]:
    return {
        "queries": [
            {"id": identifier, "text": text, **metadata}
            for (identifier, text), metadata in zip(
                value.queries, value.query_metadata, strict=True
            )
        ],
        "documents": [
            {"id": identifier, "text": text, **metadata}
            for (identifier, text), metadata in zip(
                value.documents, value.document_metadata, strict=True
            )
        ],
        "relevant_documents": {
            str(query_id): sorted(documents)
            for query_id, documents in value.relevant_documents.items()
        },
    }


def _load_evaluation(path: Path) -> EvaluationData:
    value = json.loads(path.read_text(encoding="utf-8"))
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
        query_metadata=tuple(
            {"source_id": str(row["source_id"]), "language": str(row["language"])}
            for row in value["queries"]
        ),
        document_metadata=tuple(
            {"source_id": str(row["source_id"]), "language": str(row["language"])}
            for row in value["documents"]
        ),
    )


def _prepare_nq(directory: Path, cache_directory: Path) -> dict[str, Any]:
    training_id, training_revision = CONTRACTS["nq"].training_datasets[0]
    evaluation_id, evaluation_revision = CONTRACTS["nq"].evaluation_datasets[0]
    training_path = _download(
        training_id,
        training_revision,
        "pair/train-00000-of-00001.parquet",
        cache_directory=cache_directory,
    )
    query_path = _download(
        evaluation_id,
        evaluation_revision,
        "queries.jsonl",
        cache_directory=cache_directory,
    )
    qrels_path = _download(
        evaluation_id,
        evaluation_revision,
        "qrels/test.tsv",
        cache_directory=cache_directory,
    )

    import pyarrow.parquet as parquet

    table = parquet.ParquetFile(training_path).read_row_group(0).slice(0, TRAINING_ROWS)
    training = tuple(
        {"query": str(row["query"]), "positive": str(row["answer"])}
        for row in table.to_pylist()
    )
    if len(training) != TRAINING_ROWS:
        raise ValueError("pinned NQ training shard is unexpectedly short")

    query_rows = _jsonl_prefix(query_path.read_bytes())
    queries = {str(row["_id"]): str(row["text"]) for row in query_rows}
    corpus_url = _dataset_url(evaluation_id, evaluation_revision, "corpus.jsonl")
    corpus_prefix, corpus_headers = _bounded_response(
        corpus_url, NQ_CORPUS_PREFIX_BYTES
    )
    corpus = {
        str(row["_id"]): "\n".join(
            part for part in (str(row.get("title", "")), str(row["text"])) if part
        )
        for row in _jsonl_prefix(corpus_prefix)
    }
    evaluation = select_nq_evaluation(queries, corpus, _qrels_tsv(qrels_path))

    _write_jsonl(directory / "train.jsonl", training)
    _write_json(directory / "evaluation.json", _serialize_evaluation(evaluation))
    return {
        "training_rows": len(training),
        "evaluation_queries": len(evaluation.queries),
        "evaluation_documents": len(evaluation.documents),
        "sources": {
            "training": {
                "repo_id": training_id,
                "revision": training_revision,
                "filename": "pair/train-00000-of-00001.parquet",
                "sha256": _sha256(training_path),
                "downloaded_bytes": training_path.stat().st_size,
            },
            "queries": {
                "repo_id": evaluation_id,
                "revision": evaluation_revision,
                "filename": "queries.jsonl",
                "sha256": _sha256(query_path),
                "downloaded_bytes": query_path.stat().st_size,
            },
            "qrels": {
                "repo_id": evaluation_id,
                "revision": evaluation_revision,
                "filename": "qrels/test.tsv",
                "sha256": _sha256(qrels_path),
                "downloaded_bytes": qrels_path.stat().st_size,
            },
            "corpus_prefix": {
                "repo_id": evaluation_id,
                "revision": evaluation_revision,
                "filename": "corpus.jsonl",
                "sha256": _sha256_bytes(corpus_prefix),
                "downloaded_bytes": len(corpus_prefix),
                "content_range": corpus_headers.get("content-range"),
                "parsed_documents": len(corpus),
            },
        },
    }


def _miracl_paths(language: str, split: str) -> tuple[str, str]:
    prefix = f"miracl-v1.0-{language}"
    return (
        f"{prefix}/topics/topics.{prefix}-{split}.tsv",
        f"{prefix}/qrels/qrels.{prefix}-{split}.tsv",
    )


def _stream_miracl_documents(
    language: str,
    *,
    qrels: Mapping[str, Sequence[tuple[str, str]]],
    required_queries: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, Any]]:
    repo_id, revision = CONTRACTS["miracl"].training_datasets[1]
    filename = f"miracl-corpus-v1.0-{language}/docs-0.jsonl.gz"
    url = _dataset_url(repo_id, revision, filename)
    request = Request(
        url,
        headers={"Range": f"bytes=0-{MIRACL_CORPUS_PREFIX_BYTES - 1}"},
    )
    document_queries: dict[str, list[tuple[str, str]]] = {}
    for split, rows in qrels.items():
        for query_id, document_id in rows:
            document_queries.setdefault(document_id, []).append((split, query_id))
    selected = {}
    found_queries = {split: set() for split in required_queries}
    inspected = hashlib.sha256()
    count = 0
    headers: dict[str, str] = {}
    with urlopen(request, timeout=120) as response:
        headers = {name.lower(): value for name, value in response.headers.items()}
        with gzip.GzipFile(fileobj=response) as stream:
            for line in stream:
                inspected.update(line)
                count += 1
                row = json.loads(line)
                document_id = str(row["docid"])
                if document_id in document_queries:
                    selected[document_id] = "\n".join(
                        part
                        for part in (str(row.get("title", "")), str(row["text"]))
                        if part
                    )
                    for split, query_id in document_queries[document_id]:
                        if split in found_queries:
                            found_queries[split].add(query_id)
                    if all(
                        len(found_queries[split]) >= required
                        for split, required in required_queries.items()
                    ):
                        break
                if count >= MIRACL_MAX_DOCUMENTS:
                    break
    missing = {
        split: required - len(found_queries[split])
        for split, required in required_queries.items()
        if len(found_queries[split]) < required
    }
    if missing:
        raise ValueError(
            f"bounded MIRACL {language} corpus prefix missed query coverage: "
            f"{missing}"
        )
    return selected, {
        "repo_id": repo_id,
        "revision": revision,
        "filename": filename,
        "content_range": headers.get("content-range"),
        "inspected_jsonl_sha256": "sha256:" + inspected.hexdigest(),
        "inspected_documents": count,
    }


def _first_positive_rows(
    topics: Mapping[str, str],
    qrels: Sequence[tuple[str, str]],
    *,
    count: int,
    available_documents: set[str] | None = None,
) -> tuple[tuple[str, str, str], ...]:
    output = []
    seen_queries = set()
    for query_id, document_id in qrels:
        if (
            query_id in seen_queries
            or query_id not in topics
            or (
                available_documents is not None
                and document_id not in available_documents
            )
        ):
            continue
        output.append((query_id, topics[query_id], document_id))
        seen_queries.add(query_id)
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(f"qrels yielded {len(output)} rows; expected {count}")
    return tuple(output)


def _prepare_miracl(directory: Path, cache_directory: Path) -> dict[str, Any]:
    topics_id, topics_revision = CONTRACTS["miracl"].training_datasets[0]
    training = []
    evaluation_rows = []
    sources: dict[str, Any] = {}
    for language in LANGUAGES:
        split_values = {}
        for split in ("train", "dev"):
            topic_filename, qrels_filename = _miracl_paths(language, split)
            topic_path = _download(
                topics_id,
                topics_revision,
                topic_filename,
                cache_directory=cache_directory,
            )
            qrels_path = _download(
                topics_id,
                topics_revision,
                qrels_filename,
                cache_directory=cache_directory,
            )
            split_values[split] = (
                _topics_tsv(topic_path),
                _qrels_tsv(qrels_path),
            )
            sources[f"{language}_{split}_topics"] = {
                "repo_id": topics_id,
                "revision": topics_revision,
                "filename": topic_filename,
                "sha256": _sha256(topic_path),
                "downloaded_bytes": topic_path.stat().st_size,
            }
            sources[f"{language}_{split}_qrels"] = {
                "repo_id": topics_id,
                "revision": topics_revision,
                "filename": qrels_filename,
                "sha256": _sha256(qrels_path),
                "downloaded_bytes": qrels_path.stat().st_size,
            }
        documents, corpus_source = _stream_miracl_documents(
            language,
            qrels={
                split: values[1] for split, values in split_values.items()
            },
            required_queries={"train": 2, "dev": 1},
        )
        train_rows = _first_positive_rows(
            *split_values["train"],
            count=2,
            available_documents=set(documents),
        )
        dev_rows = _first_positive_rows(
            *split_values["dev"],
            count=1,
            available_documents=set(documents),
        )
        sources[f"{language}_corpus_prefix"] = corpus_source
        training.extend(
            {
                "query": query,
                "positive": documents[document_id],
                "language": language,
                "source_query_id": query_id,
                "source_document_id": document_id,
            }
            for query_id, query, document_id in train_rows
        )
        query_id, query, document_id = dev_rows[0]
        evaluation_rows.append(
            (query_id, query, document_id, documents[document_id])
        )
    if len(training) != TRAINING_ROWS:
        raise ValueError("bounded MIRACL preparation produced the wrong row count")
    evaluation = _evaluation_data(evaluation_rows, languages=LANGUAGES)
    _write_jsonl(directory / "train.jsonl", training)
    _write_json(directory / "evaluation.json", _serialize_evaluation(evaluation))
    return {
        "training_rows": len(training),
        "evaluation_queries": len(evaluation.queries),
        "evaluation_documents": len(evaluation.documents),
        "languages": list(LANGUAGES),
        "sources": sources,
    }


def prepare_data(row: RowName, artifact_root: Path) -> Path:
    """Materialize only the real rows needed by one acceptance lifecycle."""

    contract = frozen_contract(row)
    serialized_contract = json.loads(json.dumps(asdict(contract)))
    directory = artifact_root.expanduser().resolve() / row / "data"
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != serialized_contract:
            raise ValueError(f"prepared {row} data contract differs: {manifest_path}")
        _load_evaluation(directory / "evaluation.json")
        return directory
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"data directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    cache_directory = artifact_root.expanduser().resolve() / "hf-cache"
    prepared = (
        _prepare_nq(directory, cache_directory)
        if row == "nq"
        else _prepare_miracl(directory, cache_directory)
    )
    _write_json(
        manifest_path,
        {
            "schema_version": "representax-nq-miracl-real-data-v1",
            "contract": serialized_contract,
            "bounds": {
                "training_rows": TRAINING_ROWS,
                "evaluation_queries": EVALUATION_QUERIES,
                "nq_corpus_prefix_bytes": NQ_CORPUS_PREFIX_BYTES,
                "miracl_corpus_prefix_bytes_per_language": (
                    MIRACL_CORPUS_PREFIX_BYTES
                ),
                "miracl_max_documents_per_language": MIRACL_MAX_DOCUMENTS,
            },
            **prepared,
        },
    )
    return directory


def resolve_checkpoint(
    row: RowName,
    artifact_root: Path,
    checkpoint: Path | None,
) -> Path:
    """Resolve exactly the paper-pinned model snapshot."""

    contract = frozen_contract(row)
    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {resolved}")
        if resolved.name != contract.model_revision:
            raise ValueError(
                f"checkpoint directory must be pinned snapshot "
                f"{contract.model_revision}: {resolved}"
            )
        return resolved
    from huggingface_hub import snapshot_download

    allow_patterns = (
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "modules.json",
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "1_Pooling/config.json",
    )
    resolved = Path(
        snapshot_download(
            repo_id=contract.model_id,
            revision=contract.model_revision,
            cache_dir=artifact_root.expanduser().resolve() / "hf-cache",
            allow_patterns=list(allow_patterns),
        )
    ).resolve()
    if resolved.name != contract.model_revision:
        raise ValueError(f"Hub resolved an unexpected model revision: {resolved}")
    return resolved


def load_miracl_text_encoder(
    model_name_or_path: str | Path,
    *,
    revision: str,
    local_files_only: bool,
    parameter_dtype: str,
    compute_dtype: str,
    sequence_length_buckets: Sequence[int],
    rematerialization: str = "none",
) -> tuple[Any, Any]:
    """Load the native Jina text tower and its route-aware tokenizer."""

    from transformers import AutoTokenizer

    from representax.integrations import load_jina_v5_small_text_encoder
    from representax.models.processing import make_text_processor

    model = load_jina_v5_small_text_encoder(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        rematerialization=rematerialization,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=True,
    )
    processor = make_text_processor(
        tokenizer=tokenizer,
        batch_builder=type(model).make_batch,
        sequence_length_buckets=sequence_length_buckets,
        prompts={"query": "Query: ", "document": "Document: "},
    )
    return model, processor


def acceptance_job(
    row: RowName,
    *,
    checkpoint: Path,
    data_directory: Path,
) -> Any:
    """Build the two-update canonical lifecycle used by both paper rows."""

    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        ExportConfig,
        InformationRetrievalEvaluatorConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig

    contract = frozen_contract(row)
    evaluation = _load_evaluation(data_directory / "evaluation.json")
    model_parameters = {
        "model_name_or_path": str(checkpoint),
        "revision": contract.model_revision,
        "local_files_only": True,
        "parameter_dtype": "bfloat16" if row == "miracl" else "float32",
        "compute_dtype": "bfloat16",
        "sequence_length_buckets": (MAXIMUM_LENGTH,),
    }
    evaluation_source = f"paper-acceptance://{row}"
    return JobConfig(
        name=f"paper-{contract.workload}-real-data-acceptance",
        model=ModelConfig(
            target=contract.model_target,
            parameters=model_parameters,
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(scale=20.0, symmetric=False),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 2e-5, "weight_decay": 0.0},
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data_directory / "train.jsonl"), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval.RetrievalCollator"
            ),
            drop_remainder=True,
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=TRAINING_BATCH_SIZE,
            max_steps=STEPS,
            seed=7,
            batch=BatchConfig(micro_batch_size=TRAINING_BATCH_SIZE),
            activation_rematerialization="full" if row == "miracl" else "none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(
            every=1,
            keep=2,
            save_final=True,
            asynchronous=True,
        ),
        logging=LoggingConfig(console_every=1),
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
                    name=contract.workload,
                    relevant_documents=dict(evaluation.relevant_documents),
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
            primary_metric=f"valid/{contract.workload}/cosine_ndcg@10",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(selection="final"),
    )


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


def _canonical_export_evaluation(
    row: RowName,
    model: Any,
    processor: Any,
    data: EvaluationData,
) -> dict[str, float]:
    from representax.evaluation import InformationRetrievalEvaluator
    from representax.precision import resolve_precision_policy
    from representax.train.evaluation import EvaluationRunner

    contract = frozen_contract(row)
    evaluator = InformationRetrievalEvaluator(
        name=contract.workload,
        relevant_documents=data.relevant_documents,
        score_functions=("cosine",),
        main_score_function="cosine",
        accuracy_at_k=(1, 4),
        precision_recall_at_k=(1, 4, 10, 100),
        mrr_at_k=(4,),
        ndcg_at_k=(4, 10),
        map_at_k=(4,),
    )
    from representax.config import PrecisionConfig

    result = EvaluationRunner(
        evaluator,
        precision=resolve_precision_policy(PrecisionConfig.bfloat16_mixed()),
    ).run(model, _evaluation_batches(processor, data), iteration=STEPS)
    return {name: float(value) for name, value in result.metrics.items()}


def _metric_rows(path: Path, event: str) -> list[dict[str, Any]]:
    output = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == event:
                output.append(row)
    return output


def run_acceptance(
    row: RowName,
    *,
    artifact_root: Path,
    checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Execute load, update, evaluation, resume, export, and export evaluation."""

    import jax

    from experiments.paper.dense_retrieval import fixed_rows_resolver
    from representax import load_inference_bundle
    from representax.train import run_job
    from representax.train.job import load_model

    root = artifact_root.expanduser().resolve()
    data_directory = prepare_data(row, root)
    checkpoint = resolve_checkpoint(row, root, checkpoint)
    run_directory = root / row / "run"
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"run directory must be empty: {run_directory}")
    evaluation = _load_evaluation(data_directory / "evaluation.json")
    job = acceptance_job(
        row,
        checkpoint=checkpoint,
        data_directory=data_directory,
    )
    processor_model, processor = load_model(
        job.model,
        key=jax.random.key(job.training.seed),
        activation_rematerialization=job.training.activation_rematerialization,
    )
    if processor is None:
        raise RuntimeError("acceptance model did not load its pinned processor")
    del processor_model
    gc.collect()
    jax.clear_caches()

    resolvers = {
        "paper-acceptance": fixed_rows_resolver(_evaluation_records(evaluation))
    }
    paused = run_job(job, run_directory, resolvers=resolvers, stop_after=1)
    if paused.completed_iterations != 1:
        raise RuntimeError("run_job did not execute the midpoint update")
    del paused
    gc.collect()
    jax.clear_caches()
    completed = run_job(job, run_directory, resolvers=resolvers, resume=True)
    jax.block_until_ready(completed.state)
    if not completed.resumed or completed.completed_iterations != STEPS:
        raise RuntimeError("run_job did not resume through the final update")
    if completed.inference_bundle is None:
        raise RuntimeError("run_job did not export an inference bundle")
    inference_bundle = completed.inference_bundle
    resumed = completed.resumed
    completed_iterations = completed.completed_iterations
    del completed
    gc.collect()
    jax.clear_caches()

    reloaded_model, reloaded_job = load_inference_bundle(inference_bundle)
    exported_metrics = _canonical_export_evaluation(
        row,
        reloaded_model,
        processor,
        evaluation,
    )
    metric_rows = _metric_rows(run_directory / "metrics.jsonl", "evaluation")
    training_rows = _metric_rows(run_directory / "metrics.jsonl", "training_step")
    if len(metric_rows) < 2:
        raise RuntimeError("canonical evaluation did not run on start and end")
    if len(training_rows) != STEPS:
        raise RuntimeError("canonical metric stream has the wrong update count")
    final_metrics = {
        name: float(value)
        for name, value in metric_rows[-1]["metrics"].items()
        if name.startswith("valid/")
    }
    if exported_metrics.keys() != final_metrics.keys() or any(
        abs(exported_metrics[name] - final_metrics[name]) > 1e-7
        for name in final_metrics
    ):
        raise RuntimeError("exported-artifact canonical evaluation changed metrics")
    losses = [float(value["metrics"]["train/loss"]) for value in training_rows]
    if not all(value == value and abs(value) != float("inf") for value in losses):
        raise RuntimeError("training emitted a non-finite loss")
    checkpoint_path = run_directory / "checkpoints" / "1"
    if not checkpoint_path.is_dir():
        raise RuntimeError("midpoint checkpoint is missing")
    result = {
        "schema_version": "representax-nq-miracl-acceptance-v1",
        "row": row,
        "workload": frozen_contract(row).workload,
        "contract": asdict(frozen_contract(row)),
        "data_manifest": str(data_directory / "manifest.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": 1,
        "resumed": resumed,
        "completed_iterations": completed_iterations,
        "training_losses": losses,
        "canonical_evaluation_events": len(metric_rows),
        "final_evaluation": final_metrics,
        "exported_artifact_evaluation": exported_metrics,
        "inference_bundle": str(inference_bundle),
        "reloaded_job_name": reloaded_job.name,
        "evaluation_queries": len(evaluation.queries),
        "evaluation_documents": len(evaluation.documents),
        "languages": sorted(
            {metadata["language"] for metadata in evaluation.query_metadata}
        ),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(root / row / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("row", choices=tuple(CONTRACTS))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    data_directory = prepare_data(arguments.row, arguments.artifact_root)
    if arguments.prepare_only:
        print(data_directory)
        return
    result = run_acceptance(
        arguments.row,
        artifact_root=arguments.artifact_root,
        checkpoint=arguments.checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
