"""Matched CLIP image-text retrieval preflight for the paper campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.paper.provenance import reference_source, write_reference_result

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
MULTIMODAL_MANIFEST = ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json"
FRAMEWORKS = ("representax", "sentence-transformers")
TRAINING_IMAGES = 512
TRAINING_REPEATS = 4
GRAD_CACHE_MICRO_BATCH = 8
EVALUATION_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    train_dataset: Mapping[str, Any]
    evaluation_dataset: Mapping[str, Any]
    reference_version: str
    global_batch_size: int
    image_shape: tuple[int, int, int]


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the one image-text row frozen across the paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(MULTIMODAL_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "image-text-retrieval"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "image-text-retrieval"
    )
    if campaign_row["frameworks"] != ["representax", "sentence-transformers"]:
        raise ValueError("the frozen image-text frameworks changed")
    if panel_row["reference"] != "sentence-transformers":
        raise ValueError("the frozen image-text reference changed")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the image-text reference requires a release")
    image_shape = campaign_row["image_shape"]
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        train_dataset=panel["datasets"][panel_row["train"]],
        evaluation_dataset=panel["datasets"][panel_row["evaluate"]],
        reference_version=reference.release,
        global_batch_size=int(campaign_row["global_batch"]),
        image_shape=(
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _tree_sha256(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(_sha256(path).encode())
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


def _download_image(url: str, destination: Path) -> None:
    from PIL import Image

    request = urllib.request.Request(url, headers={"User-Agent": "representax/1"})
    last_error: BaseException | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                image.verify()
            destination.write_bytes(payload)
            return
        except BaseException as error:
            last_error = error
            if attempt != 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def _prepare_coco(directory: Path, count: int) -> tuple[Path, ...]:
    import datasets

    contract = frozen_contract()
    source = datasets.load_dataset(
        contract.train_dataset["repo_id"],
        revision=contract.train_dataset["revision"],
        split=contract.train_dataset["split"],
        streaming=True,
    )
    rows = tuple(source.take(count))
    if len(rows) != count:
        raise ValueError(f"COCO produced {len(rows)} rows; expected {count}")
    image_directory = directory / "coco-images"
    image_directory.mkdir()

    def materialize(item: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
        index, row = item
        captions = row["captions"]
        if not captions:
            raise ValueError(f"COCO row {index} has no captions")
        relative = Path("coco-images") / f"{int(row['image_id']):012d}.jpg"
        url = str(row["coco_url"])
        _download_image(url, directory / relative)
        return {
            "source_index": index,
            "image_id": int(row["image_id"]),
            "caption": str(captions[0]).strip(),
            "image": str(relative),
            "source_url": url,
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        records = tuple(executor.map(materialize, enumerate(rows)))
    presentations = tuple(
        {**record, "presentation_cycle": cycle}
        for cycle in range(TRAINING_REPEATS)
        for record in records
    )
    _write_jsonl(directory / "train.jsonl", presentations)
    return tuple(directory / str(record["image"]) for record in records)


def _prepare_flickr(directory: Path) -> tuple[tuple[Path, ...], dict[int, set[int]]]:
    import datasets

    contract = frozen_contract()
    dataset = contract.evaluation_dataset
    common = {
        "path": dataset["repo_id"],
        "revision": dataset["revision"],
    }
    queries = datasets.load_dataset(
        **common,
        name="query",
        split=dataset["split"],
    )
    corpus = datasets.load_dataset(**common, name="corpus", split="corpus")
    qrels = datasets.load_dataset(
        **common,
        name="qrels",
        split=dataset["split"],
    )
    image_directory = directory / "flickr-images"
    image_directory.mkdir()
    document_ids = {str(row["id"]): index for index, row in enumerate(corpus)}
    query_ids = {
        str(row["id"]): len(document_ids) + index for index, row in enumerate(queries)
    }
    image_paths: list[Path] = []
    documents: list[dict[str, Any]] = []
    for index, row in enumerate(corpus):
        relative = Path("flickr-images") / f"{index:04d}.jpg"
        image = row["image"].convert("RGB")
        image.save(directory / relative, format="JPEG", quality=95)
        image.close()
        image_paths.append(directory / relative)
        documents.append(
            {
                "kind": "document",
                "identifier": document_ids[str(row["id"])],
                "image": str(relative),
                "text": "",
                "valid": True,
            }
        )

    relevant: dict[int, set[int]] = {}
    for row in qrels:
        if float(row["score"]) <= 0:
            continue
        relevant.setdefault(query_ids[str(row["query-id"])], set()).add(
            document_ids[str(row["corpus-id"])]
        )
    query_records = [
        {
            "kind": "query",
            "identifier": query_ids[str(row["id"])],
            "text": str(row["text"]),
            "image": "",
            "valid": True,
        }
        for row in queries
    ]
    if set(relevant) != {int(row["identifier"]) for row in query_records}:
        raise ValueError("Flickr30k qrels do not cover every query")

    def pad(records: list[dict[str, Any]], *, kind: str) -> None:
        remainder = len(records) % EVALUATION_BATCH_SIZE
        for _ in range((-remainder) % EVALUATION_BATCH_SIZE):
            records.append(
                {
                    "kind": kind,
                    "identifier": -1,
                    "text": "",
                    "image": str(Path("flickr-images") / "0000.jpg"),
                    "valid": False,
                }
            )

    pad(query_records, kind="query")
    pad(documents, kind="document")
    _write_jsonl(directory / "evaluation.jsonl", (*query_records, *documents))
    return tuple(image_paths), relevant


def prepare_data(
    output: Path,
    *,
    training_images: int = TRAINING_IMAGES,
) -> dict[str, Any]:
    """Materialize one deterministic preflight subset and the full held-out panel."""

    if training_images <= 0:
        raise ValueError("training_images must be positive")
    output.mkdir(parents=True, exist_ok=False)
    coco_images = _prepare_coco(output, training_images)
    flickr_images, relevant = _prepare_flickr(output)
    files = {
        name: {"rows": sum(1 for _ in path.open()), "sha256": _sha256(path)}
        for name, path in (
            ("train.jsonl", output / "train.jsonl"),
            ("evaluation.jsonl", output / "evaluation.jsonl"),
        )
    }
    manifest = {
        "schema_version": "representax-image-text-preflight-data-v1",
        "contract": asdict(frozen_contract()),
        "training_images": training_images,
        "training_repeats": TRAINING_REPEATS,
        "training_presentations": training_images * TRAINING_REPEATS,
        "evaluation_queries": len(relevant),
        "evaluation_documents": len(flickr_images),
        "relevant_documents": {
            str(query): sorted(documents) for query, documents in relevant.items()
        },
        "files": files,
        "coco_image_tree_sha256": _tree_sha256(coco_images, output),
        "flickr_image_tree_sha256": _tree_sha256(flickr_images, output),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _open_image(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


class ImageTextRetrievalCollator:
    """Build aligned caption/image MNR batches with the model processor."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-image-text-collator-v1",
            "processor": self.processor.data_contract(),
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        from representax.core import Route
        from representax.tasks.retrieval import retrieval_batch

        captions = tuple(str(row["caption"]) for row in rows)
        images = tuple(
            _open_image(self.root_directory / str(row["image"])) for row in rows
        )
        size = len(rows)
        return retrieval_batch(
            query=self.processor(captions, route=Route.QUERY),
            document=self.processor(images, route=Route.DOCUMENT),
            positive_mask=np.eye(size, dtype=np.bool_),
        )


class ImageTextEvaluationCollator:
    """Build homogeneous caption-query or image-document evaluation batches."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-image-text-evaluation-collator-v1",
            "processor": self.processor.data_contract(),
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.core import Route
        from representax.evaluation import retrieval_evaluation_batch

        kinds = {str(row["kind"]) for row in rows}
        if len(kinds) != 1:
            raise ValueError("image-text evaluation batches must be homogeneous")
        kind = kinds.pop()
        if kind == "query":
            inputs = self.processor(
                tuple(str(row["text"]) for row in rows),
                route=Route.QUERY,
            )
        elif kind == "document":
            inputs = self.processor(
                tuple(
                    _open_image(self.root_directory / str(row["image"])) for row in rows
                ),
                route=Route.DOCUMENT,
            )
        else:
            raise ValueError(f"unknown image-text evaluation kind {kind!r}")
        return retrieval_evaluation_batch(
            inputs,
            jnp.asarray(tuple(int(row["identifier"]) for row in rows), dtype=jnp.int32),
            kind=kind,
            valid=jnp.asarray(
                tuple(bool(row["valid"]) for row in rows),
                dtype=jnp.bool_,
            ),
        )


def _representax_job(
    *, checkpoint: Path, data_directory: Path, steps: int, seed: int
) -> Any:
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EvaluationConfig,
        ExportConfig,
        GradCacheConfig,
        HuggingFaceExportConfig,
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

    contract = frozen_contract()
    manifest = _document(data_directory / "manifest.json")
    if int(manifest["training_presentations"]) < contract.global_batch_size * steps:
        raise ValueError(
            "preflight data does not contain enough training presentations"
        )
    relevant = {
        int(query): frozenset(int(document) for document in documents)
        for query, documents in manifest["relevant_documents"].items()
    }

    def data(path: Path, collator: str) -> DataConfig:
        return DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=ComponentConfig(
                target=collator,
                parameters={"root_directory": str(data_directory)},
            ),
            drop_remainder=True,
            num_threads=0,
            prefetch_buffer_size=1,
        )

    source_checkpoint = checkpoint / "0_CLIPModel"
    if not (source_checkpoint / "config.json").is_file():
        source_checkpoint = checkpoint
    return JobConfig(
        name="paper-preflight-image-text-retrieval",
        model=ModelConfig(
            target="representax.models.clip:load_clip",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
            },
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
                    "warmup_steps": 1,
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=data(
            data_directory / "train.jsonl",
            "experiments.paper.image_text:ImageTextRetrievalCollator",
        ),
        training=TrainingConfig(
            global_batch_size=contract.global_batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(micro_batch_size=contract.global_batch_size),
            grad_cache=GradCacheConfig(micro_batch_size=GRAD_CACHE_MICRO_BATCH),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(every=steps // 2, keep=2, save_final=True),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        evaluation=EvaluationConfig(
            data=data(
                data_directory / "evaluation.jsonl",
                "experiments.paper.image_text:ImageTextEvaluationCollator",
            ),
            batch_size=EVALUATION_BATCH_SIZE,
            evaluators=(
                InformationRetrievalEvaluatorConfig(
                    name="flickr30k",
                    relevant_documents=relevant,
                    score_functions=("cosine",),
                    main_score_function="cosine",
                    accuracy_at_k=(1, 5, 10),
                    precision_recall_at_k=(1, 5, 10),
                    mrr_at_k=(10,),
                    ndcg_at_k=(10,),
                    map_at_k=(10,),
                ),
            ),
            on_start=True,
            on_end=True,
            primary_metric="valid/flickr30k/cosine_recall@10",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(
            selection="final",
            huggingface=HuggingFaceExportConfig(
                source_checkpoint=str(source_checkpoint),
                adapter=ComponentConfig(
                    target="representax.models.clip:CLIPCheckpointAdapter",
                    parameters={"rematerialization": "none"},
                ),
                verify_reload=True,
            ),
        ),
    )


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return _read_jsonl(path)


def _steady_state(rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, Any]:
    durations = []
    for row in rows:
        if row.get("event") != "training_step":
            continue
        metrics = row["metrics"]
        if "perf/compilation_and_first_step_seconds" in metrics:
            continue
        duration = metrics.get("perf/step_seconds")
        if duration is not None and float(duration) > 0:
            durations.append(float(duration))
    if not durations:
        return {"measured_steps": 0}
    return {
        "measured_steps": len(durations),
        "median_step_seconds": statistics.median(durations),
        "examples_per_second": batch_size * len(durations) / sum(durations),
    }


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
    from representax.config import PrecisionConfig
    from representax.core import Route, encode
    from representax.models.clip import CLIPEncoder
    from representax.models.clip.checkpoint import clip_checkpoint_directory
    from representax.models.clip.processing import make_clip_processor
    from representax.precision import precision_context, resolve_precision_policy
    from representax.train import run_job

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("image-text preflight requires exactly one visible GPU")
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
    )
    started = time.perf_counter()
    paused = run_job(job, run_directory, stop_after=steps // 2)
    if paused.completed_iterations != steps // 2:
        raise RuntimeError("Representax did not stop at the midpoint checkpoint")
    del paused
    gc.collect()
    jax.clear_caches()
    completed = run_job(job, run_directory, resume=True)
    jax.block_until_ready(completed.state)
    if not completed.resumed or completed.completed_iterations != steps:
        raise RuntimeError("Representax did not resume to the final update")
    if completed.inference_bundle is None:
        raise RuntimeError("Representax did not produce an inference bundle")

    trained_model = completed.state.model
    if not isinstance(trained_model, CLIPEncoder):
        raise TypeError("image-text training returned a different model family")
    source = clip_checkpoint_directory(checkpoint)
    processor = make_clip_processor(
        source,
        trained_model.config,
        normalize_output=trained_model.normalize_output,
    )
    probe_rows = _read_jsonl(data_directory / "train.jsonl")[:4]
    collator = ImageTextRetrievalCollator(
        processor=processor,
        root_directory=data_directory,
    )
    probe = collator(probe_rows)
    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    with precision_context(precision):
        expected = (
            encode(trained_model, probe.query, route=Route.QUERY),
            encode(trained_model, probe.document, route=Route.DOCUMENT),
        )
    jax.block_until_ready(expected)
    restored, restored_job = load_inference_bundle(completed.inference_bundle)
    if restored_job.name != job.name:
        raise RuntimeError("native reload reconstructed a different job")
    if not isinstance(restored, CLIPEncoder):
        raise TypeError("native reload returned a different model family")
    with precision_context(precision):
        actual = (
            encode(restored, probe.query, route=Route.QUERY),
            encode(restored, probe.document, route=Route.DOCUMENT),
        )
    jax.block_until_ready(actual)
    reload_difference = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in zip(expected, actual, strict=True)
    )
    if reload_difference != 0.0:
        raise RuntimeError("native inference reload changed CLIP embeddings")

    rows = _metric_rows(run_directory / "metrics.jsonl")
    updates = [row for row in rows if row.get("event") == "training_step"]
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    if len(updates) != steps or len(evaluations) != 2:
        raise RuntimeError("Representax evidence is missing updates or evaluations")
    update_norms = [
        float(row["metrics"]["train/update_global_norm"]) for row in updates
    ]
    if not all(np.isfinite(update_norms)) or not any(
        value > 0 for value in update_norms
    ):
        raise RuntimeError("Representax produced no finite nonzero update")
    return {
        "schema_version": "representax-image-text-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": frozen_contract().global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "elapsed_seconds": time.perf_counter() - started,
        "steady_state": _steady_state(rows, frozen_contract().global_batch_size),
        "initial_evaluation": {
            name: float(value)
            for name, value in evaluations[0]["metrics"].items()
            if name.startswith("valid/")
        },
        "final_evaluation": {
            name: float(value)
            for name, value in evaluations[-1]["metrics"].items()
            if name.startswith("valid/")
        },
        "final_loss": float(updates[-1]["metrics"]["train/loss"]),
        "final_update_global_norm": float(
            updates[-1]["metrics"]["train/update_global_norm"]
        ),
        "resumed": completed.resumed,
        "checkpoint": str(run_directory / "checkpoints" / str(steps // 2)),
        "inference_bundle": str(completed.inference_bundle),
        "huggingface_export": str(completed.inference_bundle / "huggingface"),
        "native_reload_maximum_absolute_difference": reload_difference,
        "device": jax.devices()[0].device_kind,
    }


def _reference_evaluation(
    model: Any, data_directory: Path, *, batch_size: int
) -> dict[str, float]:
    from representax.evaluation.retrieval import information_retrieval_metrics

    records = _read_jsonl(data_directory / "evaluation.jsonl")
    queries = [row for row in records if row["kind"] == "query" and row["valid"]]
    documents = [row for row in records if row["kind"] == "document" and row["valid"]]
    query_embeddings = model.encode(
        [row["text"] for row in queries],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    document_embeddings = []
    for start in range(0, len(documents), batch_size):
        images = [
            _open_image(data_directory / row["image"])
            for row in documents[start : start + batch_size]
        ]
        document_embeddings.append(
            model.encode(
                images,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        )
        for image in images:
            image.close()
    document_embeddings = np.concatenate(document_embeddings)
    query_embeddings = query_embeddings / np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    document_embeddings = document_embeddings / np.maximum(
        np.linalg.norm(document_embeddings, axis=1, keepdims=True), 1e-12
    )
    scores = query_embeddings @ document_embeddings.T
    top = np.argsort(-scores, axis=1)[:, :10]
    document_ids = np.asarray([int(row["identifier"]) for row in documents])
    ranked = document_ids[top]
    query_ids = np.asarray([int(row["identifier"]) for row in queries])
    manifest = _document(data_directory / "manifest.json")
    relevant = {
        int(query): frozenset(int(document) for document in values)
        for query, values in manifest["relevant_documents"].items()
    }
    metrics = information_retrieval_metrics(
        ranked,
        query_ids,
        relevant,
        accuracy_at_k=(1, 5, 10),
        precision_recall_at_k=(1, 5, 10),
        mrr_at_k=(10,),
        ndcg_at_k=(10,),
        map_at_k=(10,),
    )
    return {f"valid/flickr30k/cosine_{name}": value for name, value in metrics.items()}


def _sentence_transformers_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import datasets
    import sentence_transformers
    import torch
    import transformers
    from benchmarks.samplers import sequential_sentence_transformers_batches
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )

    contract = frozen_contract()
    if sentence_transformers.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected sentence-transformers=={contract.reference_version}, "
            f"found {sentence_transformers.__version__}"
        )
    model = SentenceTransformer(str(checkpoint), local_files_only=True)
    initial_started = time.perf_counter()
    initial_evaluation = _reference_evaluation(
        model, data_directory, batch_size=EVALUATION_BATCH_SIZE
    )
    initial_evaluation_seconds = time.perf_counter() - initial_started
    rows = _read_jsonl(data_directory / "train.jsonl")
    train_dataset = datasets.Dataset.from_dict(
        {
            "caption": [row["caption"] for row in rows],
            "image": [str(data_directory / row["image"]) for row in rows],
        }
    ).cast_column("image", datasets.Image())
    loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=GRAD_CACHE_MICRO_BATCH,
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=contract.global_batch_size,
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
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=loss,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    final_started = time.perf_counter()
    final_evaluation = _reference_evaluation(
        model, data_directory, batch_size=EVALUATION_BATCH_SIZE
    )
    final_evaluation_seconds = time.perf_counter() - final_started
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    midpoint = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
    if not midpoint.is_dir():
        raise RuntimeError("Sentence Transformers did not save the midpoint")
    final_checkpoint = run_directory / "checkpoints" / f"checkpoint-{steps}"
    if not final_checkpoint.is_dir():
        raise RuntimeError("Sentence Transformers did not save the final checkpoint")
    midpoint_model = SentenceTransformer(str(midpoint), local_files_only=True)
    checkpoint_model = SentenceTransformer(str(final_checkpoint), local_files_only=True)
    reloaded = SentenceTransformer(str(export), local_files_only=True)
    probe_caption = str(rows[0]["caption"])
    expected = checkpoint_model.encode([probe_caption], convert_to_numpy=True)
    midpoint_probe = midpoint_model.encode([probe_caption], convert_to_numpy=True)
    actual = reloaded.encode([probe_caption], convert_to_numpy=True)
    reload_difference = float(np.max(np.abs(expected - actual)))
    if not np.array_equal(expected, actual) or not np.all(np.isfinite(midpoint_probe)):
        raise RuntimeError("Sentence Transformers checkpoint or export reload failed")
    losses = [
        float(row["loss"])
        for row in trainer.state.log_history
        if row.get("loss") is not None
    ]
    return {
        "schema_version": "representax-image-text-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "training_seconds": training_seconds,
        "examples_per_second": contract.global_batch_size * steps / training_seconds,
        "initial_evaluation_seconds": initial_evaluation_seconds,
        "final_evaluation_seconds": final_evaluation_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_loss": losses[-1],
        "training_metrics": output.metrics,
        "checkpoint": str(midpoint),
        "inference_bundle": str(export),
        "reload_maximum_absolute_difference": reload_difference,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device": torch.cuda.get_device_name(),
    }


def _worker(arguments: argparse.Namespace) -> None:
    function = (
        _representax_worker
        if arguments.framework == "representax"
        else _sentence_transformers_worker
    )
    report = function(
        checkpoint=arguments.checkpoint,
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        report = write_reference_result(
            arguments.report,
            report,
            reference="sentence-transformers",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    commands = {}
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        log = output / f"{framework}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.paper.image_text",
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
        commands[framework] = command
        with log.open("x", encoding="utf-8") as stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        reports[framework] = _document(report)
    summary = {
        "schema_version": "representax-image-text-preflight-v1",
        "contract": {
            **asdict(frozen_contract()),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "gpu": arguments.gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
        },
        "commands": commands,
        **reports,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(reports, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--training-images", type=int, default=TRAINING_IMAGES)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=4)
    worker.add_argument("--seed", type=int, default=7)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=4)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--gpu", type=int, default=1)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    training_images=arguments.training_images,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "worker":
        _worker(arguments)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()


__all__ = [
    "FrozenContract",
    "ImageTextEvaluationCollator",
    "ImageTextRetrievalCollator",
    "frozen_contract",
    "prepare_data",
]
