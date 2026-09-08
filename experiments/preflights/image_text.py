"""Matched CLIP image-text retrieval preflight for the paper campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
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

from experiments.preflights.accelerator import (
    Platform,
    data_parallel_job,
    initialize_jax,
    process_local_rows,
    torch_device_report,
    torch_rank,
    torch_reset_peak_memory,
    torch_synchronize,
    torch_world_size,
    use_fixed_text_padding,
)
from experiments.preflights.provenance import reference_source, write_reference_result
from experiments.preflights.timing import CudaStepTimer, warm_step_summary

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
MULTIMODAL_MANIFEST = ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json"
FRAMEWORKS = ("representax", "sentence-transformers")
TRAINING_IMAGES = 512
CAPTIONS_PER_IMAGE = 4
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


def _distinct_captions(
    values: Iterable[Any],
    *,
    count: int,
) -> tuple[str, ...]:
    selected = []
    seen = set()
    for value in values:
        caption = str(value).strip()
        if not caption or caption in seen:
            continue
        seen.add(caption)
        selected.append(caption)
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(
        f"expected {count} distinct nonempty captions, found {len(selected)}"
    )


def _select_coco_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    captions_per_image: int = CAPTIONS_PER_IMAGE,
) -> tuple[tuple[int, Mapping[str, Any], tuple[str, ...]], ...]:
    selected = []
    image_ids: set[int] = set()
    for source_index, row in enumerate(rows):
        image_id = int(row["image_id"])
        if image_id in image_ids:
            continue
        try:
            chosen = _distinct_captions(
                row["captions"],
                count=captions_per_image,
            )
        except ValueError:
            continue
        image_ids.add(image_id)
        selected.append((source_index, row, chosen))
        if len(selected) == count:
            return tuple(selected)
    raise ValueError(f"COCO contains only {len(selected)} unique usable rows")


def _batch_unique_caption_order(
    rows: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    if len(rows) % batch_size:
        raise ValueError("each COCO caption cycle must contain complete batches")
    batch_count = len(rows) // batch_size
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["caption"]), []).append(row)
    largest_group = max(map(len, grouped.values()))
    if largest_group > batch_count:
        raise ValueError(
            "a repeated COCO caption cannot be distributed across unique batches: "
            f"{largest_group} occurrences over {batch_count} batches"
        )

    random = np.random.default_rng(seed)
    tie_order = random.permutation(batch_count)
    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    available = [(0, int(tie_order[index]), index) for index in range(batch_count)]
    heapq.heapify(available)
    groups = sorted(
        grouped.values(),
        key=lambda group: (-len(group), int(random.integers(0, 2**31))),
    )
    for group in groups:
        group_order = random.permutation(len(group))
        selected_batches = [heapq.heappop(available) for _ in group]
        for (_, tie, index), row_index in zip(
            selected_batches, group_order, strict=True
        ):
            batches[index].append(group[int(row_index)])
            heapq.heappush(available, (len(batches[index]), tie, index))
    if any(len(batch) != batch_size for batch in batches):
        raise RuntimeError("caption allocation did not produce complete COCO batches")

    ordered = []
    for batch in batches:
        ordered.extend(batch[int(index)] for index in random.permutation(len(batch)))
    return tuple(ordered)


def _prepare_coco(
    directory: Path, count: int, *, captions_per_image: int
) -> tuple[Path, ...]:
    import datasets

    contract = frozen_contract()
    source = datasets.load_dataset(
        contract.train_dataset["repo_id"],
        revision=contract.train_dataset["revision"],
        split=contract.train_dataset["split"],
        streaming=True,
    )
    rows = _select_coco_rows(
        source,
        count=count,
        captions_per_image=captions_per_image,
    )
    image_directory = directory / "coco-images"
    image_directory.mkdir()

    def materialize(
        item: tuple[int, Mapping[str, Any], tuple[str, ...]],
    ) -> dict[str, Any]:
        index, row, captions = item
        relative = Path("coco-images") / f"{int(row['image_id']):012d}.jpg"
        url = str(row["coco_url"])
        _download_image(url, directory / relative)
        return {
            "source_index": index,
            "image_id": int(row["image_id"]),
            "captions": captions,
            "image": str(relative),
            "source_url": url,
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        records = tuple(executor.map(materialize, rows))
    presentations = []
    for caption_index in range(captions_per_image):
        cycle = [
            {
                **{name: value for name, value in record.items() if name != "captions"},
                "caption": record["captions"][caption_index],
                "caption_index": caption_index,
            }
            for record in records
        ]
        presentations.extend(
            _batch_unique_caption_order(
                cycle,
                batch_size=frozen_contract().global_batch_size,
                seed=caption_index,
            )
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
    captions_per_image: int = CAPTIONS_PER_IMAGE,
) -> dict[str, Any]:
    """Materialize one deterministic preflight subset and the full held-out panel."""

    if training_images <= 0:
        raise ValueError("training_images must be positive")
    if captions_per_image <= 0:
        raise ValueError("captions_per_image must be positive")
    output.mkdir(parents=True, exist_ok=False)
    coco_images = _prepare_coco(
        output,
        training_images,
        captions_per_image=captions_per_image,
    )
    flickr_images, relevant = _prepare_flickr(output)
    files = {
        name: {"rows": sum(1 for _ in path.open()), "sha256": _sha256(path)}
        for name, path in (
            ("train.jsonl", output / "train.jsonl"),
            ("evaluation.jsonl", output / "evaluation.jsonl"),
        )
    }
    manifest = {
        "contract": asdict(frozen_contract()),
        "training_images": training_images,
        "captions_per_image": captions_per_image,
        "training_presentations": training_images * captions_per_image,
        "unique_image_ids": True,
        "batch_unique_captions": True,
        "distinct_captions_per_image": True,
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


def _validate_training_manifest(manifest: Mapping[str, Any]) -> None:
    captions_are_safe = manifest.get("batch_unique_captions") or manifest.get(
        "unique_captions"
    )
    if not manifest.get("unique_image_ids") or not captions_are_safe:
        raise ValueError(
            "image-text training data predates duplicate-free preparation; rebuild it"
        )


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
        from representax.tasks.retrieval import (
            process_local_retrieval_batch,
            retrieval_batch,
        )

        local_rows, offset, global_size = process_local_rows(rows)
        captions = tuple(str(row["caption"]) for row in local_rows)
        images = tuple(
            _open_image(self.root_directory / str(row["image"])) for row in local_rows
        )
        query = self.processor(captions, route=Route.QUERY)
        document = self.processor(images, route=Route.DOCUMENT)
        if len(local_rows) == global_size:
            return retrieval_batch(
                query=query,
                document=document,
                positive_mask=np.eye(global_size, dtype=np.bool_),
            )
        positive_mask = np.zeros((len(local_rows), global_size), dtype=np.bool_)
        positive_mask[
            np.arange(len(local_rows)), offset + np.arange(len(local_rows))
        ] = True
        return process_local_retrieval_batch(
            query=query,
            document=document,
            positive_mask=positive_mask,
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
    *,
    checkpoint: Path,
    data_directory: Path,
    steps: int,
    seed: int,
    negative_scope: str = "global",
    symmetric: bool = False,
    warmup_steps: int = 1,
    training_file: str = "train.jsonl",
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
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError("warmup_steps must be non-negative and less than steps")
    manifest = _document(data_directory / "manifest.json")
    _validate_training_manifest(manifest)
    if int(manifest["training_presentations"]) < contract.global_batch_size:
        raise ValueError(
            "preflight data does not contain one complete contrastive batch"
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
            num_threads=8,
            prefetch_buffer_size=8,
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
        loss=MNRConfig(
            scale=20.0,
            symmetric=symmetric,
            negative_scope=negative_scope,
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
                    "peak_value": 2e-5,
                    "warmup_steps": warmup_steps,
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=data(
            data_directory / training_file,
            "experiments.preflights.image_text:ImageTextRetrievalCollator",
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
                "experiments.preflights.image_text:ImageTextEvaluationCollator",
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
    platform: Platform = "gpu",
    negative_scope: str = "global",
    symmetric: bool = False,
    warmup_steps: int = 1,
    training_file: str = "train.jsonl",
) -> dict[str, Any]:
    jax = initialize_jax(platform)

    from representax import load_inference_bundle
    from representax.config import PrecisionConfig
    from representax.core import Route, encode
    from representax.models.clip import CLIPEncoder
    from representax.models.clip.checkpoint import clip_checkpoint_directory
    from representax.models.clip.processing import make_clip_processor
    from representax.precision import precision_context, resolve_precision_policy
    from representax.train import run_job

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        negative_scope=negative_scope,
        symmetric=symmetric,
        warmup_steps=warmup_steps,
        training_file=training_file,
    )
    if jax.device_count() > 1:
        job = data_parallel_job(
            job,
            device_count=jax.device_count(),
            platform=platform,
            training_only=platform == "tpu",
        )
    if platform == "tpu":
        run_directory = run_directory / f"process-{jax.process_index()}"
    started = time.perf_counter()
    if platform == "gpu":
        paused = run_job(job, run_directory, stop_after=steps // 2)
        if paused.completed_iterations != steps // 2:
            raise RuntimeError("Representax did not stop at the midpoint checkpoint")
        del paused
        gc.collect()
        jax.clear_caches()
        completed = run_job(job, run_directory, resume=True)
    else:
        completed = run_job(job, run_directory)
    jax.block_until_ready(completed.state)
    if completed.completed_iterations != steps or completed.resumed != (
        platform == "gpu"
    ):
        raise RuntimeError("Representax did not resume to the final update")
    rows = _metric_rows(run_directory / "metrics.jsonl")
    updates = [row for row in rows if row.get("event") == "training_step"]
    if platform == "tpu":
        return {
            "schema_version": "representax-image-text-worker-v1",
            "framework": "representax",
            "steps": steps,
            "global_batch_size": frozen_contract().global_batch_size,
            "platform": platform,
            "device_count": jax.device_count(),
            "process_count": jax.process_count(),
            "elapsed_seconds": time.perf_counter() - started,
            "steady_state": _steady_state(
                rows,
                frozen_contract().global_batch_size,
            ),
            "final_loss": float(updates[-1]["metrics"]["train/loss"]),
            "training_metrics": [row["metrics"] for row in updates],
            "inference_bundle": None,
        }
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
    _validate_training_manifest(manifest)
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
    platform: Platform = "gpu",
) -> dict[str, Any]:
    import datasets
    import sentence_transformers
    import transformers
    from benchmarks.samplers import sequential_sentence_transformers_batches
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )

    contract = frozen_contract()
    world_size = torch_world_size()
    if contract.global_batch_size % world_size:
        raise ValueError("global batch must divide the accelerator count")
    local_batch_size = contract.global_batch_size // world_size
    if platform == "tpu":
        run_directory = run_directory / f"process-{torch_rank()}"
    if sentence_transformers.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected sentence-transformers=={contract.reference_version}, "
            f"found {sentence_transformers.__version__}"
        )
    model = SentenceTransformer(str(checkpoint), local_files_only=True)
    if platform == "tpu":
        use_fixed_text_padding(model, 77)
    initial_started = time.perf_counter()
    initial_evaluation = (
        _reference_evaluation(model, data_directory, batch_size=EVALUATION_BATCH_SIZE)
        if platform == "gpu"
        else None
    )
    initial_evaluation_seconds = time.perf_counter() - initial_started
    rows = _read_jsonl(data_directory / "train.jsonl")
    train_dataset = datasets.Dataset.from_dict(
        {
            "caption": [row["caption"] for row in rows],
            "image": [str(data_directory / row["image"]) for row in rows],
        }
    ).cast_column("image", datasets.Image())
    loss = (
        MultipleNegativesRankingLoss(model, scale=20.0, gather_across_devices=False)
        if platform == "tpu"
        else CachedMultipleNegativesRankingLoss(
            model,
            scale=20.0,
            mini_batch_size=GRAD_CACHE_MICRO_BATCH,
        )
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=local_batch_size,
        max_steps=steps,
        learning_rate=2e-5,
        optim="adamw_torch_fused" if platform == "gpu" else "adamw_torch",
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
        save_strategy="steps" if platform == "gpu" else "no",
        save_steps=steps // 2,
        save_total_limit=2,
        dataloader_drop_last=True,
        dataloader_num_workers=8 if platform == "gpu" else 0,
        dataloader_prefetch_factor=1 if platform == "gpu" else None,
        dataloader_persistent_workers=platform == "gpu",
        dataloader_pin_memory=platform == "gpu",
        batch_sampler=sequential_sentence_transformers_batches,
        seed=seed,
        data_seed=seed,
    )
    timer = CudaStepTimer()
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[timer.callback()],
    )
    torch_reset_peak_memory()
    started = time.perf_counter()
    output = trainer.train()
    torch_synchronize()
    training_seconds = time.perf_counter() - started
    losses = [
        float(row["loss"])
        for row in trainer.state.log_history
        if row.get("loss") is not None
    ]
    if platform == "tpu":
        return {
            "schema_version": "representax-image-text-worker-v1",
            "framework": "sentence-transformers",
            "framework_version": sentence_transformers.__version__,
            "transformers_version": transformers.__version__,
            "steps": steps,
            "global_batch_size": contract.global_batch_size,
            "platform": platform,
            "device_count": world_size,
            "training_seconds": training_seconds,
            "examples_per_second": contract.global_batch_size
            * steps
            / training_seconds,
            "steady_state": warm_step_summary(
                timer.rows,
                batch_size=contract.global_batch_size,
                excluded_steps=(1, 2),
            ),
            "step_timings": [
                {"step": step, "seconds": duration} for step, duration in timer.rows
            ],
            "losses": losses,
            "inference_bundle": None,
            **torch_device_report(),
        }
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
        "steady_state": warm_step_summary(
            timer.rows,
            batch_size=contract.global_batch_size,
        ),
        "step_timings": [
            {"step": step, "seconds": duration} for step, duration in timer.rows
        ],
        "initial_evaluation_seconds": initial_evaluation_seconds,
        "final_evaluation_seconds": final_evaluation_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_loss": losses[-1],
        "losses": losses,
        "training_metrics": output.metrics,
        "checkpoint": str(midpoint),
        "inference_bundle": str(export),
        "reload_maximum_absolute_difference": reload_difference,
        **torch_device_report(),
    }


def _worker(arguments: argparse.Namespace) -> None:
    _validate_training_manifest(_document(arguments.data_directory / "manifest.json"))
    function = (
        _representax_worker
        if arguments.framework == "representax"
        else _sentence_transformers_worker
    )
    parameters = dict(
        checkpoint=arguments.checkpoint,
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        steps=arguments.steps,
        seed=arguments.seed,
        platform=arguments.platform,
    )
    if arguments.framework == "representax":
        parameters["negative_scope"] = arguments.negative_scope
        parameters["symmetric"] = arguments.symmetric
        parameters["warmup_steps"] = arguments.warmup_steps
        parameters["training_file"] = arguments.training_file
    report = function(**parameters)
    if arguments.platform == "tpu":
        rank = (
            initialize_jax("tpu").process_index()
            if arguments.framework == "representax"
            else torch_rank()
        )
        if rank != 0:
            return
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        report = write_reference_result(
            arguments.report,
            report,
            reference="sentence-transformers",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


def _xla_worker(_index: int, arguments: argparse.Namespace) -> None:
    _worker(arguments)


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
            "experiments.preflights.image_text",
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
            "--platform",
            "gpu",
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
                    "JAX_COMPILATION_CACHE_DIR": os.environ.get(
                        "REPRESENTAX_JAX_CACHE_DIR", str(output / "jax-cache")
                    ),
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
    prepare.add_argument("--captions-per-image", type=int, default=CAPTIONS_PER_IMAGE)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=4)
    worker.add_argument("--seed", type=int, default=7)
    worker.add_argument("--platform", choices=("gpu", "tpu"), default="gpu")
    worker.add_argument(
        "--negative-scope", choices=("local", "global"), default="global"
    )
    worker.add_argument("--symmetric", action="store_true")
    worker.add_argument("--warmup-steps", type=int, default=1)
    worker.add_argument("--training-file", default="train.jsonl")

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
                    captions_per_image=arguments.captions_per_image,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "worker":
        if (
            arguments.framework == "sentence-transformers"
            and arguments.platform == "tpu"
        ):
            import torch_xla

            torch_xla.launch(_xla_worker, args=(arguments,))
        else:
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
