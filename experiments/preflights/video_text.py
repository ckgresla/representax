"""Matched MSR-VTT video-text retrieval preflight for the paper campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.preflights.provenance import reference_source, write_reference_result
from experiments.preflights.timing import CudaStepTimer, warm_step_summary

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
MULTIMODAL_MANIFEST = ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json"
FRAMEWORKS = ("representax", "sentence-transformers")
PREFLIGHT_BATCH_SIZE = 2
PREFLIGHT_TRAINING_VIDEOS = 8
PREFLIGHT_EVALUATION_QUERIES = 8
PREFLIGHT_EVALUATION_DOCUMENTS = 32
GRAD_CACHE_MICRO_BATCH = 1
EVALUATION_BATCH_SIZE = 1
VIDEO_HEIGHT = 224
VIDEO_WIDTH = 224
VIDEO_FPS = 2.0
VIDEO_PIXELS = 32 * 28 * 28


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    train_dataset: Mapping[str, Any]
    evaluation_dataset: Mapping[str, Any]
    reference_version: str
    global_batch_size: int
    video_frames: int


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the video-text row frozen across the paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(MULTIMODAL_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "video-text-retrieval"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "video-text-retrieval"
    )
    if campaign_row["frameworks"] != ["representax", "sentence-transformers"]:
        raise ValueError("the frozen video-text frameworks changed")
    if panel_row["reference"] != "sentence-transformers":
        raise ValueError("the frozen video-text reference changed")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the video-text reference requires a release")
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        train_dataset=panel["datasets"][panel_row["train"]],
        evaluation_dataset=panel["datasets"][panel_row["evaluate"]],
        reference_version=reference.release,
        global_batch_size=int(campaign_row["global_batch"]),
        video_frames=int(campaign_row["video_frames"]),
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


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _decode_video(path: Path, *, frames: int) -> np.ndarray:
    """Decode and uniformly sample one clip through the system ffmpeg binary."""

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        (
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={VIDEO_WIDTH}:{VIDEO_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
        ),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    frame_bytes = VIDEO_HEIGHT * VIDEO_WIDTH * 3
    if not result.stdout or len(result.stdout) % frame_bytes:
        raise ValueError(f"ffmpeg returned an invalid frame buffer for {path}")
    decoded = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        (-1, VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    )
    indices = np.rint(np.linspace(0, len(decoded) - 1, frames)).astype(np.int64)
    return np.ascontiguousarray(decoded[indices])


def _save_video(path: Path, video: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, video, allow_pickle=False)
    os.replace(temporary, path)


def _prepare_training(directory: Path, count: int) -> tuple[Path, ...]:
    import datasets
    from huggingface_hub import hf_hub_download

    contract = frozen_contract()
    source = datasets.load_dataset(
        contract.train_dataset["repo_id"],
        contract.train_dataset["subset"],
        revision=contract.train_dataset["revision"],
        split=contract.train_dataset["split"],
        streaming=True,
    )
    rows = tuple(source.take(count))
    if len(rows) != count:
        raise ValueError(
            f"MSR-VTT produced {len(rows)} training rows; expected {count}"
        )
    records = []
    paths = []
    for source_index, row in enumerate(rows):
        filename = str(row["video"])
        source_path = Path(
            hf_hub_download(
                contract.train_dataset["repo_id"],
                f"raw_videos/{filename}",
                repo_type="dataset",
                revision=contract.train_dataset["revision"],
                cache_dir=os.environ.get("HF_HOME"),
            )
        )
        relative = Path("video/train") / f"{row['video_id']}.npy"
        path = directory / relative
        _save_video(path, _decode_video(source_path, frames=contract.video_frames))
        captions = tuple(
            str(value).strip() for value in row["caption"] if str(value).strip()
        )
        if not captions:
            raise ValueError(f"MSR-VTT training row {source_index} has no caption")
        paths.append(path)
        records.append(
            {
                "source_index": source_index,
                "video_id": str(row["video_id"]),
                "caption": captions[0],
                "video": str(relative),
            }
        )
    _write_jsonl(directory / "train.jsonl", records)
    return tuple(paths)


def _prepare_evaluation(
    directory: Path,
    *,
    query_count: int,
    document_count: int,
) -> tuple[tuple[Path, ...], dict[int, set[int]]]:
    import datasets

    contract = frozen_contract()
    source = (
        datasets.load_dataset(
            contract.evaluation_dataset["repo_id"],
            revision=contract.evaluation_dataset["revision"],
            split=contract.evaluation_dataset["split"],
            streaming=True,
        )
        .cast_column("video", datasets.Video(decode=False))
        .cast_column("audio", datasets.Audio(decode=False))
    )
    rows = tuple(source.take(document_count))
    if len(rows) != document_count:
        raise ValueError(
            f"MSR-VTT produced {len(rows)} evaluation rows; expected {document_count}"
        )
    video_paths = []
    query_records = []
    for index, row in enumerate(rows[:query_count]):
        payload = row["video"].get("bytes")
        if payload is None:
            payload = Path(row["video"]["path"]).read_bytes()
        relative = Path("video/evaluation") / f"{index:04d}.npy"
        path = directory / relative
        with tempfile.NamedTemporaryFile(suffix=".mp4") as stream:
            stream.write(payload)
            stream.flush()
            _save_video(
                path,
                _decode_video(Path(stream.name), frames=contract.video_frames),
            )
        video_paths.append(path)
        query_records.append(
            {
                "kind": "query",
                "identifier": index,
                "video": str(relative),
                "text": "",
                "valid": True,
            }
        )
    documents = [
        {
            "kind": "document",
            "identifier": index,
            "video": "",
            "text": str(row["caption"]).strip(),
            "valid": True,
        }
        for index, row in enumerate(rows)
    ]
    _write_jsonl(directory / "evaluation.jsonl", (*query_records, *documents))
    return tuple(video_paths), {index: {index} for index in range(query_count)}


def prepare_data(
    output: Path,
    *,
    training_videos: int = PREFLIGHT_TRAINING_VIDEOS,
    evaluation_queries: int = PREFLIGHT_EVALUATION_QUERIES,
    evaluation_documents: int = PREFLIGHT_EVALUATION_DOCUMENTS,
) -> dict[str, Any]:
    """Materialize deterministic, bounded MSR-VTT preflight data."""

    if min(training_videos, evaluation_queries, evaluation_documents) <= 0:
        raise ValueError("MSR-VTT preflight counts must be positive")
    if evaluation_queries > evaluation_documents:
        raise ValueError("evaluation queries must not exceed evaluation documents")
    output.mkdir(parents=True, exist_ok=False)
    training_paths = _prepare_training(output, training_videos)
    evaluation_paths, relevant = _prepare_evaluation(
        output,
        query_count=evaluation_queries,
        document_count=evaluation_documents,
    )
    files = {
        name: {"rows": sum(1 for _ in path.open()), "sha256": _sha256(path)}
        for name, path in (
            ("train.jsonl", output / "train.jsonl"),
            ("evaluation.jsonl", output / "evaluation.jsonl"),
        )
    }
    manifest = {
        "schema_version": "representax-video-text-preflight-data-v1",
        "contract": asdict(frozen_contract()),
        "video_frames": frozen_contract().video_frames,
        "video_fps": VIDEO_FPS,
        "video_shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
        "training_videos": training_videos,
        "training_presentations": training_videos,
        "evaluation_queries": evaluation_queries,
        "evaluation_documents": evaluation_documents,
        "relevant_documents": {
            str(query): sorted(documents) for query, documents in relevant.items()
        },
        "files": files,
        "training_video_tree_sha256": _tree_sha256(training_paths, output),
        "evaluation_video_tree_sha256": _tree_sha256(evaluation_paths, output),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _load_video(root: Path, relative: str) -> np.ndarray:
    return np.load(root / relative, allow_pickle=False)


def _representax_video(root: Path, relative: str) -> dict[str, Any]:
    return {"frames": _load_video(root, relative), "fps": VIDEO_FPS}


def _reference_video(root: Path, relative: str) -> dict[str, Any]:
    frames = _load_video(root, relative)
    return {
        "array": frames,
        "video_metadata": {
            "fps": VIDEO_FPS,
            "total_num_frames": len(frames),
            "frames_indices": list(range(len(frames))),
        },
    }


class VideoTextRetrievalCollator:
    """Build aligned video-query and caption-document MNR batches."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-video-text-collator-v1",
            "processor": self.processor.data_contract(),
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        from representax.core import Route
        from representax.tasks.retrieval import retrieval_batch

        size = len(rows)
        return retrieval_batch(
            query=self.processor(
                tuple(
                    {
                        "video": _representax_video(
                            self.root_directory, str(row["video"])
                        )
                    }
                    for row in rows
                ),
                route=Route.QUERY,
            ),
            document=self.processor(
                tuple(str(row["caption"]) for row in rows),
                route=Route.DOCUMENT,
            ),
            positive_mask=np.eye(size, dtype=np.bool_),
        )


class VideoTextEvaluationCollator:
    """Build homogeneous video-query or caption-document evaluation batches."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-video-text-evaluation-collator-v1",
            "processor": self.processor.data_contract(),
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.core import Route
        from representax.evaluation import retrieval_evaluation_batch

        kinds = {str(row["kind"]) for row in rows}
        if len(kinds) != 1:
            raise ValueError("video-text evaluation batches must be homogeneous")
        kind = kinds.pop()
        if kind == "query":
            inputs = self.processor(
                tuple(
                    {
                        "video": _representax_video(
                            self.root_directory, str(row["video"])
                        )
                    }
                    for row in rows
                ),
                route=Route.QUERY,
            )
        elif kind == "document":
            inputs = self.processor(
                tuple(str(row["text"]) for row in rows),
                route=Route.DOCUMENT,
            )
        else:
            raise ValueError(f"unknown video-text evaluation kind {kind!r}")
        return retrieval_evaluation_batch(
            inputs,
            jnp.asarray(tuple(int(row["identifier"]) for row in rows)),
            kind=kind,
            valid=jnp.asarray(tuple(bool(row["valid"]) for row in rows)),
        )


def _representax_job(
    *,
    checkpoint: Path,
    data_directory: Path,
    steps: int,
    seed: int,
    batch_size: int = PREFLIGHT_BATCH_SIZE,
    export_enabled: bool = True,
) -> Any:
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    if batch_size < 2:
        raise ValueError("cached MNR requires at least one in-batch negative")
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        DDPConfig,
        EvaluationConfig,
        ExportConfig,
        GradCacheConfig,
        HuggingFaceExportConfig,
        InformationRetrievalEvaluatorConfig,
        JobConfig,
        LoggingConfig,
        LoRAConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig

    manifest = _document(data_directory / "manifest.json")
    if int(manifest["training_presentations"]) < batch_size * steps:
        raise ValueError("preflight data does not contain enough presentations")
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
            num_threads=2,
            prefetch_buffer_size=2,
        )

    contract = frozen_contract()
    return JobConfig(
        name="paper-preflight-video-text-retrieval",
        model=ModelConfig(
            target="representax.models.qwen2_5_omni:load_qwen2_5_omni",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": [256],
                "patch_count_buckets": [800],
                "audio_chunk_count_buckets": [1],
                "audio_token_count_buckets": [64],
                "video_min_pixels": VIDEO_PIXELS,
                "video_max_pixels": VIDEO_PIXELS,
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
            "experiments.preflights.video_text:VideoTextRetrievalCollator",
        ),
        training=TrainingConfig(
            global_batch_size=batch_size,
            max_steps=steps,
            seed=seed,
            mesh=MeshConfig(axis_shapes=(1,), axis_names=("data",)),
            sharding=DDPConfig(axis="data"),
            batch=BatchConfig(micro_batch_size=batch_size),
            grad_cache=GradCacheConfig(micro_batch_size=GRAD_CACHE_MICRO_BATCH),
            adapter=LoRAConfig(rank=4, alpha=8.0, target_pattern="text"),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(every=steps // 2, keep=2, save_final=True),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        evaluation=EvaluationConfig(
            data=data(
                data_directory / "evaluation.jsonl",
                "experiments.preflights.video_text:VideoTextEvaluationCollator",
            ),
            batch_size=EVALUATION_BATCH_SIZE,
            evaluators=(
                InformationRetrievalEvaluatorConfig(
                    name="msrvtt-preflight",
                    relevant_documents=relevant,
                    score_functions=("cosine",),
                    main_score_function="cosine",
                    accuracy_at_k=(1, 5),
                    precision_recall_at_k=(1, 5, 10),
                    mrr_at_k=(10,),
                    ndcg_at_k=(10,),
                    map_at_k=(10,),
                ),
            ),
            on_start=True,
            on_end=True,
            primary_metric="valid/msrvtt-preflight/cosine_recall@10",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=(
            ExportConfig(
                selection="final",
                huggingface=HuggingFaceExportConfig(
                    source_checkpoint=str(checkpoint),
                    adapter=ComponentConfig(
                        target=(
                            "representax.models.qwen2_5_omni:"
                            "Qwen2_5OmniCheckpointAdapter"
                        ),
                        parameters={"rematerialization": "none"},
                    ),
                    verify_reload=True,
                ),
            )
            if export_enabled
            else ExportConfig(enabled=False)
        ),
    )


def _timing(rows: Sequence[Mapping[str, Any]], batch_size: int) -> dict[str, Any]:
    compilations = []
    durations = []
    latest = {
        int(row["iteration"]): row
        for row in rows
        if row.get("event") == "training_step"
    }
    for row in (latest[iteration] for iteration in sorted(latest)):
        metrics = row["metrics"]
        if value := metrics.get("perf/compilation_and_first_step_seconds"):
            compilations.append(float(value))
            continue
        if value := metrics.get("perf/step_seconds"):
            durations.append(float(value))
    return {
        "compilation_and_first_step_seconds": compilations,
        "warmed_steps": len(durations),
        "median_warmed_step_seconds": (
            None if not durations else statistics.median(durations)
        ),
        "warmed_examples_per_second": (
            None if not durations else batch_size * len(durations) / sum(durations)
        ),
    }


def _representax_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import jax

    from representax import load_inference_bundle
    from representax.config import PrecisionConfig
    from representax.core import Route, encode
    from representax.models.qwen2_5_omni import (
        Qwen2_5OmniEncoder,
        make_qwen2_5_omni_processor,
    )
    from representax.precision import precision_context, resolve_precision_policy
    from representax.train import run_job

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("video-text preflight requires exactly one visible GPU")
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        batch_size=batch_size,
    )
    started = time.perf_counter()
    if not (run_directory / "run.json").is_file():
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
    if not isinstance(trained_model, Qwen2_5OmniEncoder):
        raise TypeError("video-text training returned a different model family")

    processor = make_qwen2_5_omni_processor(
        checkpoint,
        trained_model.config,
        sequence_length_buckets=(256,),
        patch_count_buckets=(800,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(64,),
        video_min_pixels=VIDEO_PIXELS,
        video_max_pixels=VIDEO_PIXELS,
    )
    probe_rows = _read_jsonl(data_directory / "train.jsonl")[
        : min(batch_size, PREFLIGHT_BATCH_SIZE)
    ]
    probe = VideoTextRetrievalCollator(
        processor=processor,
        root_directory=data_directory,
    )(probe_rows)
    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    with precision_context(precision):
        expected = (
            encode(trained_model, probe.query, route=Route.QUERY),
            encode(trained_model, probe.document, route=Route.DOCUMENT),
        )
    jax.block_until_ready(expected)
    restored, restored_job = load_inference_bundle(completed.inference_bundle)
    if restored_job.name != job.name or not isinstance(restored, Qwen2_5OmniEncoder):
        raise RuntimeError("native reload reconstructed a different model or job")
    restored = jax.tree.map(
        lambda value, template: (
            jax.device_put(value, template.sharding)
            if isinstance(value, jax.Array) and isinstance(template, jax.Array)
            else value
        ),
        restored,
        trained_model,
    )
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
        raise RuntimeError("native inference reload changed video-text embeddings")

    rows = _read_jsonl(run_directory / "metrics.jsonl")
    updates = list(
        {
            int(row["iteration"]): row
            for row in rows
            if row.get("event") == "training_step"
        }.values()
    )
    evaluations = list(
        {
            int(row["iteration"]): row
            for row in rows
            if row.get("event") == "evaluation"
        }.values()
    )
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
        "schema_version": "representax-video-text-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": batch_size,
        "frozen_global_batch_size": frozen_contract().global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "elapsed_seconds": time.perf_counter() - started,
        "timing": _timing(rows, batch_size),
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
        "final_update_global_norm": update_norms[-1],
        "resumed": completed.resumed,
        "checkpoint": str(run_directory / "checkpoints" / str(steps // 2)),
        "inference_bundle": str(completed.inference_bundle),
        "huggingface_export": str(completed.inference_bundle / "huggingface"),
        "native_reload_maximum_absolute_difference": reload_difference,
        "device": jax.devices()[0].device_kind,
    }


def _reference_evaluation(
    model: Any,
    data_directory: Path,
    *,
    batch_size: int,
) -> dict[str, float]:
    from representax.evaluation.retrieval import information_retrieval_metrics

    records = _read_jsonl(data_directory / "evaluation.jsonl")
    queries = [row for row in records if row["kind"] == "query" and row["valid"]]
    documents = [row for row in records if row["kind"] == "document" and row["valid"]]
    query_embeddings = model.encode(
        [
            {"video": _reference_video(data_directory, str(row["video"]))}
            for row in queries
        ],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    document_embeddings = model.encode(
        [str(row["text"]) for row in documents],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    query_embeddings /= np.maximum(
        np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12
    )
    document_embeddings /= np.maximum(
        np.linalg.norm(document_embeddings, axis=1, keepdims=True), 1e-12
    )
    top = np.argsort(-(query_embeddings @ document_embeddings.T), axis=1)[:, :10]
    ranked = np.asarray([int(row["identifier"]) for row in documents])[top]
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
        accuracy_at_k=(1, 5),
        precision_recall_at_k=(1, 5, 10),
        mrr_at_k=(10,),
        ndcg_at_k=(10,),
        map_at_k=(10,),
    )
    return {
        f"valid/msrvtt-preflight/cosine_{name}": value
        for name, value in metrics.items()
    }


class _StopAtMidpoint:
    """Construct the tiny Trainer callback lazily inside the reference process."""

    @staticmethod
    def build(step: int) -> Any:
        from transformers import TrainerCallback

        class StopAtMidpointCallback(TrainerCallback):
            def on_step_end(
                self, _args: Any, state: Any, control: Any, **_: Any
            ) -> Any:
                if state.global_step == step:
                    control.should_training_stop = True
                return control

        return StopAtMidpointCallback()


def _sentence_transformers_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import datasets
    import sentence_transformers
    import torch
    import transformers
    from benchmarks.samplers import sequential_sentence_transformers_batches
    from peft import LoraConfig
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )

    class PreflightTrainer(SentenceTransformerTrainer):
        def add_model_card_callback(self, _default_args_dict: dict[str, Any]) -> None:
            # Its dataset-statistics pass recursively tensorizes every video frame.
            return None

    contract = frozen_contract()
    if sentence_transformers.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected sentence-transformers=={contract.reference_version}, "
            f"found {sentence_transformers.__version__}"
        )
    model = SentenceTransformer(
        str(checkpoint),
        device="cuda",
        local_files_only=True,
        model_kwargs={"dtype": torch.bfloat16},
    )
    model[0].processing_kwargs.update(
        {
            "video": {
                "min_pixels": VIDEO_PIXELS,
                "max_pixels": VIDEO_PIXELS,
                "do_sample_frames": False,
            }
        }
    )
    model.add_adapter(
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=(
                r"model\.layers\.\d+\."
                r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
                r"mlp\.(?:gate_proj|up_proj|down_proj))"
            ),
            bias="none",
        )
    )
    training_prompt = model.prompts.get(model.default_prompt_name or "")
    if not training_prompt:
        raise RuntimeError("the frozen video-text checkpoint has no default prompt")
    initial_started = time.perf_counter()
    initial_evaluation = _reference_evaluation(
        model, data_directory, batch_size=EVALUATION_BATCH_SIZE
    )
    initial_evaluation_seconds = time.perf_counter() - initial_started
    rows = _read_jsonl(data_directory / "train.jsonl")[: batch_size * steps]
    train_dataset = datasets.Dataset.from_dict(
        {
            "video": [str(row["video"]) for row in rows],
            "caption": [str(row["caption"]) for row in rows],
        }
    )

    def load_video_batch(batch: Mapping[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            "video": [
                _reference_video(data_directory, str(relative))
                for relative in batch["video"]
            ],
            "caption": list(batch["caption"]),
        }

    train_dataset.set_transform(load_video_batch)
    loss = CachedMultipleNegativesRankingLoss(
        model, scale=20.0, mini_batch_size=GRAD_CACHE_MICRO_BATCH
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=batch_size,
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
        prompts=training_prompt,
    )
    midpoint_step = steps // 2
    timer = CudaStepTimer()
    midpoint_callback = _StopAtMidpoint.build(midpoint_step)
    trainer = PreflightTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=loss,
        callbacks=[midpoint_callback, timer.callback()],
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.train()
    midpoint = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
    if trainer.state.global_step != steps // 2 or not midpoint.is_dir():
        raise RuntimeError(
            "Sentence Transformers did not stop at its midpoint checkpoint"
        )
    trainer.remove_callback(type(midpoint_callback))
    output = trainer.train(resume_from_checkpoint=str(midpoint))
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    if trainer.state.global_step != steps:
        raise RuntimeError("Sentence Transformers did not resume to the final update")
    final_started = time.perf_counter()
    final_evaluation = _reference_evaluation(
        model, data_directory, batch_size=EVALUATION_BATCH_SIZE
    )
    final_evaluation_seconds = time.perf_counter() - final_started
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    final_checkpoint = run_directory / "checkpoints" / f"checkpoint-{steps}"
    if not final_checkpoint.is_dir():
        raise RuntimeError("Sentence Transformers did not save its final checkpoint")
    probe = {"video": _reference_video(data_directory, str(rows[0]["video"]))}
    losses = [
        float(row["loss"])
        for row in trainer.state.log_history
        if row.get("loss") is not None
    ]
    peak_device_bytes = int(torch.cuda.max_memory_allocated())
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = torch.cuda.get_device_name()
    del trainer, loss, model
    gc.collect()
    torch.cuda.empty_cache()

    def load_probe(path: Path) -> np.ndarray:
        restored = SentenceTransformer(
            str(path),
            device="cuda",
            local_files_only=True,
            model_kwargs={"dtype": torch.bfloat16},
        )
        restored[0].processing_kwargs.update(
            {
                "video": {
                    "min_pixels": VIDEO_PIXELS,
                    "max_pixels": VIDEO_PIXELS,
                    "do_sample_frames": False,
                }
            }
        )
        embedding = restored.encode([probe], convert_to_numpy=True)
        del restored
        gc.collect()
        torch.cuda.empty_cache()
        return embedding

    midpoint_probe = load_probe(midpoint)
    expected = load_probe(final_checkpoint)
    actual = load_probe(export)
    reload_difference = float(np.max(np.abs(expected - actual)))
    if not np.array_equal(expected, actual) or not np.all(np.isfinite(midpoint_probe)):
        raise RuntimeError("Sentence Transformers checkpoint or export reload failed")
    warmed = warm_step_summary(
        timer.rows,
        batch_size=batch_size,
        excluded_steps=(1, midpoint_step + 1),
    )
    return {
        "schema_version": "representax-video-text-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "global_batch_size": batch_size,
        "frozen_global_batch_size": contract.global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "training_seconds": training_seconds,
        "timing": {
            "execution": "eager",
            "compilation_and_first_step_seconds": [],
            "warmed_steps": warmed["measured_steps"],
            "median_warmed_step_seconds": warmed["median_step_seconds"],
            "warmed_examples_per_second": warmed["examples_per_second"],
        },
        "initial_evaluation_seconds": initial_evaluation_seconds,
        "final_evaluation_seconds": final_evaluation_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_loss": losses[-1],
        "training_metrics": output.metrics,
        "checkpoint": str(midpoint),
        "inference_bundle": str(export),
        "reload_maximum_absolute_difference": reload_difference,
        "peak_device_bytes": peak_device_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "trainable_parameters": trainable_parameters,
        "device": device,
        "resumed": True,
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
        batch_size=arguments.batch_size,
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
            "experiments.preflights.video_text",
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
            "--batch-size",
            str(arguments.batch_size),
        ]
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
        if framework == "representax":
            environment.update(
                {
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
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
        "schema_version": "representax-video-text-preflight-v1",
        "scope": "bounded-readiness-preflight-not-paper-result",
        "contract": {
            **asdict(frozen_contract()),
            "preflight_batch_size": arguments.batch_size,
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
    prepare.add_argument(
        "--training-videos", type=int, default=PREFLIGHT_TRAINING_VIDEOS
    )
    prepare.add_argument(
        "--evaluation-queries", type=int, default=PREFLIGHT_EVALUATION_QUERIES
    )
    prepare.add_argument(
        "--evaluation-documents", type=int, default=PREFLIGHT_EVALUATION_DOCUMENTS
    )

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=4)
    worker.add_argument("--seed", type=int, default=7)
    worker.add_argument("--batch-size", type=int, default=PREFLIGHT_BATCH_SIZE)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=4)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--batch-size", type=int, default=PREFLIGHT_BATCH_SIZE)
    pair.add_argument("--gpu", type=int, default=0)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    training_videos=arguments.training_videos,
                    evaluation_queries=arguments.evaluation_queries,
                    evaluation_documents=arguments.evaluation_documents,
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
    "VideoTextEvaluationCollator",
    "VideoTextRetrievalCollator",
    "frozen_contract",
    "prepare_data",
]
