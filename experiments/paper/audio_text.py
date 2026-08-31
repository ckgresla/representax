"""Matched AudioCaps audio-text retrieval preflight for the paper campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
MULTIMODAL_MANIFEST = ROOT / "benchmarks/configs/paper-multimodal-jepa-v1.json"
FRAMEWORKS = ("representax", "sentence-transformers")
SAMPLE_RATE = 16_000
PREFLIGHT_BATCH_SIZE = 8
PREFLIGHT_TRAINING_AUDIOS = 32
PREFLIGHT_TRAINING_REPEATS = 4
PREFLIGHT_EVALUATION_QUERIES = 16
PREFLIGHT_EVALUATION_DOCUMENTS = 128
GRAD_CACHE_MICRO_BATCH = 1
EVALUATION_BATCH_SIZE = 4


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    train_dataset: Mapping[str, Any]
    evaluation_dataset: Mapping[str, Any]
    reference_version: str
    global_batch_size: int
    audio_seconds: int


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the audio-text row frozen across the paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(MULTIMODAL_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "audio-text-retrieval"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "audio-text-retrieval"
    )
    if campaign_row["frameworks"] != ["representax", "sentence-transformers"]:
        raise ValueError("the frozen audio-text frameworks changed")
    if panel_row["reference"] != "sentence-transformers":
        raise ValueError("the frozen audio-text reference changed")
    model = panel["models"][panel_row["model"]]
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        train_dataset=panel["datasets"][panel_row["train"]],
        evaluation_dataset=panel["datasets"][panel_row["evaluate"]],
        reference_version=panel["references"][panel_row["reference"]],
        global_batch_size=int(campaign_row["global_batch"]),
        audio_seconds=int(campaign_row["audio_seconds"]),
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


def _normalize_audio(
    values: np.ndarray,
    sampling_rate: int,
    *,
    seconds: int,
) -> np.ndarray:
    """Convert one waveform to fixed-duration mono float32 at 16 kHz."""

    audio = np.asarray(values)
    if audio.ndim == 2:
        audio = audio.astype(np.float32).mean(axis=1)
    elif audio.ndim != 1:
        raise ValueError(
            f"audio must be mono or channel-last stereo, got {audio.shape}"
        )
    if np.issubdtype(audio.dtype, np.integer):
        bits = audio.dtype.itemsize * 8
        if np.issubdtype(audio.dtype, np.unsignedinteger):
            midpoint = float(2 ** (bits - 1))
            audio = (audio.astype(np.float32) - midpoint) / midpoint
        else:
            audio = audio.astype(np.float32) / float(2 ** (bits - 1))
    else:
        audio = audio.astype(np.float32)
    if sampling_rate <= 0:
        raise ValueError("audio sampling rate must be positive")
    if sampling_rate != SAMPLE_RATE:
        from scipy.signal import resample_poly

        divisor = math.gcd(sampling_rate, SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sampling_rate // divisor,
        ).astype(np.float32)
    expected = SAMPLE_RATE * seconds
    if audio.size < expected:
        audio = np.pad(audio, (0, expected - audio.size))
    return np.ascontiguousarray(audio[:expected], dtype=np.float32)


def _decode_wav(payload: bytes, *, seconds: int) -> np.ndarray:
    from scipy.io import wavfile

    sampling_rate, values = wavfile.read(io.BytesIO(payload))
    return _normalize_audio(values, int(sampling_rate), seconds=seconds)


def _save_audio(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, audio, allow_pickle=False)
    os.replace(temporary, path)


def _prepare_training(directory: Path, count: int) -> tuple[Path, ...]:
    import datasets

    contract = frozen_contract()
    source = datasets.load_dataset(
        contract.train_dataset["repo_id"],
        revision=contract.train_dataset["revision"],
        split=contract.train_dataset["split"],
        streaming=True,
    ).cast_column("audio", datasets.Audio(decode=False))
    rows = tuple(source.take(count))
    if len(rows) != count:
        raise ValueError(f"AudioCaps produced {len(rows)} rows; expected {count}")
    records = []
    paths = []
    for source_index, row in enumerate(rows):
        payload = row["audio"].get("bytes")
        if payload is None:
            payload = Path(row["audio"]["path"]).read_bytes()
        relative = Path("audio/train") / f"{int(row['audiocap_id']):08d}.npy"
        path = directory / relative
        _save_audio(
            path,
            _decode_wav(payload, seconds=contract.audio_seconds),
        )
        paths.append(path)
        records.append(
            {
                "source_index": source_index,
                "audiocap_id": int(row["audiocap_id"]),
                "caption": str(row["caption"]).strip(),
                "audio": str(relative),
            }
        )
    presentations = (
        {**record, "presentation_cycle": cycle}
        for cycle in range(PREFLIGHT_TRAINING_REPEATS)
        for record in records
    )
    _write_jsonl(directory / "train.jsonl", presentations)
    return tuple(paths)


def _prepare_evaluation(
    directory: Path,
    *,
    query_count: int,
    document_count: int,
) -> tuple[tuple[Path, ...], dict[int, set[int]]]:
    import datasets

    contract = frozen_contract()
    common = {
        "path": contract.evaluation_dataset["repo_id"],
        "revision": contract.evaluation_dataset["revision"],
    }
    queries = tuple(
        datasets.load_dataset(
            **common,
            name="query",
            split=contract.evaluation_dataset["split"],
            streaming=True,
        ).take(query_count)
    )
    corpus = tuple(datasets.load_dataset(**common, name="corpus", split="test"))
    qrels = tuple(datasets.load_dataset(**common, name="qrels", split="test"))
    query_source_ids = {str(row["id"]) for row in queries}
    relevant_source = {
        (str(row["query-id"]), str(row["corpus-id"]))
        for row in qrels
        if str(row["query-id"]) in query_source_ids and float(row["score"]) > 0
    }
    relevant_documents = {document for _, document in relevant_source}
    selected_documents = []
    for row in corpus:
        identifier = str(row["id"])
        if identifier in relevant_documents:
            selected_documents.append(row)
    for row in corpus:
        if len(selected_documents) >= document_count:
            break
        identifier = str(row["id"])
        if identifier not in relevant_documents:
            selected_documents.append(row)
    if len(selected_documents) != document_count:
        raise ValueError("AudioCaps corpus is too small for the preflight panel")

    query_ids = {str(row["id"]): index for index, row in enumerate(queries)}
    document_ids = {
        str(row["id"]): index for index, row in enumerate(selected_documents)
    }
    relevant: dict[int, set[int]] = {}
    for query_source, document_source in relevant_source:
        relevant.setdefault(query_ids[query_source], set()).add(
            document_ids[document_source]
        )
    if set(relevant) != set(query_ids.values()):
        raise ValueError("AudioCaps qrels do not cover every selected query")

    audio_paths = []
    query_records = []
    for row in queries:
        source_id = str(row["id"])
        relative = Path("audio/evaluation") / f"{source_id}.npy"
        path = directory / relative
        value = row["audio"]
        _save_audio(
            path,
            _normalize_audio(
                np.asarray(value["array"]),
                int(value["sampling_rate"]),
                seconds=contract.audio_seconds,
            ),
        )
        audio_paths.append(path)
        query_records.append(
            {
                "kind": "query",
                "identifier": query_ids[source_id],
                "audio": str(relative),
                "text": "",
                "valid": True,
            }
        )
    documents = [
        {
            "kind": "document",
            "identifier": document_ids[str(row["id"])],
            "audio": "",
            "text": str(row["text"]),
            "valid": True,
        }
        for row in selected_documents
    ]
    _write_jsonl(directory / "evaluation.jsonl", (*query_records, *documents))
    return tuple(audio_paths), relevant


def prepare_data(
    output: Path,
    *,
    training_audios: int = PREFLIGHT_TRAINING_AUDIOS,
    evaluation_queries: int = PREFLIGHT_EVALUATION_QUERIES,
    evaluation_documents: int = PREFLIGHT_EVALUATION_DOCUMENTS,
) -> dict[str, Any]:
    """Materialize deterministic, bounded AudioCaps preflight data."""

    if min(training_audios, evaluation_queries, evaluation_documents) <= 0:
        raise ValueError("AudioCaps preflight counts must be positive")
    output.mkdir(parents=True, exist_ok=False)
    training_paths = _prepare_training(output, training_audios)
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
        "schema_version": "representax-audio-text-preflight-data-v1",
        "contract": asdict(frozen_contract()),
        "sample_rate": SAMPLE_RATE,
        "training_audios": training_audios,
        "training_repeats": PREFLIGHT_TRAINING_REPEATS,
        "training_presentations": training_audios * PREFLIGHT_TRAINING_REPEATS,
        "evaluation_queries": evaluation_queries,
        "evaluation_documents": evaluation_documents,
        "relevant_documents": {
            str(query): sorted(documents) for query, documents in relevant.items()
        },
        "files": files,
        "training_audio_tree_sha256": _tree_sha256(training_paths, output),
        "evaluation_audio_tree_sha256": _tree_sha256(evaluation_paths, output),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _load_audio(root: Path, relative: str) -> np.ndarray:
    return np.load(root / relative, allow_pickle=False)


class AudioTextRetrievalCollator:
    """Build aligned audio-query and caption-document MNR batches."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-audio-text-collator-v1",
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
                    {"audio": _load_audio(self.root_directory, str(row["audio"]))}
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


class AudioTextEvaluationCollator:
    """Build homogeneous audio-query or caption-document evaluation batches."""

    def __init__(self, *, processor: Any, root_directory: str | Path) -> None:
        self.processor = processor
        self.root_directory = Path(root_directory).resolve()

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-audio-text-evaluation-collator-v1",
            "processor": self.processor.data_contract(),
            "root_directory": str(self.root_directory),
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.core import Route
        from representax.evaluation import retrieval_evaluation_batch

        kinds = {str(row["kind"]) for row in rows}
        if len(kinds) != 1:
            raise ValueError("audio-text evaluation batches must be homogeneous")
        kind = kinds.pop()
        if kind == "query":
            inputs = self.processor(
                tuple(
                    {"audio": _load_audio(self.root_directory, str(row["audio"]))}
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
            raise ValueError(f"unknown audio-text evaluation kind {kind!r}")
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
    batch_size: int,
    world_size: int = 2,
    sharding: str = "ddp",
    export_enabled: bool = True,
) -> Any:
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    if batch_size < 2:
        raise ValueError("cached MNR requires at least one in-batch negative")
    if sharding not in {"ddp", "fsdp"}:
        raise ValueError(f"unknown Representax sharding strategy {sharding!r}")
    if sharding == "ddp" and batch_size % world_size:
        raise ValueError("DDP global batch size must be divisible by the world size")
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        DDPConfig,
        EvaluationConfig,
        ExportConfig,
        FSDPConfig,
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
    if sharding == "ddp":
        mesh = MeshConfig(axis_shapes=(world_size,), axis_names=("data",))
        sharding_config = DDPConfig(axis="data")
        local_batch_size = batch_size // world_size
    else:
        mesh = MeshConfig(axis_shapes=(world_size,), axis_names=("model",))
        sharding_config = FSDPConfig(data_axis=None, parameter_axis="model")
        local_batch_size = batch_size

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
        name="paper-preflight-audio-text-retrieval",
        model=ModelConfig(
            target="representax.models.qwen2_5_omni:load_qwen2_5_omni",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": [512],
                "patch_count_buckets": [64],
                "audio_chunk_count_buckets": [16],
                "audio_token_count_buckets": [256],
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
            "experiments.paper.audio_text:AudioTextRetrievalCollator",
        ),
        training=TrainingConfig(
            global_batch_size=batch_size,
            max_steps=steps,
            seed=seed,
            mesh=mesh,
            sharding=sharding_config,
            batch=BatchConfig(micro_batch_size=local_batch_size),
            grad_cache=GradCacheConfig(micro_batch_size=GRAD_CACHE_MICRO_BATCH),
            adapter=LoRAConfig(
                rank=4,
                alpha=8.0,
                target_pattern="text",
            ),
            activation_rematerialization="none",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=CheckpointConfig(every=steps // 2, keep=2, save_final=True),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        evaluation=EvaluationConfig(
            data=data(
                data_directory / "evaluation.jsonl",
                "experiments.paper.audio_text:AudioTextEvaluationCollator",
            ),
            batch_size=EVALUATION_BATCH_SIZE,
            evaluators=(
                InformationRetrievalEvaluatorConfig(
                    name="audiocaps-preflight",
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
            primary_metric="valid/audiocaps-preflight/cosine_recall@10",
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
    batch_size: int,
    sharding: str = "ddp",
    skip_export: bool = False,
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

    world_size = len(jax.devices())
    if jax.default_backend() != "gpu" or world_size not in (1, 2):
        raise RuntimeError("audio-text preflight requires one or two visible GPUs")
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        batch_size=batch_size,
        world_size=world_size,
        sharding=sharding,
        export_enabled=not skip_export,
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
    if completed.inference_bundle is None and not skip_export:
        raise RuntimeError("Representax did not produce an inference bundle")
    trained_model = completed.state.model
    if not isinstance(trained_model, Qwen2_5OmniEncoder):
        raise TypeError("audio-text training returned a different model family")
    reload_difference = None
    if completed.inference_bundle is not None:
        processor = make_qwen2_5_omni_processor(
            checkpoint,
            trained_model.config,
            sequence_length_buckets=(512,),
            patch_count_buckets=(64,),
            audio_chunk_count_buckets=(16,),
            audio_token_count_buckets=(256,),
        )
        probe_rows = _read_jsonl(data_directory / "train.jsonl")[:2]
        probe = AudioTextRetrievalCollator(
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
        if restored_job.name != job.name or not isinstance(
            restored, Qwen2_5OmniEncoder
        ):
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
            raise RuntimeError("native inference reload changed audio-text embeddings")

    rows = _read_jsonl(run_directory / "metrics.jsonl")
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
        "schema_version": "representax-audio-text-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": batch_size,
        "local_batch_size": (
            batch_size // world_size if sharding == "ddp" else batch_size
        ),
        "world_size": world_size,
        "sharding": sharding,
        "frozen_global_batch_size": frozen_contract().global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "elapsed_seconds": time.perf_counter() - started,
        "steady_state": _steady_state(rows, batch_size),
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
        "inference_bundle": (
            None
            if completed.inference_bundle is None
            else str(completed.inference_bundle)
        ),
        "huggingface_export": (
            None
            if completed.inference_bundle is None
            else str(completed.inference_bundle / "huggingface")
        ),
        "native_reload_maximum_absolute_difference": reload_difference,
        "devices": [device.device_kind for device in jax.devices()],
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
            {
                "array": _load_audio(data_directory, str(row["audio"])),
                "sampling_rate": SAMPLE_RATE,
            }
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
        f"valid/audiocaps-preflight/cosine_{name}": value
        for name, value in metrics.items()
    }


def _sentence_transformers_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import gc

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
    initial_started = time.perf_counter()
    initial_evaluation = _reference_evaluation(
        model,
        data_directory,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    initial_evaluation_seconds = time.perf_counter() - initial_started
    rows = _read_jsonl(data_directory / "train.jsonl")[: batch_size * steps]
    train_dataset = datasets.Dataset.from_dict(
        {
            "audio": [
                {
                    "array": _load_audio(data_directory, str(row["audio"])),
                    "sampling_rate": SAMPLE_RATE,
                }
                for row in rows
            ],
            "caption": [str(row["caption"]) for row in rows],
        }
    ).with_format("numpy", columns=["audio"], output_all_columns=True)
    loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=GRAD_CACHE_MICRO_BATCH,
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
        model,
        data_directory,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    final_evaluation_seconds = time.perf_counter() - final_started
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    midpoint = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
    final_checkpoint = run_directory / "checkpoints" / f"checkpoint-{steps}"
    if not midpoint.is_dir() or not final_checkpoint.is_dir():
        raise RuntimeError("Sentence Transformers did not save resumable checkpoints")
    probe = {
        "array": _load_audio(data_directory, str(rows[0]["audio"])),
        "sampling_rate": SAMPLE_RATE,
    }
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
    return {
        "schema_version": "representax-audio-text-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "global_batch_size": batch_size,
        "frozen_global_batch_size": contract.global_batch_size,
        "grad_cache_micro_batch_size": GRAD_CACHE_MICRO_BATCH,
        "training_seconds": training_seconds,
        "examples_per_second": batch_size * steps / training_seconds,
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
    }


def _worker(arguments: argparse.Namespace) -> None:
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
        batch_size=arguments.batch_size,
    )
    if arguments.framework == "representax":
        parameters["skip_export"] = arguments.skip_export
        parameters["sharding"] = arguments.sharding
    elif arguments.skip_export:
        raise ValueError("--skip-export is only available for Representax probes")
    report = function(**parameters)
    _write_json(arguments.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    commands = {}
    gpu_assignments = {
        "representax": arguments.representax_gpus,
        "sentence-transformers": str(arguments.reference_gpu),
    }
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        log = output / f"{framework}.log"
        run_directory = output / framework
        command = [
            sys.executable,
            "-m",
            "experiments.paper.audio_text",
            "worker",
            "--framework",
            framework,
            "--checkpoint",
            str(arguments.checkpoint),
            "--data-directory",
            str(arguments.data_directory),
            "--run-directory",
            str(run_directory),
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
            "CUDA_VISIBLE_DEVICES": gpu_assignments[framework],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
        if framework == "representax":
            command.extend(("--sharding", arguments.representax_sharding))
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
        "schema_version": "representax-audio-text-preflight-v1",
        "scope": "bounded-readiness-preflight-not-paper-result",
        "contract": {
            **asdict(frozen_contract()),
            "preflight_batch_size": arguments.batch_size,
            "steps": arguments.steps,
            "seed": arguments.seed,
            "representax_gpus": arguments.representax_gpus,
            "reference_gpu": arguments.reference_gpu,
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
        "--training-audios", type=int, default=PREFLIGHT_TRAINING_AUDIOS
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
    worker.add_argument("--skip-export", action="store_true")
    worker.add_argument("--sharding", choices=("ddp", "fsdp"), default="ddp")

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=4)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--batch-size", type=int, default=PREFLIGHT_BATCH_SIZE)
    pair.add_argument("--representax-gpus", default="0,1")
    pair.add_argument("--representax-sharding", choices=("ddp", "fsdp"), default="ddp")
    pair.add_argument("--reference-gpu", type=int, default=1)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    training_audios=arguments.training_audios,
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
    "AudioTextEvaluationCollator",
    "AudioTextRetrievalCollator",
    "FrozenContract",
    "frozen_contract",
    "prepare_data",
]
