#!/usr/bin/env python3
"""Matched raw-JSONL-to-reload benchmark against Sentence Transformers 5.6.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import types
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

DATASET_ID = "sentence-transformers/stsb"
DATASET_REVISION = "ab7a5ac0e35aa22088bdcf23e7fd99b220e53308"
ORACLE_VERSION = "5.6.1"
TRANSFORMERS_VERSION = "5.3.0"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str
    checkpoint: Path | None
    maximum_length: int
    batch_size: int
    steps: int
    parameter_dtype: str = "float32"
    compute_dtype: str = "float32"
    trust_remote_code: bool = False


MODEL_SPECS = {
    "minilm": ModelSpec(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        checkpoint=None,
        maximum_length=128,
        batch_size=32,
        steps=100,
    ),
    "jina-small": ModelSpec(
        name="jina-small",
        model_id="jinaai/jina-embeddings-v5-omni-small-retrieval",
        revision="12949877f0092093f366c6450340011320152a05",
        checkpoint=None,
        maximum_length=128,
        batch_size=4,
        steps=40,
        parameter_dtype="bfloat16",
        compute_dtype="bfloat16",
        trust_remote_code=True,
    ),
}


def _model_spec(arguments: argparse.Namespace) -> ModelSpec:
    checkpoint = arguments.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    return replace(MODEL_SPECS[arguments.model], checkpoint=checkpoint)


def _checkpoint(spec: ModelSpec) -> Path:
    if spec.checkpoint is None:  # pragma: no cover - constructed by _model_spec
        raise AssertionError("benchmark model spec requires a local checkpoint")
    return spec.checkpoint


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _hash_arrays(*arrays: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()
    for array in arrays:
        value = np.asarray(array, dtype=np.float32)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes(order="C"))
    return "sha256:" + digest.hexdigest()


def _save_probe(path: Path, left: Any, right: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        left=np.asarray(left, dtype=np.float32),
        right=np.asarray(right, dtype=np.float32),
    )


def _compare_probes(
    reference: Path,
    reloaded: Path,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, float | bool]:
    import numpy as np

    with np.load(reference) as expected, np.load(reloaded) as actual:
        differences = np.concatenate(
            [
                np.abs(expected[name] - actual[name]).reshape(-1)
                for name in ("left", "right")
            ]
        )
        reference_values = np.concatenate(
            [expected[name].reshape(-1) for name in ("left", "right")]
        )
        actual_values = np.concatenate(
            [actual[name].reshape(-1) for name in ("left", "right")]
        )
    denominator = np.linalg.norm(reference_values) * np.linalg.norm(actual_values)
    cosine = float(reference_values @ actual_values / denominator)
    return {
        "allclose": bool(
            np.allclose(
                reference_values,
                actual_values,
                rtol=relative_tolerance,
                atol=absolute_tolerance,
            )
        ),
        "max_absolute_difference": float(differences.max(initial=0.0)),
        "mean_absolute_difference": float(differences.mean()),
        "cosine_similarity": cosine,
    }


def _maximum_host_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _timestamp_seconds(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _representax_report(
    spec: ModelSpec,
    *,
    data_directory: Path,
    run_directory: Path,
    probe_path: Path,
) -> dict[str, Any]:
    checkpoint = _checkpoint(spec)
    import jax

    import representax
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        DataConfig,
        EmbeddingSimilarityEvaluatorConfig,
        EvaluationConfig,
        ExportConfig,
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        TrainingConfig,
    )
    from representax.core import Encoder, Route, encode
    from representax.data import identity, mix, source
    from representax.integrations import SentencePairCollator
    from representax.tasks.pairwise import CosineRegressionConfig, PairwiseConfig
    from representax.train import run_job

    model_target = (
        "representax.integrations.load_sentence_transformer_encoder"
        if spec.name == "minilm"
        else "representax.integrations.load_jina_v5_small_text_encoder"
    )
    train_path = data_directory / "train.jsonl"
    validation_path = data_directory / "validation.jsonl"
    collator_target = "representax.integrations.SentencePairCollator"
    collator_parameters = {
        "checkpoint": str(checkpoint),
        "maximum_length": spec.maximum_length,
    }

    def data(path: Path, *, evaluation: bool = False) -> DataConfig:
        parameters = dict(collator_parameters)
        if evaluation:
            parameters["pad_to_size"] = spec.batch_size
        return DataConfig(
            recipe=mix(source(str(path), map=identity), shuffle=False),
            collate=ComponentConfig(
                target=collator_target,
                parameters=parameters,
            ),
            drop_remainder=not evaluation,
            num_threads=0,
            prefetch_buffer_size=2,
        )

    job = JobConfig(
        name=f"dense-e2e-{spec.name}",
        model=ModelConfig(
            target=model_target,
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": spec.revision,
                "local_files_only": True,
                "parameter_dtype": spec.parameter_dtype,
                "compute_dtype": spec.compute_dtype,
            },
        ),
        task=PairwiseConfig(),
        loss=CosineRegressionConfig(),
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
        data=data(train_path),
        training=TrainingConfig(
            global_batch_size=spec.batch_size,
            max_steps=spec.steps,
            seed=42,
            batch=BatchConfig(micro_batch_size=spec.batch_size),
            activation_rematerialization=(
                "full" if spec.name == "jina-small" else "none"
            ),
            donate_buffers=True,
        ),
        checkpointing=CheckpointConfig(
            every=spec.steps,
            keep=1,
            save_final=True,
            asynchronous=True,
        ),
        logging=LoggingConfig(console_every=spec.steps),
        evaluation=EvaluationConfig(
            data=data(validation_path, evaluation=True),
            batch_size=spec.batch_size,
            evaluators=(
                EmbeddingSimilarityEvaluatorConfig(
                    name="sts",
                    similarity_functions=("cosine",),
                    main_similarity="cosine",
                ),
            ),
            on_start=True,
            on_end=True,
            primary_metric="valid/sts/spearman_cosine",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(selection="final"),
    )

    job_started = time.perf_counter()
    job_wall_started = time.time()
    result = run_job(job, run_directory)
    jax.block_until_ready(result.state)
    job_seconds = time.perf_counter() - job_started

    metrics = _jsonl(run_directory / "metrics.jsonl")
    evaluations = [row for row in metrics if row["event"] == "evaluation"]
    training_rows = [row for row in metrics if row["event"] == "training_step"]
    events = _jsonl(run_directory / "events.jsonl")
    started = next(row for row in events if row["event"] == "training_started")
    finished = next(row for row in events if row["event"] == "training_finished")
    first_uses = [
        row for row in events if row["event"] == "executable_first_use_finished"
    ]

    records = _jsonl(validation_path)[: spec.batch_size]
    collator = SentencePairCollator(
        checkpoint,
        maximum_length=spec.maximum_length,
    )
    batch = collator(records)
    model = result.state.model
    if not isinstance(model, Encoder):
        raise TypeError("trained model does not satisfy the Encoder protocol")
    left = encode(model, batch.left, route=Route.GENERIC)
    right = encode(model, batch.right, route=Route.GENERIC)
    jax.block_until_ready((left, right))
    _save_probe(probe_path, left, right)
    stats = jax.devices()[0].memory_stats() or {}

    return {
        "schema_version": "representax-dense-e2e-worker-v1",
        "framework": "representax",
        "framework_version": representax.__version__
        if hasattr(representax, "__version__")
        else "0.0.1",
        "jax_version": jax.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dataset_revision": DATASET_REVISION,
        "batch_size": spec.batch_size,
        "maximum_length": spec.maximum_length,
        "steps": spec.steps,
        "precision": spec.compute_dtype,
        "job_seconds": job_seconds,
        "runtime_construction_seconds": max(
            0.0, _timestamp_seconds(started["timestamp"]) - job_wall_started
        ),
        "training_lifecycle_seconds": (
            _timestamp_seconds(finished["timestamp"])
            - _timestamp_seconds(started["timestamp"])
        ),
        "compilation_and_first_execution_seconds": sum(
            float(row["duration_seconds"]) for row in first_uses
        ),
        "initial_evaluation": evaluations[0]["metrics"],
        "final_evaluation": evaluations[-1]["metrics"],
        "final_training": training_rows[-1]["metrics"],
        "validation_embedding_sha256": _hash_arrays(left, right),
        "inference_bundle": str(result.inference_bundle),
        "allocator_peak_device_bytes": int(stats.get("peak_bytes_in_use", 0)),
        "maximum_host_rss_bytes": _maximum_host_rss_bytes(),
    }


def _pad_features(
    features: dict[str, Any], width: int, pad_token_id: int
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    result = {}
    for name, value in features.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            result[name] = value
            continue
        value = value[:, :width]
        padding = width - value.shape[1]
        fill = pad_token_id if name.endswith("input_ids") else 0
        result[name] = functional.pad(value, (0, padding), value=fill)
    return result


def _sentence_transformers_report(
    spec: ModelSpec,
    *,
    data_directory: Path,
    run_directory: Path,
    probe_path: Path,
) -> dict[str, Any]:
    checkpoint = _checkpoint(spec)
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
    from sentence_transformers.sentence_transformer.evaluation import (
        EmbeddingSimilarityEvaluator,
    )
    from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss

    from benchmarks.samplers import sequential_sentence_transformers_batches

    if sentence_transformers.__version__ != ORACLE_VERSION:
        raise RuntimeError(f"expected sentence-transformers=={ORACLE_VERSION}")
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"expected transformers=={TRANSFORMERS_VERSION}")

    load_started = time.perf_counter()
    model_kwargs = {"local_files_only": True}
    if spec.trust_remote_code:
        model_kwargs["trust_remote_code"] = True
        model_kwargs["model_kwargs"] = {"modality": "text"}
    model = SentenceTransformer(str(checkpoint), **model_kwargs)
    if spec.name == "jina-small":
        # The released multimodal wrapper treats any HTTP-looking STS sentence as
        # remote media and attempts a network download. This benchmark is an
        # explicitly text-only contract, so keep every JSONL string on that path.
        custom_module = sys.modules[type(model[0]).__module__]
        custom_module._is_media_string = lambda _value: False
        custom_module._download_if_url = lambda value: value
        transformer = model[0]
        backbone = transformer.model
        backbone.config.modality = "text"
        backbone._modality = "text"
        for name in ("visual", "audio_tower", "audio_projector"):
            if hasattr(backbone, name):
                delattr(backbone, name)

        def trainable_text_forward(self, features, truncate_dim=None, **_kwargs):
            import torch.nn.functional as functional

            device = next(self.model.parameters()).device
            batch = {
                name: value.to(device)
                for name, value in features.items()
                if isinstance(value, torch.Tensor)
            }
            attention_mask = batch["attention_mask"]
            positions = attention_mask.long().cumsum(-1) - 1
            positions = positions.masked_fill(attention_mask == 0, 0)
            batch["position_ids"] = positions.unsqueeze(0).expand(3, -1, -1)
            hidden = self.model(**batch).last_hidden_state
            last = attention_mask.sum(dim=1) - 1
            pooled = hidden[
                torch.arange(hidden.shape[0], device=hidden.device),
                last,
            ]
            if truncate_dim is not None:
                pooled = pooled[..., :truncate_dim]
            features["sentence_embedding"] = functional.normalize(
                pooled,
                p=2,
                dim=-1,
            ).float()
            return features

        transformer.forward = types.MethodType(trainable_text_forward, transformer)
        torch.cuda.empty_cache()
    model.max_seq_length = spec.maximum_length
    train_dataset = datasets.Dataset.from_json(
        str(data_directory / "train.jsonl")
    ).remove_columns(["id"])
    validation = _jsonl(data_directory / "validation.jsonl")
    model_load_and_data_seconds = time.perf_counter() - load_started

    class FixedLengthCollator(SentenceTransformerDataCollator):
        def __call__(self, features):
            batch = super().__call__(features)
            return _pad_features(
                batch,
                spec.maximum_length,
                model.tokenizer.pad_token_id,
            )

    evaluator = EmbeddingSimilarityEvaluator(
        [row["sentence1"] for row in validation],
        [row["sentence2"] for row in validation],
        [float(row["score"]) for row in validation],
        batch_size=spec.batch_size,
        main_similarity="cosine",
        similarity_fn_names=["cosine"],
        name="sts",
        show_progress_bar=False,
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=spec.batch_size,
        max_steps=spec.steps,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=spec.compute_dtype == "bfloat16",
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        save_strategy="steps",
        save_steps=spec.steps,
        save_total_limit=1,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        batch_sampler=sequential_sentence_transformers_batches,
        seed=42,
        data_seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=CosineSimilarityLoss(model),
        evaluator=evaluator,
        data_collator=FixedLengthCollator(
            preprocess_fn=model.tokenize,
            valid_label_columns=trainer_label_columns(),
        ),
    )

    initial_started = time.perf_counter()
    initial_evaluation = evaluator(model)
    initial_evaluation_seconds = time.perf_counter() - initial_started
    torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    train_output = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    final_started = time.perf_counter()
    final_evaluation = evaluator(model)
    torch.cuda.synchronize()
    final_evaluation_seconds = time.perf_counter() - final_started
    export_directory = run_directory / "final-model"
    export_started = time.perf_counter()
    trainer.save_model(str(export_directory))
    torch.cuda.synchronize()
    export_seconds = time.perf_counter() - export_started

    probe = validation[: spec.batch_size]
    left = model.encode(
        [row["sentence1"] for row in probe],
        batch_size=spec.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    right = model.encode(
        [row["sentence2"] for row in probe],
        batch_size=spec.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    _save_probe(probe_path, left, right)
    return {
        "schema_version": "representax-dense-e2e-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dataset_revision": DATASET_REVISION,
        "batch_size": spec.batch_size,
        "maximum_length": spec.maximum_length,
        "steps": spec.steps,
        "precision": spec.compute_dtype,
        "oracle_adapter": (
            "released"
            if spec.name != "jina-small"
            else "text-only autograd-preserving forward"
        ),
        "model_load_and_data_seconds": model_load_and_data_seconds,
        "initial_evaluation_seconds": initial_evaluation_seconds,
        "training_seconds": training_seconds,
        "final_evaluation_seconds": final_evaluation_seconds,
        "export_seconds": export_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "final_training_loss": float(train_output.training_loss),
        "validation_embedding_sha256": _hash_arrays(left, right),
        "inference_bundle": str(export_directory),
        "allocator_peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "allocator_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "maximum_host_rss_bytes": _maximum_host_rss_bytes(),
    }


def trainer_label_columns() -> list[str]:
    return ["label", "score"]


def _reload_report(
    framework: str,
    spec: ModelSpec,
    *,
    data_directory: Path,
    run_directory: Path,
    probe_path: Path,
) -> dict[str, Any]:
    checkpoint = _checkpoint(spec)
    import numpy as np

    rows = _jsonl(data_directory / "validation.jsonl")
    started = time.perf_counter()
    if framework == "representax":
        import jax

        from representax import load_inference_bundle
        from representax.core import Encoder, Route, encode
        from representax.integrations import SentencePairCollator

        model, _ = load_inference_bundle(run_directory / "final-model")
        if not isinstance(model, Encoder):
            raise TypeError("reloaded model does not satisfy the Encoder protocol")
        collator = SentencePairCollator(
            checkpoint,
            maximum_length=spec.maximum_length,
        )
        batch = collator(rows[: spec.batch_size])
        left = encode(model, batch.left, route=Route.GENERIC)
        right = encode(model, batch.right, route=Route.GENERIC)
        jax.block_until_ready((left, right))
        allocator = int(
            (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        )
    else:
        import sentence_transformers
        import torch

        model = sentence_transformers.SentenceTransformer(
            str(run_directory / "final-model"),
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
            model_kwargs=({"modality": "text"} if spec.trust_remote_code else None),
        )
        left = model.encode(
            [row["sentence1"] for row in rows[: spec.batch_size]],
            batch_size=spec.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        right = model.encode(
            [row["sentence2"] for row in rows[: spec.batch_size]],
            batch_size=spec.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        torch.cuda.synchronize()
        allocator = int(torch.cuda.max_memory_allocated())
    _save_probe(probe_path, left, right)
    reload_seconds = time.perf_counter() - started
    return {
        "schema_version": "representax-dense-e2e-reload-v1",
        "framework": framework,
        "model": spec.name,
        "reload_and_probe_seconds": reload_seconds,
        "validation_embedding_sha256": _hash_arrays(
            np.asarray(left), np.asarray(right)
        ),
        "allocator_peak_device_bytes": allocator,
        "maximum_host_rss_bytes": _maximum_host_rss_bytes(),
    }


def _prepare_data(directory: Path) -> None:
    import datasets

    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    counts = {}
    for split in ("train", "validation"):
        dataset = datasets.load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split=split,
            streaming=False,
        )
        target = directory / f"{split}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for index, row in enumerate(dataset):
                record = {
                    "id": f"{split}:{index:06d}",
                    "sentence1": str(row["sentence1"]),
                    "sentence2": str(row["sentence2"]),
                    "score": float(row["score"]),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, target)
        files[split] = {"path": target.name, "sha256": _sha256(target)}
        counts[split] = len(dataset)
    _atomic_json(
        directory / "manifest.json",
        {
            "schema_version": "representax-dense-e2e-data-v1",
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "files": files,
            "counts": counts,
        },
    )


def _worker(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
    report = (
        _representax_report(
            spec,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            probe_path=arguments.report.with_suffix(".probe.npz"),
        )
        if arguments.framework == "representax"
        else _sentence_transformers_report(
            spec,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            probe_path=arguments.report.with_suffix(".probe.npz"),
        )
    )
    _atomic_json(arguments.report, report)


def _reload_worker(arguments: argparse.Namespace) -> None:
    report = _reload_report(
        arguments.framework,
        _model_spec(arguments),
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        probe_path=arguments.report.with_suffix(".probe.npz"),
    )
    _atomic_json(arguments.report, report)


def _device_memory_mib(index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def _run_processes(commands: list[tuple[str, int, list[str], Path]]) -> dict[str, Any]:
    running = {}
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
                "JAX_COMPILATION_CACHE_DIR": str(
                    log_path.parent.parent
                    / (
                        "jax-cache-reload"
                        if "reload" in log_path.stem
                        else "jax-cache-train"
                    )
                ),
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
        running[name] = {
            "process": process,
            "stream": stream,
            "started": started,
            "gpu": gpu,
            "peak_device_mib": 0,
            "finished": None,
            "log": str(log_path),
        }
    while any(item["finished"] is None for item in running.values()):
        for item in running.values():
            if item["finished"] is not None:
                continue
            if item["process"].poll() is None:
                item["peak_device_mib"] = max(
                    item["peak_device_mib"], _device_memory_mib(item["gpu"])
                )
            else:
                item["finished"] = time.perf_counter()
        time.sleep(0.2)
    reports = {}
    for name, item in running.items():
        item["stream"].close()
        return_code = item["process"].returncode
        reports[name] = {
            "return_code": return_code,
            "process_wall_seconds": item["finished"] - item["started"],
            "process_peak_device_bytes": item["peak_device_mib"] * 1024 * 1024,
            "log": item["log"],
        }
        if return_code:
            raise RuntimeError(
                f"{name} failed with exit code {return_code}; see {item['log']}"
            )
    return reports


def _pair(arguments: argparse.Namespace) -> None:
    spec = _model_spec(arguments)
    checkpoint = _checkpoint(spec)
    result_directory = arguments.result_directory.resolve()
    result_directory.mkdir(parents=True, exist_ok=False)
    train_reports = {
        name: result_directory / f"{name}.json"
        for name in ("representax", "sentence-transformers")
    }
    run_directories = {
        name: result_directory / "runs" / name
        for name in ("representax", "sentence-transformers")
    }
    commands = []
    for name, gpu in (
        ("representax", arguments.representax_gpu),
        ("sentence-transformers", arguments.sentence_transformers_gpu),
    ):
        commands.append(
            (
                name,
                gpu,
                [
                    sys.executable,
                    "-m",
                    "benchmarks.dense_end_to_end",
                    "worker",
                    "--framework",
                    name,
                    "--model",
                    spec.name,
                    "--checkpoint",
                    str(checkpoint),
                    "--data-directory",
                    str(arguments.data_directory),
                    "--run-directory",
                    str(run_directories[name]),
                    "--report",
                    str(train_reports[name]),
                ],
                result_directory / "logs" / f"{name}.log",
            )
        )
    process_reports = _run_processes(commands)

    reload_reports = {
        name: result_directory / f"{name}-reload.json"
        for name in ("representax", "sentence-transformers")
    }
    reload_commands = []
    for name, gpu in (
        ("representax", arguments.representax_gpu),
        ("sentence-transformers", arguments.sentence_transformers_gpu),
    ):
        reload_commands.append(
            (
                name,
                gpu,
                [
                    sys.executable,
                    "-m",
                    "benchmarks.dense_end_to_end",
                    "reload",
                    "--framework",
                    name,
                    "--model",
                    spec.name,
                    "--checkpoint",
                    str(checkpoint),
                    "--data-directory",
                    str(arguments.data_directory),
                    "--run-directory",
                    str(run_directories[name]),
                    "--report",
                    str(reload_reports[name]),
                ],
                result_directory / "logs" / f"{name}-reload.log",
            )
        )
    reload_process_reports = _run_processes(reload_commands)
    native = json.loads(train_reports["representax"].read_text())
    upstream = json.loads(train_reports["sentence-transformers"].read_text())
    native_reload = json.loads(reload_reports["representax"].read_text())
    upstream_reload = json.loads(reload_reports["sentence-transformers"].read_text())
    native_reload_equivalence = _compare_probes(
        train_reports["representax"].with_suffix(".probe.npz"),
        reload_reports["representax"].with_suffix(".probe.npz"),
        relative_tolerance=(2e-2 if spec.compute_dtype == "bfloat16" else 1e-5),
        absolute_tolerance=(2e-3 if spec.compute_dtype == "bfloat16" else 1e-6),
    )
    upstream_reload_equivalence = _compare_probes(
        train_reports["sentence-transformers"].with_suffix(".probe.npz"),
        reload_reports["sentence-transformers"].with_suffix(".probe.npz"),
        relative_tolerance=(2e-2 if spec.compute_dtype == "bfloat16" else 1e-5),
        absolute_tolerance=(2e-3 if spec.compute_dtype == "bfloat16" else 1e-6),
    )
    if not native_reload_equivalence["allclose"]:
        raise RuntimeError(
            "Representax fresh reload changed validation embeddings beyond tolerance: "
            f"{native_reload_equivalence}"
        )
    if not upstream_reload_equivalence["allclose"]:
        raise RuntimeError(
            "Sentence Transformers fresh reload changed validation embeddings beyond "
            f"tolerance: {upstream_reload_equivalence}"
        )
    total_native = (
        process_reports["representax"]["process_wall_seconds"]
        + reload_process_reports["representax"]["process_wall_seconds"]
    )
    total_upstream = (
        process_reports["sentence-transformers"]["process_wall_seconds"]
        + reload_process_reports["sentence-transformers"]["process_wall_seconds"]
    )
    summary = {
        "schema_version": "representax-dense-e2e-comparison-v1",
        "contract": {
            "model": spec.name,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "checkpoint_model_sha256": _sha256(checkpoint / "model.safetensors"),
            "dataset": json.loads(
                (arguments.data_directory / "manifest.json").read_text()
            ),
            "batch_size": spec.batch_size,
            "maximum_length": spec.maximum_length,
            "steps": spec.steps,
            "learning_rate": 2e-5,
            "precision": spec.compute_dtype,
        },
        "representax": {
            **native,
            **process_reports["representax"],
            "reload": {
                **native_reload,
                **reload_process_reports["representax"],
                "equivalence": native_reload_equivalence,
            },
            "disk_to_verified_reload_seconds": total_native,
        },
        "sentence_transformers": {
            **upstream,
            **process_reports["sentence-transformers"],
            "reload": {
                **upstream_reload,
                **reload_process_reports["sentence-transformers"],
                "equivalence": upstream_reload_equivalence,
            },
            "disk_to_verified_reload_seconds": total_upstream,
        },
        "comparison": {
            "end_to_end_speedup": total_upstream / total_native,
            "representax_seconds_saved": total_upstream - total_native,
            "process_peak_device_memory_ratio": (
                process_reports["representax"]["process_peak_device_bytes"]
                / process_reports["sentence-transformers"]["process_peak_device_bytes"]
            ),
        },
    }
    _atomic_json(result_directory / "summary.json", summary)
    print(json.dumps(summary["comparison"], indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data-directory", type=Path, required=True)

    for command in ("worker", "reload"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--framework",
            choices=("representax", "sentence-transformers"),
            required=True,
        )
        child.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
        child.add_argument("--checkpoint", type=Path, required=True)
        child.add_argument("--data-directory", type=Path, required=True)
        child.add_argument("--run-directory", type=Path, required=True)
        child.add_argument("--report", type=Path, required=True)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--result-directory", type=Path, required=True)
    pair.add_argument("--representax-gpu", type=int, default=4)
    pair.add_argument("--sentence-transformers-gpu", type=int, default=5)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        _prepare_data(arguments.data_directory)
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "reload":
        _reload_worker(arguments)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()
