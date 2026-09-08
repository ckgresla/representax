#!/usr/bin/env python3
"""Run paired BERT parameter-scaling experiments on a fixed length schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

DEFAULT_ARTIFACT_ROOT = (
    Path(os.environ.get("REPRESENTAX_PAPER_ROOT", "/raid/representax-paper"))
    / "07-bert-parameter-scaling"
)
DEFAULT_SOURCE_DATA = Path("/raid/representax/data/dense-retrieval-msmarco-v1")
DEFAULT_TOKENIZER = Path(
    "/home/ckg/representax-artifacts/"
    "bert-scaling-ladder-msmarco-20260831-v3/inputs/tokenizer"
)
SIZES = ("bert-30m", "bert-110m", "bert-500m", "bert-1b", "bert-4b")
STEPS = 1_000
LONG_SEQUENCE_START = 901
SHORT_SEQUENCE_LENGTH = 128
LONG_SEQUENCE_LENGTH = 512
GLOBAL_BATCH_SIZE = 32
SEED = 7
LEARNING_RATE = 2e-5
GRAD_CACHE_CHUNKS = {
    "bert-30m": 32,
    "bert-110m": 16,
    "bert-500m": 4,
    "bert-1b": 2,
    "bert-4b": 1,
}
EVALUATION_BATCH_SIZES = {
    "bert-30m": 128,
    "bert-110m": 64,
    "bert-500m": 16,
    "bert-1b": 4,
    "bert-4b": 1,
}
Topology = Literal["single", "ddp", "fsdp"]
Rematerialization = Literal["none", "selective", "full"]


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
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def sequence_length_for_step(step: int, *, long_sequence_start: int) -> int:
    if step <= 0:
        raise ValueError("step must be positive")
    return SHORT_SEQUENCE_LENGTH if step < long_sequence_start else LONG_SEQUENCE_LENGTH


def _validate_schedule(*, steps: int, long_sequence_start: int) -> None:
    if steps < 3:
        raise ValueError("at least three optimizer steps are required")
    if not 2 <= long_sequence_start <= steps:
        raise ValueError("long_sequence_start must be between step 2 and max_steps")


def _source_checkpoint(artifact_root: Path, size: str) -> Path:
    return artifact_root / "checkpoints" / size


def _model_files(checkpoint: Path) -> tuple[Path, ...]:
    return tuple(sorted(checkpoint.glob("model*.safetensors")))


def prepare_checkpoint(
    artifact_root: Path,
    size: str,
    *,
    tokenizer: Path = DEFAULT_TOKENIZER,
) -> Path:
    """Create one frozen Sentence Transformers artifact from a random BERT init."""

    from experiments.preflights.bert_scaling import ladder_entry

    checkpoint = _source_checkpoint(artifact_root, size)
    manifest = checkpoint / "initialization.json"
    if manifest.is_file():
        return checkpoint
    if checkpoint.exists():
        raise FileExistsError(f"incomplete checkpoint already exists: {checkpoint}")
    if not tokenizer.is_dir():
        raise FileNotFoundError(f"pinned BERT tokenizer does not exist: {tokenizer}")

    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from transformers import BertConfig, BertModel

    entry = ladder_entry(size)
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    source = temporary / "transformers-source"
    source.mkdir(parents=True)
    for path in tokenizer.iterdir():
        if path.is_file():
            shutil.copy2(path, source / path.name)
    torch.manual_seed(SEED)
    config = BertConfig(
        vocab_size=entry.vocab_size,
        hidden_size=entry.hidden_size,
        num_hidden_layers=entry.num_hidden_layers,
        num_attention_heads=entry.num_attention_heads,
        intermediate_size=entry.intermediate_size,
        hidden_act=entry.hidden_activation,
        hidden_dropout_prob=entry.hidden_dropout_probability,
        attention_probs_dropout_prob=entry.attention_dropout_probability,
        max_position_embeddings=entry.max_position_embeddings,
        type_vocab_size=entry.type_vocab_size,
        initializer_range=entry.initializer_range,
        layer_norm_eps=entry.norm_epsilon,
        pad_token_id=entry.pad_token_id,
    )
    model = BertModel(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != entry.expected_parameters:
        raise RuntimeError(
            f"{size} initialized {parameter_count:,} parameters; "
            f"expected {entry.expected_parameters:,}"
        )
    model.save_pretrained(source, safe_serialization=True, max_shard_size="5GB")
    del model

    transformer = Transformer(str(source), max_seq_length=LONG_SEQUENCE_LENGTH)
    pooling = Pooling(
        entry.hidden_size,
        pooling_mode="mean",
        include_prompt=True,
    )
    sentence_model = SentenceTransformer(
        modules=[transformer, pooling],
        similarity_fn_name="cosine",
        device="cpu",
    )
    sentence_model.save(str(temporary), safe_serialization=True)
    del sentence_model, transformer, pooling
    shutil.rmtree(source)

    model_files = _model_files(temporary)
    if not model_files:
        raise RuntimeError("prepared BERT artifact contains no safetensor weights")
    _write_json(
        temporary / "initialization.json",
        {
            "size": size,
            "seed": SEED,
            "parameters": parameter_count,
            "architecture": entry.config_values(),
            "tokenizer": str(tokenizer.resolve()),
            "weight_files": {path.name: _sha256(path) for path in model_files},
        },
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, checkpoint)
    return checkpoint


def materialize_training_data(
    source_data: Path,
    output: Path,
    *,
    steps: int,
    long_sequence_start: int,
    batch_size: int,
) -> Path:
    """Write the exact ordered rows and static shape assigned to every update."""

    from benchmarks.dense_retrieval import (
        TRAIN_DATASET_FILE,
        TRAIN_DATASET_ID,
        _artifact_path,
        _training_table,
    )

    _validate_schedule(steps=steps, long_sequence_start=long_sequence_start)
    required = steps * batch_size
    table = _training_table(
        _artifact_path(source_data, TRAIN_DATASET_ID, TRAIN_DATASET_FILE),
        required,
    )
    rows = []
    for index, row in enumerate(table.to_pylist()):
        step = index // batch_size + 1
        rows.append(
            {
                "query": str(row["query"]),
                "positive": str(row["positive"]),
                "sequence_length": sequence_length_for_step(
                    step,
                    long_sequence_start=long_sequence_start,
                ),
            }
        )
    _write_jsonl(output, rows)
    _write_json(
        output.with_suffix(".manifest.json"),
        {
            "source_manifest": str((source_data / "manifest.json").resolve()),
            "source_manifest_sha256": _sha256(source_data / "manifest.json"),
            "steps": steps,
            "global_batch_size": batch_size,
            "rows": required,
            "short_sequence": {
                "length": SHORT_SEQUENCE_LENGTH,
                "steps": long_sequence_start - 1,
            },
            "long_sequence": {
                "length": LONG_SEQUENCE_LENGTH,
                "steps": steps - long_sequence_start + 1,
            },
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        },
    )
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ScheduledSentenceTransformersCollator:
    """Apply the same explicit 128/512 batch schedule in the reference worker."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.valid_label_columns: list[str] = []

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            return {}
        lengths = {int(row["sequence_length"]) for row in features}
        if len(lengths) != 1:
            raise ValueError("one reference batch cannot cross a length phase")
        length = lengths.pop()
        if length not in {SHORT_SEQUENCE_LENGTH, LONG_SEQUENCE_LENGTH}:
            raise ValueError(f"unsupported scheduled sequence length {length}")
        batch: dict[str, Any] = {}
        processing = {
            "text": {
                "padding": "max_length",
                "truncation": True,
                "max_length": length,
            }
        }
        for column in ("query", "positive"):
            encoded = self.model.preprocess(
                [str(row[column]) for row in features],
                processing_kwargs=processing,
            )
            for name, value in encoded.items():
                if isinstance(value, torch.Tensor) or name == "modality":
                    batch[f"{column}_{name}"] = value
        return batch


def _native_job(
    *,
    size: str,
    checkpoint: Path,
    training_data: Path,
    steps: int,
    long_sequence_start: int,
    batch_size: int,
    chunk_size: int,
    topology: Topology,
    world_size: int,
    rematerialization: Rematerialization,
    lifecycle: bool,
) -> Any:
    from representax.config import (
        BatchConfig,
        CheckpointConfig,
        ComponentConfig,
        CustomShardingConfig,
        DataConfig,
        DDPConfig,
        ExportConfig,
        FSDPConfig,
        GradCacheConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PartitionRuleConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig

    if topology == "single":
        mesh = MeshConfig()
        sharding = DDPConfig()
        local_batch_size = batch_size
    else:
        mesh = MeshConfig(axis_shapes=(world_size,), axis_names=("data",))
        if topology == "ddp":
            sharding = DDPConfig(axis="data")
        elif size == "bert-4b":
            # The depth-stacked matrices are input-major inside lax.scan. Shard
            # their contracted axis so each layer reduces its output instead of
            # materializing a complete multi-GiB stacked matrix on every device.
            sharding = CustomShardingConfig(
                data_axis="data",
                parameter_axes=("data",),
                parameter_rules=(
                    PartitionRuleConfig(
                        pattern=r"\.tower\.embeddings\.(?:word|position)$",
                        axes=(None, "data"),
                    ),
                    PartitionRuleConfig(
                        pattern=(
                            r"\.tower\.layers\.blocks\."
                            r"(?:attention\.(?:qkv|output)|mlp\.(?:input|output))"
                            r"\.weight$"
                        ),
                        axes=(None, "data", None),
                    ),
                ),
            )
        else:
            sharding = FSDPConfig(data_axis="data", parameter_axis="data")
        local_batch_size = batch_size // world_size
    if batch_size % world_size:
        raise ValueError("global batch size must divide evenly across data replicas")
    return JobConfig(
        name=f"paper-bert-parameter-scaling-{size}-{topology}",
        model=ModelConfig(
            target="representax.models:SentenceEncoder.load_from_hf",
            parameters={
                "model_name_or_path": str(checkpoint),
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": [
                    SHORT_SEQUENCE_LENGTH,
                    LONG_SEQUENCE_LENGTH,
                ],
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
                    "peak_value": LEARNING_RATE,
                    "warmup_steps": max(1, round(steps * 0.06)),
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(source(str(training_data), map=identity), shuffle=False),
            collate=ComponentConfig(
                target=(
                    "experiments.preflights.bert_scaling:ScheduledRetrievalCollator"
                ),
                parameters={
                    "admitted_lengths": [
                        SHORT_SEQUENCE_LENGTH,
                        LONG_SEQUENCE_LENGTH,
                    ]
                },
            ),
            drop_remainder=True,
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=batch_size,
            max_steps=steps,
            seed=SEED,
            mesh=mesh,
            sharding=sharding,
            batch=BatchConfig(micro_batch_size=local_batch_size),
            grad_cache=GradCacheConfig(micro_batch_size=chunk_size),
            activation_rematerialization=rematerialization,
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=(
            CheckpointConfig(
                every=long_sequence_start - 1,
                keep=1,
                save_final=False,
                asynchronous=False,
            )
            if lifecycle
            else None
        ),
        logging=LoggingConfig(console_every=1, timing=True, accelerator=True),
        export=ExportConfig(enabled=lifecycle, selection="final"),
    )


def _phase_summary(
    rows: Sequence[tuple[int, float]],
    *,
    batch_size: int,
    long_sequence_start: int,
) -> dict[str, Any]:
    phases = {
        "sequence-128": (1, long_sequence_start - 1),
        "sequence-512": (long_sequence_start, max(step for step, _ in rows)),
    }
    result = {}
    for name, (start, end) in phases.items():
        selected = [
            duration
            for step, duration in rows
            if start <= step <= end and step != start
        ]
        if not selected:
            raise ValueError(f"{name} has no warmed optimizer-step timings")
        result[name] = {
            "first_step": start,
            "last_step": end,
            "warmed_steps": len(selected),
            "median_step_seconds": statistics.median(selected),
            "examples_per_second": batch_size * len(selected) / sum(selected),
        }
    return result


def _accelerator_summary(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, dict[str, float]]:
    measurements: dict[str, list[float]] = {}
    for row in rows:
        for name, value in row.items():
            if name.startswith("accelerator/"):
                measurements.setdefault(name, []).append(float(value))
    return {
        name: {
            "mean": statistics.fmean(values),
            "maximum": max(values),
        }
        for name, values in sorted(measurements.items())
    }


def _native_worker(arguments: argparse.Namespace) -> dict[str, Any] | None:
    import jax

    from representax.train import run_job

    job = _native_job(
        size=arguments.size,
        checkpoint=arguments.checkpoint,
        training_data=arguments.training_data,
        steps=arguments.steps,
        long_sequence_start=arguments.long_sequence_start,
        batch_size=arguments.batch_size,
        chunk_size=arguments.chunk_size,
        topology=arguments.topology,
        world_size=arguments.world_size,
        rematerialization=arguments.rematerialization,
        lifecycle=arguments.lifecycle,
    )
    started = time.perf_counter()
    if arguments.native_stage == "initial":
        paused = run_job(
            job,
            arguments.run_directory,
            stop_after=arguments.long_sequence_start - 1,
        )
        if paused.completed_iterations != arguments.long_sequence_start - 1:
            raise RuntimeError("native run did not stop at the length transition")
        jax.block_until_ready(paused.state)
        return None
    if arguments.native_stage == "resume":
        completed = run_job(job, arguments.run_directory, resume=True)
    else:
        completed = run_job(job, arguments.run_directory)
    jax.block_until_ready(completed.state)
    wall_seconds = time.perf_counter() - started
    rows = []
    compile_rows = []
    losses = []
    accelerator_rows = []
    with (arguments.run_directory / "metrics.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "accelerator":
                accelerator_rows.append(row["metrics"])
                continue
            if row.get("event") != "training_step":
                continue
            step = int(row["iteration"])
            metrics = row["metrics"]
            losses.append((step, float(metrics["train/loss"])))
            if "perf/compilation_and_first_step_seconds" in metrics:
                compile_rows.append(
                    (step, float(metrics["perf/compilation_and_first_step_seconds"]))
                )
            elif "perf/step_seconds" in metrics:
                rows.append((step, float(metrics["perf/step_seconds"])))
    memory = [device.memory_stats() or {} for device in jax.devices("gpu")]
    return {
        "framework": "representax",
        "size": arguments.size,
        "topology": arguments.topology,
        "parameter_layout": (
            "contracted-input"
            if arguments.size == "bert-4b" and arguments.topology == "fsdp"
            else "default"
        ),
        "world_size": arguments.world_size,
        "steps": arguments.steps,
        "global_batch_size": arguments.batch_size,
        "grad_cache_chunk_size": arguments.chunk_size,
        "rematerialization": arguments.rematerialization,
        "phase_timing": _phase_summary(
            rows,
            batch_size=arguments.batch_size,
            long_sequence_start=arguments.long_sequence_start,
        ),
        "compilation_and_first_step": [
            {"step": step, "seconds": seconds} for step, seconds in compile_rows
        ],
        "losses": [{"step": step, "loss": loss} for step, loss in losses],
        "wall_seconds": wall_seconds,
        "device_memory": memory,
        "accelerator_samples": len(accelerator_rows),
        "accelerator_summary": _accelerator_summary(accelerator_rows),
        "completed_iterations": completed.completed_iterations,
        "resumed": completed.resumed,
        "inference_bundle": (
            None
            if completed.inference_bundle is None
            else str(completed.inference_bundle)
        ),
    }


def _native_lifecycle_commands() -> tuple[list[str], list[str]]:
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    return (
        [*command, "--native-stage", "initial"],
        [*command, "--native-stage", "resume"],
    )


def _native_lifecycle_worker(arguments: argparse.Namespace) -> None:
    started = time.perf_counter()
    for command in _native_lifecycle_commands():
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    result = json.loads(arguments.report.read_text())
    result["wall_seconds"] = time.perf_counter() - started
    result["resume_process_boundary"] = True
    _write_json(arguments.report, result)


def _reference_worker(arguments: argparse.Namespace) -> dict[str, Any] | None:
    import datasets
    import torch
    from benchmarks.samplers import sequential_sentence_transformers_batches
    from experiments.preflights.timing import CudaStepTimer
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
    )

    from representax.train.accelerator import AcceleratorMonitor

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    process_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if process_world_size != arguments.world_size:
        raise ValueError("reference launch world size differs from its run contract")
    torch.cuda.set_device(local_rank)
    model = SentenceTransformer(
        str(arguments.checkpoint),
        device=f"cuda:{local_rank}",
        local_files_only=True,
    )
    model.max_seq_length = LONG_SEQUENCE_LENGTH
    loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=20.0,
        mini_batch_size=arguments.chunk_size,
        gather_across_devices=arguments.world_size > 1,
        show_progress_bar=False,
    )
    trainer_arguments: dict[str, Any] = {}
    if arguments.topology == "fsdp":
        fsdp_config = {
            "transformer_layer_cls_to_wrap": ["BertLayer"],
            "use_orig_params": True,
            "sync_module_states": True,
        }
        if arguments.rematerialization == "full":
            fsdp_config["activation_checkpointing"] = True
        trainer_arguments.update(
            fsdp="full_shard auto_wrap",
            fsdp_config=fsdp_config,
        )
    training_arguments = SentenceTransformerTrainingArguments(
        output_dir=str(arguments.run_directory / "checkpoints"),
        per_device_train_batch_size=arguments.batch_size // arguments.world_size,
        max_steps=arguments.steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(arguments.steps * 0.06)),
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        # Accelerate's legacy FSDP optimizer save can send ranks into different
        # collectives. Export final inference weights after the timed loop instead.
        save_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        batch_sampler=sequential_sentence_transformers_batches,
        # The serialized base model retains BERT's pooler, while the sentence
        # representation uses mean pooling and therefore bypasses those weights.
        ddp_find_unused_parameters=True,
        seed=SEED,
        data_seed=SEED,
        **trainer_arguments,
    )
    timer = CudaStepTimer(
        arguments.run_directory / "step-timings.jsonl" if rank == 0 else None
    )
    losses: list[float] = []
    loss_stream = (
        (arguments.run_directory / "losses.jsonl").open(
            "x", encoding="utf-8", buffering=1
        )
        if rank == 0
        else None
    )

    class RecordingTrainer(SentenceTransformerTrainer):
        def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
            value = super().compute_loss(*args, **kwargs)
            loss_value = value[0] if isinstance(value, tuple) else value
            recorded_loss = float(loss_value.detach())
            losses.append(recorded_loss)
            if loss_stream is not None:
                loss_stream.write(
                    json.dumps(
                        {"step": len(losses), "loss": recorded_loss},
                        sort_keys=True,
                    )
                    + "\n"
                )
            return value

    trainer = RecordingTrainer(
        model=model,
        args=training_arguments,
        train_dataset=datasets.Dataset.from_list(_read_jsonl(arguments.training_data)),
        loss=loss,
        data_collator=ScheduledSentenceTransformersCollator(model),
        callbacks=[timer.callback()],
    )
    torch.cuda.reset_peak_memory_stats()
    accelerator_rows: list[dict[str, Any]] = []
    accelerator_monitor = None
    if rank == 0:
        accelerator_monitor = AcceleratorMonitor(
            lambda metrics: accelerator_rows.append(
                {
                    "timestamp_unix_seconds": time.time(),
                    "metrics": dict(metrics),
                }
            )
        )
        accelerator_monitor.start()
    started = time.perf_counter()
    try:
        trainer.train()
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
    finally:
        try:
            if accelerator_monitor is not None:
                accelerator_monitor.close()
        finally:
            try:
                timer.close()
            finally:
                if loss_stream is not None:
                    loss_stream.close()
    local_memory = {
        "rank": rank,
        "allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    memories = [None] * arguments.world_size if rank == 0 else None
    if arguments.world_size > 1:
        torch.distributed.gather_object(local_memory, memories, dst=0)
        torch.distributed.barrier()
    else:
        memories = [local_memory]
    result = {
        "framework": "sentence-transformers",
        "size": arguments.size,
        "topology": arguments.topology,
        "world_size": arguments.world_size,
        "steps": arguments.steps,
        "global_batch_size": arguments.batch_size,
        "grad_cache_chunk_size": arguments.chunk_size,
        "activation_checkpointing": arguments.rematerialization == "full",
        "phase_timing": _phase_summary(
            timer.rows,
            batch_size=arguments.batch_size,
            long_sequence_start=arguments.long_sequence_start,
        ),
        "losses": [
            {"step": step, "loss": value}
            for step, value in enumerate(losses[: arguments.steps], start=1)
        ],
        "wall_seconds": wall_seconds,
        "device_memory": memories,
        "accelerator_samples": len(accelerator_rows),
        "accelerator_summary": _accelerator_summary(
            [row["metrics"] for row in accelerator_rows]
        ),
        "completed_iterations": int(trainer.state.global_step),
        "resumed": False,
        "checkpoint_policy": (
            "final-model-only" if arguments.lifecycle else "disabled"
        ),
        "export_status": "pending" if arguments.lifecycle else "disabled",
        "inference_bundle": None,
    }
    if rank == 0:
        _write_jsonl(arguments.run_directory / "accelerator.jsonl", accelerator_rows)
        _write_json(arguments.report, result)
    if arguments.world_size > 1:
        torch.distributed.barrier()
    if arguments.lifecycle:
        final_model = arguments.run_directory / "final-model"
        if arguments.topology == "fsdp":
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel,
                StateDictType,
            )

            config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FullyShardedDataParallel.state_dict_type(
                trainer.model_wrapped,
                StateDictType.FULL_STATE_DICT,
                config,
            ):
                state_dict = trainer.model_wrapped.state_dict()
            if rank == 0:
                export_model = SentenceTransformer(
                    str(arguments.checkpoint),
                    device="cpu",
                    local_files_only=True,
                )
                export_model.load_state_dict(state_dict, strict=True)
                export_model.save_pretrained(str(final_model))
            del state_dict
            torch.distributed.barrier()
        else:
            trainer.save_model(str(final_model))
        if rank == 0:
            result["export_status"] = "complete"
            result["inference_bundle"] = str(final_model)
            _write_json(arguments.report, result)
    if rank != 0:
        return None
    return result


def _worker(arguments: argparse.Namespace) -> None:
    arguments.run_directory = arguments.run_directory.resolve()
    arguments.report = arguments.report.resolve()
    if (
        arguments.framework == "representax"
        and arguments.lifecycle
        and arguments.native_stage is None
    ):
        _native_lifecycle_worker(arguments)
        return
    result = (
        _native_worker(arguments)
        if arguments.framework == "representax"
        else _reference_worker(arguments)
    )
    if result is not None:
        _write_json(arguments.report, result)
        print(json.dumps(result["phase_timing"], indent=2, sort_keys=True))


def _source_snapshot(output: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    patch = subprocess.run(
        ("git", "diff", "--binary", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    source = output / "source"
    source.mkdir(parents=True)
    (source / "working-tree.patch").write_bytes(patch)
    shutil.copy2(__file__, source / "run.py")
    return {
        "representax_commit": git("rev-parse", "HEAD").strip(),
        "working_tree_status": git("status", "--porcelain=v1").splitlines(),
        "working_tree_patch_sha256": _sha256(source / "working-tree.patch"),
    }


def _worker_command(
    arguments: argparse.Namespace,
    *,
    framework: str,
    gpus: Sequence[int],
    training_data: Path,
    output: Path,
) -> list[str]:
    chunk_size = (
        arguments.representax_chunk_size
        if framework == "representax"
        else arguments.reference_chunk_size
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--framework",
        framework,
        "--size",
        arguments.size,
        "--checkpoint",
        str(_source_checkpoint(arguments.artifact_root, arguments.size)),
        "--training-data",
        str(training_data),
        "--run-directory",
        str(output / framework),
        "--report",
        str(output / f"{framework}.json"),
        "--steps",
        str(arguments.steps),
        "--long-sequence-start",
        str(arguments.long_sequence_start),
        "--batch-size",
        str(arguments.batch_size),
        "--chunk-size",
        str(chunk_size),
        "--topology",
        arguments.topology,
        "--world-size",
        str(len(gpus)),
        "--rematerialization",
        arguments.rematerialization,
    ]
    if arguments.lifecycle:
        command.append("--lifecycle")
    if framework == "sentence-transformers" and len(gpus) > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={len(gpus)}",
            str(Path(__file__).resolve()),
            *command[2:],
        ]
    return command


def _pair_output(arguments: argparse.Namespace) -> Path:
    category = "runs" if arguments.lifecycle else "probes"
    world_size = (
        len(arguments.gpus) if arguments.sequential else len(arguments.gpus) // 2
    )
    topology = arguments.topology
    if arguments.topology != "single":
        topology = f"{topology}-{world_size}gpu"
    if arguments.sequential:
        topology = f"{topology}-sequential"
    if arguments.rematerialization != "none":
        topology = f"{topology}-{arguments.rematerialization}-remat"
    return (
        arguments.artifact_root
        / category
        / arguments.size
        / topology
        / (
            f"steps-{arguments.steps}-batch-{arguments.batch_size}"
            f"-chunks-rx{arguments.representax_chunk_size}"
            f"-st{arguments.reference_chunk_size}"
        )
    )


def _evaluate(arguments: argparse.Namespace) -> None:
    from benchmarks.dense_retrieval import (
        _directory_sha256,
        _evaluation_data,
        _representax_offline_metrics,
    )

    if arguments.artifact_kind == "representax":
        from transformers import AutoTokenizer

        from representax import load_inference_bundle
        from representax.models import SentenceBatch, make_text_processor
        from representax.models.bert import BertEncoder

        model, _ = load_inference_bundle(arguments.artifact)
        tokenizer = AutoTokenizer.from_pretrained(
            arguments.checkpoint,
            local_files_only=True,
        )
        processor = make_text_processor(
            tokenizer=tokenizer,
            batch_builder=BertEncoder.make_batch,
            max_sequence_length=LONG_SEQUENCE_LENGTH,
            sequence_length_buckets=(LONG_SEQUENCE_LENGTH,),
            include_prompt=True,
            pooling_batch_builder=SentenceBatch,
        )
    else:
        from representax.models import SentenceEncoder

        model, processor = SentenceEncoder.load_from_hf(
            arguments.artifact,
            local_files_only=True,
            parameter_dtype="float32",
            compute_dtype="bfloat16",
            sequence_length_buckets=(LONG_SEQUENCE_LENGTH,),
        )
    evaluation = _representax_offline_metrics(
        model,
        processor,
        _evaluation_data(arguments.data_directory),
        batch_size=arguments.evaluation_batch_size,
        iteration=arguments.iteration,
    )
    result = {
        "status": "accepted",
        "evaluated_by": "representax",
        "artifact_kind": arguments.artifact_kind,
        "artifact": str(arguments.artifact.resolve()),
        "artifact_sha256": _directory_sha256(arguments.artifact),
        "data_manifest": str((arguments.data_directory / "manifest.json").resolve()),
        "data_manifest_sha256": _sha256(arguments.data_directory / "manifest.json"),
        "evaluation_batch_size": arguments.evaluation_batch_size,
        "iteration": arguments.iteration,
        **evaluation,
    }
    _write_json(arguments.output.resolve(), result)
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


def _shared_evaluation(
    *,
    output: Path,
    source_data: Path,
    reports: Mapping[str, Mapping[str, Any]],
    gpu_assignments: Mapping[str, Sequence[int]],
    iteration: int,
    size: str,
    checkpoint: Path,
    sequential: bool,
) -> dict[str, Any]:
    evaluation = output / "shared-evaluation"
    evaluation.mkdir()
    processes = []
    for framework in ("representax", "sentence-transformers"):
        artifact = reports[framework]["inference_bundle"]
        if artifact is None:
            raise RuntimeError(f"{framework} did not publish an inference artifact")
        report = evaluation / f"{framework}.json"
        log = (evaluation / f"{framework}.log").open("x", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu_assignments[framework][0]),
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            TOKENIZERS_PARALLELISM="false",
        )
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "evaluate",
                "--artifact-kind",
                framework,
                "--artifact",
                str(artifact),
                "--checkpoint",
                str(checkpoint),
                "--data-directory",
                str(source_data),
                "--evaluation-batch-size",
                str(EVALUATION_BATCH_SIZES[size]),
                "--iteration",
                str(iteration),
                "--output",
                str(report),
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((framework, process, log, report))
        if sequential:
            return_code = process.wait()
            log.close()
            if return_code:
                raise RuntimeError(
                    f"shared BERT evaluation failed for {framework}: {return_code}"
                )
    failures = []
    for framework, process, log, _report in () if sequential else processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append({"framework": framework, "return_code": return_code})
    if failures:
        raise RuntimeError(f"shared BERT evaluation failed: {failures}")
    results = {
        framework: json.loads(report.read_text())
        for framework, _, _, report in processes
    }
    native_metrics = results["representax"]["metrics"]
    reference_metrics = results["sentence-transformers"]["metrics"]
    if native_metrics.keys() != reference_metrics.keys():
        raise RuntimeError("shared evaluator emitted different metric sets")
    results["metric_differences"] = {
        name: float(native_metrics[name]) - float(reference_metrics[name])
        for name in native_metrics
    }
    return results


def _pair(arguments: argparse.Namespace) -> dict[str, Any]:
    from experiments.preflights.provenance import reference_runtime_provenance

    _validate_schedule(
        steps=arguments.steps,
        long_sequence_start=arguments.long_sequence_start,
    )
    if not 0.0 < arguments.memory_fraction <= 1.0:
        raise ValueError("memory fraction must be in (0, 1]")
    if arguments.sequential and arguments.topology == "single":
        raise ValueError("sequential execution is only useful for distributed runs")
    if arguments.sequential and len(arguments.gpus) < 2:
        raise ValueError("sequential distributed execution needs at least two GPUs")
    if (
        not arguments.sequential
        and arguments.topology == "single"
        and len(arguments.gpus) != 2
    ):
        raise ValueError("single topology requires one GPU per framework")
    if (
        not arguments.sequential
        and arguments.topology != "single"
        and len(arguments.gpus) not in {4, 6}
    ):
        raise ValueError(
            "distributed topology requires two or three GPUs per framework"
        )
    if len(set(arguments.gpus)) != len(arguments.gpus):
        raise ValueError("GPU indices must be unique")
    output = _pair_output(arguments)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = prepare_checkpoint(
        arguments.artifact_root,
        arguments.size,
        tokenizer=arguments.tokenizer,
    )
    training_data = materialize_training_data(
        arguments.source_data,
        output / "training.jsonl",
        steps=arguments.steps,
        long_sequence_start=arguments.long_sequence_start,
        batch_size=arguments.batch_size,
    )
    source = _source_snapshot(output)
    if arguments.sequential:
        assignments = {
            "representax": arguments.gpus,
            "sentence-transformers": arguments.gpus,
        }
    else:
        midpoint = len(arguments.gpus) // 2
        assignments = {
            "representax": arguments.gpus[:midpoint],
            "sentence-transformers": arguments.gpus[midpoint:],
        }
    processes = []
    started = time.perf_counter()
    for framework, gpus in assignments.items():
        log = (output / f"{framework}.log").open("x", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=",".join(str(gpu) for gpu in gpus),
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        if framework == "representax":
            environment.update(
                JAX_DEFAULT_MATMUL_PRECISION="highest",
                XLA_PYTHON_CLIENT_PREALLOCATE="true",
                XLA_PYTHON_CLIENT_MEM_FRACTION=str(arguments.memory_fraction),
                JAX_COMPILATION_CACHE_DIR=str(output / "jax-cache"),
            )
        command = _worker_command(
            arguments,
            framework=framework,
            gpus=gpus,
            training_data=training_data,
            output=output,
        )
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        processes.append((framework, process, log))
        if arguments.sequential:
            return_code = process.wait()
            log.close()
            if return_code:
                failures = [{"framework": framework, "return_code": return_code}]
                _write_json(output / "failures.json", failures)
                raise RuntimeError(f"paired BERT worker failed: {failures}")
    failures = []
    for framework, process, log in () if arguments.sequential else processes:
        return_code = process.wait()
        log.close()
        if return_code:
            failures.append({"framework": framework, "return_code": return_code})
    if failures:
        _write_json(output / "failures.json", failures)
        raise RuntimeError(f"paired BERT workers failed: {failures}")
    reports = {
        framework: json.loads((output / f"{framework}.json").read_text())
        for framework in assignments
    }
    shared_evaluation = (
        _shared_evaluation(
            output=output,
            source_data=arguments.source_data,
            reports=reports,
            gpu_assignments=assignments,
            iteration=arguments.steps,
            size=arguments.size,
            checkpoint=checkpoint,
            sequential=arguments.sequential,
        )
        if arguments.lifecycle
        else None
    )
    comparisons = {}
    for phase in ("sequence-128", "sequence-512"):
        native = reports["representax"]["phase_timing"][phase]
        reference = reports["sentence-transformers"]["phase_timing"][phase]
        comparisons[phase] = {
            "representax_examples_per_second": native["examples_per_second"],
            "sentence_transformers_examples_per_second": reference[
                "examples_per_second"
            ],
            "representax_speedup": (
                native["examples_per_second"] / reference["examples_per_second"]
            ),
        }
    summary = {
        "status": "accepted",
        "size": arguments.size,
        "topology": arguments.topology,
        "gpu_assignments": assignments,
        "scientific_contract": {
            "initialization": json.loads(
                (checkpoint / "initialization.json").read_text()
            ),
            "dataset": str(training_data),
            "dataset_sha256": _sha256(training_data),
            "steps": arguments.steps,
            "global_batch_size": arguments.batch_size,
            "sequence_schedule": {
                "short_sequence_length": SHORT_SEQUENCE_LENGTH,
                "short_sequence_steps": arguments.long_sequence_start - 1,
                "long_sequence_length": LONG_SEQUENCE_LENGTH,
                "long_sequence_steps": (
                    arguments.steps - arguments.long_sequence_start + 1
                ),
                "long_sequence_start": arguments.long_sequence_start,
            },
            "loss": "asymmetric cosine MNR scale=20",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "warmup_fraction": 0.06,
            "precision": "BF16 compute with FP32 master parameters",
            "seed": SEED,
        },
        "execution_contract": {
            "representax_grad_cache_chunk_size": (arguments.representax_chunk_size),
            "reference_grad_cache_chunk_size": arguments.reference_chunk_size,
            "world_size_per_framework": len(assignments["representax"]),
            "topology": arguments.topology,
            "representax_rematerialization": arguments.rematerialization,
            "representax_parameter_layout": (
                "contracted-input"
                if arguments.size == "bert-4b" and arguments.topology == "fsdp"
                else "default"
            ),
            "reference_activation_checkpointing": (
                arguments.rematerialization == "full"
            ),
            "representax_memory_fraction": arguments.memory_fraction,
            "workers_ran_sequentially": arguments.sequential,
        },
        "source": source,
        "reference_source": reference_runtime_provenance("sentence-transformers"),
        "reports": reports,
        "shared_evaluation": shared_evaluation,
        "comparison": comparisons,
        "paired_wall_seconds": time.perf_counter() - started,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    return summary


def _latest_evaluation(run_directory: Path, framework: str) -> dict[str, Any]:
    candidates = tuple(
        path
        for path in (
            run_directory / "shared-evaluation" / f"{framework}.json",
            run_directory / "shared-evaluation" / f"{framework}-retry.json",
        )
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"shared evaluation is missing for {framework}: {run_directory}"
        )
    selected = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    return json.loads(selected.read_text())


def _aggregate_row(run_directory: Path, artifact_root: Path) -> dict[str, Any]:
    reports = {
        framework: json.loads((run_directory / f"{framework}.json").read_text())
        for framework in ("representax", "sentence-transformers")
    }
    native = reports["representax"]
    reference = reports["sentence-transformers"]
    matched_fields = (
        "size",
        "topology",
        "world_size",
        "steps",
        "global_batch_size",
    )
    differences = {
        field: (native.get(field), reference.get(field))
        for field in matched_fields
        if native.get(field) != reference.get(field)
    }
    if differences:
        raise ValueError(f"paired report contracts differ: {differences}")
    steps = int(native["steps"])
    if any(int(report["completed_iterations"]) != steps for report in reports.values()):
        raise ValueError(f"paired report is incomplete: {run_directory}")

    evaluations = {
        framework: _latest_evaluation(run_directory, framework) for framework in reports
    }
    native_metrics = evaluations["representax"]["metrics"]
    reference_metrics = evaluations["sentence-transformers"]["metrics"]
    if native_metrics.keys() != reference_metrics.keys():
        raise ValueError(f"shared evaluation metric sets differ: {run_directory}")

    initialization = json.loads(
        (
            artifact_root / "checkpoints" / str(native["size"]) / "initialization.json"
        ).read_text()
    )
    comparisons = {}
    for phase in ("sequence-128", "sequence-512"):
        native_rate = float(native["phase_timing"][phase]["examples_per_second"])
        reference_rate = float(reference["phase_timing"][phase]["examples_per_second"])
        comparisons[phase] = {
            "representax_examples_per_second": native_rate,
            "sentence_transformers_examples_per_second": reference_rate,
            "representax_speedup": native_rate / reference_rate,
        }

    native_peak = max(
        int(device["peak_bytes_in_use"]) for device in native["device_memory"]
    )
    reference_peak = max(
        int(device["allocated_bytes"]) for device in reference["device_memory"]
    )
    metric = "valid/NanoMSMARCO/cosine_ndcg@10"
    source_directory = run_directory / "source"
    source = {
        "snapshot_directory": str(source_directory.resolve()),
        "launcher_sha256": _sha256(source_directory / "run.py"),
        "working_tree_patch_sha256": _sha256(source_directory / "working-tree.patch"),
    }
    return {
        "size": native["size"],
        "parameters": int(initialization["parameters"]),
        "topology": native["topology"],
        "world_size": int(native["world_size"]),
        "global_batch_size": int(native["global_batch_size"]),
        "steps": steps,
        "representax_chunk_size": int(native["grad_cache_chunk_size"]),
        "sentence_transformers_chunk_size": int(reference["grad_cache_chunk_size"]),
        "representax_rematerialization": native.get("rematerialization", "none"),
        "sentence_transformers_activation_checkpointing": reference.get(
            "activation_checkpointing", False
        ),
        "comparison": comparisons,
        "peak_live_bytes": {
            "representax": native_peak,
            "sentence-transformers": reference_peak,
        },
        "shared_evaluation": {
            "metric": metric,
            "representax": float(native_metrics[metric]),
            "sentence-transformers": float(reference_metrics[metric]),
            "difference": float(native_metrics[metric])
            - float(reference_metrics[metric]),
        },
        "final_loss": {
            framework: statistics.fmean(
                float(row["loss"]) for row in report["losses"][-20:]
            )
            for framework, report in reports.items()
        },
        "source": source,
        "run_directory": str(run_directory.resolve()),
    }


def _markdown_results(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Controlled BERT parameter scaling",
        "",
        (
            "| Size | Topology | GPUs | Parameters | RX 128 ex/s | ST 128 ex/s "
            "| Ratio | RX 512 ex/s | ST 512 ex/s | Ratio | RX/ST peak GiB "
            "| RX/ST nDCG@10 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gib = 1024**3
    for row in rows:
        short = row["comparison"]["sequence-128"]
        long = row["comparison"]["sequence-512"]
        memory = row["peak_live_bytes"]
        quality = row["shared_evaluation"]
        row_format = (
            "| {size} | {topology} | {world_size} | {parameters:,} | "
            "{rx128:.3f} | {st128:.3f} | {ratio128:.3f}x | "
            "{rx512:.3f} | {st512:.3f} | {ratio512:.3f}x | "
            "{rx_memory:.2f}/{st_memory:.2f} | "
            "{rx_quality:.6f}/{st_quality:.6f} |"
        )
        lines.append(
            row_format.format(
                **row,
                rx128=short["representax_examples_per_second"],
                st128=short["sentence_transformers_examples_per_second"],
                ratio128=short["representax_speedup"],
                rx512=long["representax_examples_per_second"],
                st512=long["sentence_transformers_examples_per_second"],
                ratio512=long["representax_speedup"],
                rx_memory=memory["representax"] / gib,
                st_memory=memory["sentence-transformers"] / gib,
                rx_quality=quality["representax"],
                st_quality=quality["sentence-transformers"],
            )
        )
    lines.extend(
        (
            "",
            "Steady-state rates exclude the first compiled or warmed step of each "
            "sequence shape. Peak memory compares JAX live allocation with PyTorch "
            "allocated memory, not either framework's reserved pool.",
            "",
        )
    )
    return "\n".join(lines)


def _aggregate(arguments: argparse.Namespace) -> dict[str, Any]:
    rows = [
        _aggregate_row(path.resolve(), arguments.artifact_root.resolve())
        for path in arguments.run_directory
    ]
    rows.sort(
        key=lambda row: (
            SIZES.index(str(row["size"])),
            int(row["world_size"]),
            str(row["topology"]),
        )
    )
    summary = {
        "status": "accepted",
        "rows": rows,
    }
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    (output / "results.md").write_text(_markdown_results(rows))
    print(_markdown_results(rows))
    return summary


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--size", choices=SIZES, required=True)
    parser.add_argument("--topology", choices=("single", "ddp", "fsdp"), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--long-sequence-start", type=int, default=LONG_SEQUENCE_START)
    parser.add_argument("--batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    parser.add_argument("--representax-chunk-size", type=int)
    parser.add_argument("--reference-chunk-size", type=int)
    parser.add_argument("--memory-fraction", type=float, default=0.95)
    parser.add_argument(
        "--rematerialization",
        choices=("none", "selective", "full"),
        default="none",
    )
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument(
        "--lifecycle", action=argparse.BooleanOptionalAction, default=True
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    prepare.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    prepare.add_argument("--size", choices=SIZES, action="append")
    run = commands.add_parser("run")
    _add_run_arguments(run)
    worker = commands.add_parser("worker")
    worker.add_argument(
        "--framework", choices=("representax", "sentence-transformers"), required=True
    )
    worker.add_argument("--size", choices=SIZES, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--training-data", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, required=True)
    worker.add_argument("--long-sequence-start", type=int, required=True)
    worker.add_argument("--batch-size", type=int, required=True)
    worker.add_argument("--chunk-size", type=int, required=True)
    worker.add_argument("--topology", choices=("single", "ddp", "fsdp"), required=True)
    worker.add_argument("--world-size", type=int, required=True)
    worker.add_argument(
        "--rematerialization",
        choices=("none", "selective", "full"),
        required=True,
    )
    worker.add_argument("--lifecycle", action="store_true")
    worker.add_argument(
        "--native-stage",
        choices=("initial", "resume"),
        help=argparse.SUPPRESS,
    )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument(
        "--artifact-kind",
        choices=("representax", "sentence-transformers"),
        required=True,
    )
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-directory", type=Path, required=True)
    evaluate.add_argument("--evaluation-batch-size", type=int, required=True)
    evaluate.add_argument("--iteration", type=int, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    aggregate.add_argument("--run-directory", type=Path, action="append", required=True)
    aggregate.add_argument("--output-directory", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        for size in arguments.size or SIZES:
            checkpoint = prepare_checkpoint(
                arguments.artifact_root.resolve(),
                size,
                tokenizer=arguments.tokenizer.resolve(),
            )
            print(checkpoint)
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "evaluate":
        _evaluate(arguments)
    elif arguments.command == "aggregate":
        _aggregate(arguments)
    else:
        arguments.artifact_root = arguments.artifact_root.resolve()
        arguments.source_data = arguments.source_data.resolve()
        arguments.tokenizer = arguments.tokenizer.resolve()
        default_chunk_size = GRAD_CACHE_CHUNKS[arguments.size]
        if arguments.representax_chunk_size is None:
            arguments.representax_chunk_size = default_chunk_size
        if arguments.reference_chunk_size is None:
            arguments.reference_chunk_size = default_chunk_size
        _pair(arguments)


if __name__ == "__main__":
    main()
