"""Matched outcome-reward preflight against TRL RewardTrainer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from experiments.paper.provenance import reference_source, write_reference_result

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
TEXT_REWARD_MANIFEST = ROOT / "benchmarks/configs/paper-text-reward-v1.json"
FRAMEWORKS = ("representax", "trl")
MICRO_BATCH_SIZE = 4
EVALUATION_BATCH_SIZE = 4
SEQUENCE_BUCKETS = (128, 256, 512, 1024)
PaddingMode = Literal["dynamic", "static"]


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    global_batch_size: int
    maximum_length: int
    objective: str
    reference_version: str


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve and validate the outcome-reward row in both paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(TEXT_REWARD_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "pairwise-outcome-reward"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "pairwise-outcome-reward"
    )
    if campaign_row["frameworks"] != ["representax", "trl"]:
        raise ValueError("unexpected outcome-reward frameworks")
    if panel_row["reference"] != "trl":
        raise ValueError("unexpected outcome-reward reference")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the outcome-reward reference requires a release")
    dataset = panel["datasets"][panel_row["train"][0]]
    if panel_row["train"] != panel_row["evaluate"]:
        raise ValueError("outcome-reward training and evaluation datasets differ")
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        dataset_id=dataset["repo_id"],
        dataset_revision=dataset["revision"],
        global_batch_size=int(campaign_row["global_batch"]),
        maximum_length=int(campaign_row["maximum_sequence_length"]),
        objective=panel_row["objective"],
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


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise TypeError("UltraFeedback preferences must contain message lists")
    messages = []
    for message in value:
        if not isinstance(message, Mapping):
            raise TypeError("UltraFeedback messages must be mappings")
        role = str(message["role"])
        content = str(message["content"])
        messages.append({"role": role, "content": content})
    return messages


def preference_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    maximum_length: int,
    tokenize: Callable[[Sequence[Mapping[str, str]]], Sequence[int]],
) -> tuple[dict[str, Any], ...]:
    """Take the first finite, non-truncated preference rows in source order."""

    if count <= 0:
        raise ValueError("preference row count must be positive")
    selected = []
    for source_index, row in enumerate(rows):
        chosen = _messages(row["chosen"])
        rejected = _messages(row["rejected"])
        chosen_ids = [int(value) for value in tokenize(chosen)]
        rejected_ids = [int(value) for value in tokenize(rejected)]
        if not chosen_ids or not rejected_ids:
            raise ValueError("tokenized preference sequences must be non-empty")
        if max(len(chosen_ids), len(rejected_ids)) > maximum_length:
            continue
        selected.append(
            {
                "source_index": source_index,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "score_chosen": float(row["score_chosen"]),
                "score_rejected": float(row["score_rejected"]),
            }
        )
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"UltraFeedback contains only {len(selected)} usable rows")
    return tuple(selected)


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    for batch in parquet.ParquetFile(path).iter_batches(batch_size=2_048):
        yield from batch.to_pylist()


def _resolve_parquet(
    path: Path | None,
    *,
    filename: str,
    cache_directory: Path,
    contract: FrozenContract,
) -> Path:
    if path is not None:
        return path.resolve()
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            contract.dataset_id,
            filename,
            repo_type="dataset",
            revision=contract.dataset_revision,
            cache_dir=cache_directory,
        )
    ).resolve()


def prepare_data(
    output: Path,
    *,
    checkpoint: Path,
    cache_directory: Path,
    train_parquet: Path | None,
    test_parquet: Path | None,
    training_rows: int,
    evaluation_rows: int,
) -> dict[str, Any]:
    """Materialize deterministic, disjoint UltraFeedback preflight views."""

    from transformers import AutoTokenizer

    contract = frozen_contract()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer: Any = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )

    def tokenize(messages: Sequence[Mapping[str, str]]) -> Sequence[int]:
        encoded: Any = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        return encoded["input_ids"]

    train_source = _resolve_parquet(
        train_parquet,
        filename="data/train-00000-of-00001.parquet",
        cache_directory=cache_directory,
        contract=contract,
    )
    test_source = _resolve_parquet(
        test_parquet,
        filename="data/test-00000-of-00001.parquet",
        cache_directory=cache_directory,
        contract=contract,
    )
    train = preference_rows(
        _parquet_rows(train_source),
        count=training_rows,
        maximum_length=contract.maximum_length,
        tokenize=tokenize,
    )
    evaluation = preference_rows(
        _parquet_rows(test_source),
        count=evaluation_rows,
        maximum_length=contract.maximum_length,
        tokenize=tokenize,
    )
    train_path = output / "train.jsonl"
    evaluation_path = output / "evaluation.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(evaluation_path, evaluation)
    lengths = [
        len(row[field])
        for row in (*train, *evaluation)
        for field in ("chosen_ids", "rejected_ids")
    ]
    manifest = {
        "schema_version": "representax-outcome-reward-data-v1",
        "dataset_id": contract.dataset_id,
        "dataset_revision": contract.dataset_revision,
        "model_id": contract.model_id,
        "model_revision": contract.model_revision,
        "train_source": str(train_source),
        "test_source": str(test_source),
        "training_rows": len(train),
        "evaluation_rows": len(evaluation),
        "maximum_length": contract.maximum_length,
        "observed_minimum_tokens": min(lengths),
        "observed_maximum_tokens": max(lengths),
        "train_sha256": _sha256(train_path),
        "evaluation_sha256": _sha256(evaluation_path),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _representax_job(
    *,
    checkpoint: Path,
    data_directory: Path,
    steps: int,
    seed: int,
    global_batch_size: int | None = None,
    micro_batch_size: int = MICRO_BATCH_SIZE,
    sequence_length_buckets: Sequence[int] = SEQUENCE_BUCKETS,
    lifecycle: bool = True,
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
        RewardEvaluatorConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.reward_modeling import (
        BradleyTerryConfig,
        PairwiseRewardConfig,
    )

    contract = frozen_contract()
    resolved_global_batch_size = global_batch_size or contract.global_batch_size
    if resolved_global_batch_size % micro_batch_size:
        raise ValueError("global batch size must be divisible by microbatch size")
    manifest = _document(data_directory / "manifest.json")
    required = resolved_global_batch_size * steps
    if int(manifest["training_rows"]) < required:
        raise ValueError(f"preflight requires at least {required} training rows")
    if lifecycle and (steps < 2 or steps % 2):
        raise ValueError("preflight steps must be an even integer of at least two")

    def data(path: Path) -> DataConfig:
        return DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=ComponentConfig(
                target=("representax.tasks.reward_modeling:PairwiseRewardCollator")
            ),
            drop_remainder=True,
            num_threads=2,
            prefetch_buffer_size=2,
        )

    return JobConfig(
        name="paper-preflight-outcome-reward",
        model=ModelConfig(
            target="representax.models.qwen_reward:load_qwen_reward_model",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "head_seed": seed,
                "sequence_length_buckets": list(sequence_length_buckets),
            },
        ),
        task=PairwiseRewardConfig(),
        loss=BradleyTerryConfig(),
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
                    "peak_value": 1e-5,
                    "warmup_steps": 1,
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=data(data_directory / "train.jsonl"),
        training=TrainingConfig(
            global_batch_size=resolved_global_batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=(
                    resolved_global_batch_size // micro_batch_size
                ),
            ),
            activation_rematerialization="full",
            donate_buffers=True,
            precision=PrecisionConfig.bfloat16_mixed(),
        ),
        checkpointing=(
            CheckpointConfig(every=steps // 2, keep=1, save_final=True)
            if lifecycle
            else None
        ),
        logging=LoggingConfig(
            console_every=1,
            timing=True,
            accelerator=lifecycle,
        ),
        evaluation=(
            EvaluationConfig(
                data=data(data_directory / "evaluation.jsonl"),
                batch_size=EVALUATION_BATCH_SIZE,
                evaluators=(
                    RewardEvaluatorConfig(name="ultrafeedback", mode="pairwise"),
                ),
                on_start=True,
                on_end=True,
                primary_metric="valid/ultrafeedback/pairwise_accuracy",
                primary_metric_mode="max",
                save_best=False,
            )
            if lifecycle
            else None
        ),
        export=(
            ExportConfig(
                selection="final",
                huggingface=HuggingFaceExportConfig(
                    source_checkpoint=str(checkpoint),
                    adapter=ComponentConfig(
                        target=(
                            "representax.models.qwen_reward:QwenRewardCheckpointAdapter"
                        )
                    ),
                    verify_reload=True,
                ),
            )
            if lifecycle
            else ExportConfig(enabled=False)
        ),
    )


def steady_state(
    rows: Sequence[Mapping[str, Any]], *, batch_size: int
) -> dict[str, float]:
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
        return {"measured_steps": 0.0}
    return {
        "measured_steps": float(len(durations)),
        "median_step_seconds": statistics.median(durations),
        "examples_per_second": batch_size * len(durations) / sum(durations),
    }


def reference_timing(
    rows: Sequence[Mapping[str, Any]], *, batch_size: int
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["step"]))
    if not ordered:
        raise ValueError("TRL recorded no optimizer-step timings")
    first = float(ordered[0]["seconds"])
    warmed = [float(row["seconds"]) for row in ordered[1:]]
    return {
        "first_step_seconds": first,
        "compilation_seconds": 0.0,
        "warm_steps": len(warmed),
        "warm_median_step_seconds": (statistics.median(warmed) if warmed else None),
        "warm_examples_per_second": (
            batch_size * len(warmed) / sum(warmed) if warmed else None
        ),
        "steps": ordered,
    }


def optimizer_token_capacities(
    input_shapes: Sequence[Sequence[int]],
    *,
    gradient_accumulation_steps: int,
) -> list[int]:
    """Sum padded model-input tokens for each optimizer update."""

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    if len(input_shapes) % gradient_accumulation_steps:
        raise ValueError("microbatch shapes do not form complete optimizer updates")
    capacities = []
    for start in range(0, len(input_shapes), gradient_accumulation_steps):
        capacities.append(
            sum(
                int(shape[0]) * int(shape[1])
                for shape in input_shapes[start : start + gradient_accumulation_steps]
            )
        )
    return capacities


def _representax_probe_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    global_batch_size: int,
    micro_batch_size: int,
    padding: PaddingMode,
) -> dict[str, Any]:
    import jax

    from representax.train import run_job

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("outcome-reward probe requires one visible GPU")
    sequence_buckets = (
        (frozen_contract().maximum_length,) if padding == "static" else SEQUENCE_BUCKETS
    )
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        sequence_length_buckets=sequence_buckets,
        lifecycle=False,
    )
    completed = run_job(job, run_directory)
    jax.block_until_ready(completed.state)
    rows = _read_jsonl(run_directory / "metrics.jsonl")
    training = tuple(row for row in rows if row.get("event") == "training_step")
    capacities = [
        int(row["metrics"]["perf/token_capacity"])
        for row in training
        if "perf/token_capacity" in row["metrics"]
    ]
    return {
        "schema_version": "representax-outcome-reward-probe-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": global_batch_size // micro_batch_size,
        "padding": padding,
        "padding_scope": "global-batch",
        "sequence_length_buckets": list(sequence_buckets),
        "model_input_shape": [
            micro_batch_size * 2,
            frozen_contract().maximum_length,
        ],
        "pair_execution": "one-concatenated-forward",
        "padded_tokens_per_update": capacities,
        "actual_tokens_per_update": [
            int(row["metrics"]["perf/tokens"])
            for row in training
            if "perf/tokens" in row["metrics"]
        ],
        "compilation_and_first_step_seconds": sum(
            float(row["metrics"].get("perf/compilation_and_first_step_seconds", 0.0))
            for row in training
        ),
        "timing": steady_state(training, batch_size=global_batch_size),
        "training_metrics": [row["metrics"] for row in training],
    }


def _trl_probe_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    global_batch_size: int,
    micro_batch_size: int,
    padding: PaddingMode,
) -> dict[str, Any]:
    import torch
    import transformers
    from torch.utils.data import SequentialSampler
    from transformers import AutoTokenizer
    from transformers.trainer_callback import TrainerCallback

    trl = import_module("trl")
    RewardConfig = trl.RewardConfig
    RewardTrainer = trl.RewardTrainer
    DataCollatorForPreference = import_module(
        "trl.trainer.reward_trainer"
    ).DataCollatorForPreference
    contract = frozen_contract()
    if trl.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected trl=={contract.reference_version}, found {trl.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TRL outcome-reward probe requires one visible GPU")
    if global_batch_size % micro_batch_size:
        raise ValueError("global batch size must be divisible by microbatch size")
    accumulation_steps = global_batch_size // micro_batch_size

    class ProbeRewardTrainer(RewardTrainer):
        input_shapes: list[list[int]]
        actual_tokens: list[int]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_shapes = []
            self.actual_tokens = []

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(dataset)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            if model.training:
                self.input_shapes.append(list(inputs["input_ids"].shape))
                self.actual_tokens.append(int(inputs["attention_mask"].sum().item()))
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.started = 0.0
            self.rows: list[dict[str, float | int]] = []

        def on_step_begin(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            del args, control, kwargs
            torch.cuda.synchronize()
            self.rows.append(
                {
                    "step": int(state.global_step),
                    "seconds": time.perf_counter() - self.started,
                }
            )

    run_directory.mkdir(parents=True, exist_ok=False)
    train = _reference_dataset(data_directory / "train.jsonl")
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    pad_to_multiple_of = contract.maximum_length if padding == "static" else None
    arguments = RewardConfig(
        output_dir=str(run_directory / "unused-checkpoints"),
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        max_steps=steps,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=seed,
        data_seed=seed,
        max_length=contract.maximum_length,
        pad_to_multiple_of=pad_to_multiple_of,
        center_rewards_coefficient=None,
        model_init_kwargs={
            "dtype": "bfloat16",
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )
    timer = StepTimer()
    collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    trainer = ProbeRewardTrainer(
        model=str(checkpoint),
        args=arguments,
        train_dataset=train,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    torch.cuda.synchronize()
    if int(trainer.state.global_step) != steps:
        raise RuntimeError("TRL probe did not complete every optimizer update")
    input_shapes = trainer.input_shapes
    padded_capacities = optimizer_token_capacities(
        input_shapes,
        gradient_accumulation_steps=accumulation_steps,
    )
    actual_capacities = [
        sum(trainer.actual_tokens[start : start + accumulation_steps])
        for start in range(0, len(trainer.actual_tokens), accumulation_steps)
    ]
    return {
        "schema_version": "representax-outcome-reward-probe-v1",
        "framework": "trl",
        "framework_version": trl.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "global_batch_size": global_batch_size,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "padding": padding,
        "padding_scope": "microbatch",
        "input_shapes": input_shapes,
        "pair_execution": "one-concatenated-forward",
        "padded_tokens_per_update": padded_capacities,
        "actual_tokens_per_update": actual_capacities,
        "timing": reference_timing(timer.rows, batch_size=global_batch_size),
        "training_metrics": list(trainer.state.log_history),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _representax_kernel_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import equinox as eqx
    import jax

    from representax.config import PrecisionConfig
    from representax.models.qwen_reward import load_qwen_reward_model
    from representax.precision import (
        precision_context,
        prepare_master_model,
        resolve_precision_policy,
    )
    from representax.tasks.reward_modeling import (
        PairwiseRewardCollator,
        PairwiseRewardTask,
    )

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("outcome-reward kernel probe requires one visible GPU")
    if iterations <= 0:
        raise ValueError("kernel iterations must be positive")
    model, processor = load_qwen_reward_model(
        checkpoint,
        revision=frozen_contract().model_revision,
        local_files_only=True,
        parameter_dtype=jax.numpy.bfloat16,
        compute_dtype=jax.numpy.bfloat16,
        head_seed=seed,
        sequence_length_buckets=(frozen_contract().maximum_length,),
    )
    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    model = prepare_master_model(model, precision)
    batch = PairwiseRewardCollator(processor=processor)(
        _read_jsonl(data_directory / "train.jsonl")[:MICRO_BATCH_SIZE]
    )
    task = PairwiseRewardTask()

    @eqx.filter_value_and_grad
    def loss_and_gradient(candidate, inputs):
        with precision_context(precision):
            return task.loss(candidate, inputs).loss

    kernel = eqx.filter_jit(loss_and_gradient)

    def execute() -> tuple[float, float]:
        started = time.perf_counter()
        value, gradients = kernel(model, batch)
        jax.block_until_ready((value, gradients))
        return time.perf_counter() - started, float(value)

    first_seconds, loss_value = execute()
    durations = [execute()[0] for _ in range(iterations)]
    median_seconds = statistics.median(durations)
    return {
        "schema_version": "representax-outcome-reward-kernel-probe-v1",
        "framework": "representax",
        "input_shape": [MICRO_BATCH_SIZE * 2, frozen_contract().maximum_length],
        "pairs": MICRO_BATCH_SIZE,
        "rematerialization": "full",
        "optimizer": False,
        "first_compiled_forward_backward_seconds": first_seconds,
        "warm_forward_backward_seconds": durations,
        "warm_median_forward_backward_seconds": median_seconds,
        "warm_pairs_per_second": MICRO_BATCH_SIZE / median_seconds,
        "loss": loss_value,
    }


def _trl_kernel_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    DataCollatorForPreference = import_module(
        "trl.trainer.reward_trainer"
    ).DataCollatorForPreference
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TRL outcome-reward kernel probe requires one visible GPU")
    if iterations <= 0:
        raise ValueError("kernel iterations must be positive")
    torch.manual_seed(seed)
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=frozen_contract().maximum_length,
    )
    batch = collator(_read_jsonl(data_directory / "train.jsonl")[:MICRO_BATCH_SIZE])
    batch = {name: value.cuda() for name, value in batch.items()}

    def execute() -> tuple[float, float]:
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(**batch, use_cache=False).logits.squeeze(-1)
        chosen, rejected = torch.chunk(logits, chunks=2)
        loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
        loss.backward()
        torch.cuda.synchronize()
        return time.perf_counter() - started, float(loss.detach())

    torch.cuda.reset_peak_memory_stats()
    first_seconds, loss_value = execute()
    durations = [execute()[0] for _ in range(iterations)]
    median_seconds = statistics.median(durations)
    return {
        "schema_version": "representax-outcome-reward-kernel-probe-v1",
        "framework": "trl",
        "transformers_version": transformers.__version__,
        "input_shape": list(batch["input_ids"].shape),
        "pairs": MICRO_BATCH_SIZE,
        "rematerialization": "gradient-checkpointing",
        "optimizer": False,
        "first_eager_forward_backward_seconds": first_seconds,
        "warm_forward_backward_seconds": durations,
        "warm_median_forward_backward_seconds": median_seconds,
        "warm_pairs_per_second": MICRO_BATCH_SIZE / median_seconds,
        "loss": loss_value,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
    }


def _native_probe(model: Any, checkpoint: Path, rows: Sequence[Any]) -> np.ndarray:
    import jax

    from representax.config import PrecisionConfig
    from representax.core import score_logits
    from representax.models.qwen_reward import make_qwen_reward_processor
    from representax.precision import precision_context, resolve_precision_policy
    from representax.tasks.reward_modeling import PairwiseRewardCollator

    processor = make_qwen_reward_processor(
        checkpoint,
        model.config.backbone.pad_token_id,
        sequence_length_buckets=SEQUENCE_BUCKETS,
    )
    batch = PairwiseRewardCollator(processor=processor)(rows)
    precision = resolve_precision_policy(PrecisionConfig.bfloat16_mixed())
    with precision_context(precision):
        values = np.stack(
            (
                np.asarray(score_logits(model, batch.chosen)),
                np.asarray(score_logits(model, batch.rejected)),
            ),
            axis=-1,
        )
    jax.block_until_ready(values)
    return values


def _elapsed_run_seconds(events: Sequence[Mapping[str, Any]]) -> float:
    started = next(
        row["timestamp"] for row in events if row.get("event") == "training_started"
    )
    finished = next(
        row["timestamp"]
        for row in reversed(events)
        if row.get("event") == "training_finished"
    )
    return (
        datetime.fromisoformat(finished) - datetime.fromisoformat(started)
    ).total_seconds()


def _representax_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    padding: PaddingMode = "static",
) -> dict[str, Any]:
    import jax

    from representax import load_inference_bundle
    from representax.models.qwen_reward import QwenRewardModel
    from representax.train import run_job

    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("outcome-reward preflight requires one visible GPU")
    contract = frozen_contract()
    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
        sequence_length_buckets=(
            (contract.maximum_length,) if padding == "static" else SEQUENCE_BUCKETS
        ),
    )
    run_manifest_path = run_directory / "run.json"
    run_manifest = _document(run_manifest_path) if run_manifest_path.is_file() else None
    if run_manifest is not None and run_manifest.get("status") == "completed":
        inference_bundle = run_directory / "final-model"
        trained_model, _ = load_inference_bundle(inference_bundle)
        completed_iterations = int(run_manifest["completed_iterations"])
        resumed = True
    else:
        started = time.perf_counter()
        if run_manifest is None:
            paused = run_job(job, run_directory, stop_after=steps // 2)
            if paused.completed_iterations != steps // 2:
                raise RuntimeError(
                    "Representax did not stop at the midpoint checkpoint"
                )
            del paused
            gc.collect()
            jax.clear_caches()
        completed = run_job(job, run_directory, resume=True)
        jax.block_until_ready(completed.state)
        if completed.inference_bundle is None:
            raise RuntimeError("Representax did not export an inference bundle")
        inference_bundle = completed.inference_bundle
        trained_model = completed.state.model
        completed_iterations = completed.completed_iterations
        resumed = completed.resumed
        del started
    if not resumed or completed_iterations != steps:
        raise RuntimeError("Representax did not resume to the final update")
    if not isinstance(trained_model, QwenRewardModel):
        raise TypeError("outcome-reward training returned a different model family")

    probe_rows = _read_jsonl(data_directory / "evaluation.jsonl")[:2]
    native_probe = _native_probe(trained_model, checkpoint, probe_rows)
    restored, reloaded_job = load_inference_bundle(inference_bundle)
    reload_probe = _native_probe(restored, checkpoint, probe_rows)
    if not np.array_equal(native_probe, reload_probe):
        raise RuntimeError("native inference reload changed reward scores")

    rows = _read_jsonl(run_directory / "metrics.jsonl")
    events = _read_jsonl(run_directory / "events.jsonl")
    training = tuple(row for row in rows if row.get("event") == "training_step")
    evaluations = tuple(row for row in rows if row.get("event") == "evaluation")
    compilation = sum(
        float(row["metrics"].get("perf/compilation_and_first_step_seconds", 0.0))
        for row in training
    )
    bundle_manifest = _document(inference_bundle / "manifest.json")
    peak_device_bytes = max(
        (
            int(row["metrics"]["accelerator/0/memory_used_bytes"])
            for row in rows
            if row.get("event") == "accelerator"
        ),
        default=0,
    )
    shutil.rmtree(run_directory / "checkpoints", ignore_errors=True)
    return {
        "schema_version": "representax-outcome-reward-worker-v1",
        "framework": "representax",
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": (contract.global_batch_size // MICRO_BATCH_SIZE),
        "maximum_length": contract.maximum_length,
        "padding": padding,
        "padding_scope": "global-batch",
        "model_input_shape": [MICRO_BATCH_SIZE * 2, contract.maximum_length],
        "pair_execution": "one-concatenated-forward",
        "precision": "bfloat16-compute-float32-master",
        "elapsed_seconds": _elapsed_run_seconds(events),
        "compilation_and_first_step_seconds": compilation,
        "steady_state": steady_state(training, batch_size=contract.global_batch_size),
        "initial_evaluation": evaluations[0]["metrics"],
        "final_evaluation": evaluations[-1]["metrics"],
        "final_training": training[-1]["metrics"],
        "training_metrics": [row["metrics"] for row in training],
        "resumed": resumed,
        "checkpoint_iterations": [steps // 2, steps],
        "checkpoint_resume_verified": True,
        "checkpoint_artifacts_retained": False,
        "inference_bundle": str(inference_bundle),
        "huggingface_export": str(inference_bundle / "huggingface"),
        "huggingface_verified": bundle_manifest["huggingface"]["verified"],
        "reload_job_name": reloaded_job.name,
        "reload_probe_maximum_absolute_difference": float(
            np.max(np.abs(native_probe - reload_probe))
        ),
        "probe_scores": native_probe.tolist(),
        "peak_device_bytes": peak_device_bytes,
    }


def _reference_dataset(path: Path) -> Any:
    import datasets

    dataset = datasets.Dataset.from_json(str(path))
    required = {"chosen_ids", "rejected_ids"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"preference data is missing columns: {sorted(missing)}")
    return dataset


def _trl_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
    padding: PaddingMode = "static",
) -> dict[str, Any]:
    import torch
    import transformers
    from torch.utils.data import SequentialSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.trainer_callback import TrainerCallback

    trl = import_module("trl")
    RewardConfig = trl.RewardConfig
    RewardTrainer = trl.RewardTrainer
    DataCollatorForPreference = import_module(
        "trl.trainer.reward_trainer"
    ).DataCollatorForPreference

    contract = frozen_contract()
    if trl.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected trl=={contract.reference_version}, found {trl.__version__}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TRL outcome-reward preflight requires one visible GPU")

    class SequentialRewardTrainer(RewardTrainer):
        input_shapes: list[list[int]]
        actual_tokens: list[int]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_shapes = []
            self.actual_tokens = []

        def _get_train_sampler(self, train_dataset=None):
            dataset = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(dataset)

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            if model.training:
                self.input_shapes.append(list(inputs["input_ids"].shape))
                self.actual_tokens.append(int(inputs["attention_mask"].sum().item()))
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.started = 0.0
            self.rows: list[dict[str, float | int]] = []

        def on_step_begin(self, args, state, control, **kwargs):
            del args, state, control, kwargs
            torch.cuda.synchronize()
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            del args, control, kwargs
            torch.cuda.synchronize()
            self.rows.append(
                {
                    "step": int(state.global_step),
                    "seconds": time.perf_counter() - self.started,
                }
            )

    class StopAtMidpoint(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if int(state.global_step) >= steps // 2:
                control.should_training_stop = True
            return control

    train = _reference_dataset(data_directory / "train.jsonl")
    evaluation = _reference_dataset(data_directory / "evaluation.jsonl")
    tokenizer: Any = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    pad_to_multiple_of = contract.maximum_length if padding == "static" else None
    training_arguments = RewardConfig(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        per_device_eval_batch_size=EVALUATION_BATCH_SIZE,
        gradient_accumulation_steps=contract.global_batch_size // MICRO_BATCH_SIZE,
        max_steps=steps,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        disable_tqdm=True,
        save_strategy="steps",
        save_steps=steps // 2,
        save_total_limit=1,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=seed,
        data_seed=seed,
        max_length=contract.maximum_length,
        pad_to_multiple_of=pad_to_multiple_of,
        center_rewards_coefficient=None,
        model_init_kwargs={
            "dtype": "bfloat16",
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
    )

    first_timer = StepTimer()
    collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    trainer = SequentialRewardTrainer(
        model=str(checkpoint),
        args=training_arguments,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[first_timer, StopAtMidpoint()],
    )
    torch.cuda.reset_peak_memory_stats()
    initial_evaluation = trainer.evaluate()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    first_training_seconds = time.perf_counter() - started
    if int(trainer.state.global_step) != steps // 2:
        raise RuntimeError("TRL did not stop at the midpoint checkpoint")
    midpoint = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
    if not midpoint.is_dir():
        raise RuntimeError("TRL did not save its midpoint checkpoint")
    first_input_shapes = trainer.input_shapes
    first_actual_tokens = trainer.actual_tokens
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    second_timer = StepTimer()
    tokenizer = cast(
        Any, AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    )
    collator = DataCollatorForPreference(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    trainer = SequentialRewardTrainer(
        model=str(checkpoint),
        args=training_arguments,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[second_timer],
    )
    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=str(midpoint))
    torch.cuda.synchronize()
    second_training_seconds = time.perf_counter() - started
    if int(trainer.state.global_step) != steps:
        raise RuntimeError("TRL did not resume to the final update")
    final_evaluation = trainer.evaluate()
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    tokenizer.save_pretrained(export)

    probe_rows = _read_jsonl(data_directory / "evaluation.jsonl")[:2]
    collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)

    def probe(model: Any) -> np.ndarray:
        batch = collator(probe_rows)
        batch = {name: value.cuda() for name, value in batch.items()}
        model.eval()
        with torch.no_grad():
            values = model(**batch).logits.squeeze(-1)
        chosen, rejected = torch.chunk(values, chunks=2)
        return torch.stack((chosen, rejected), dim=-1).float().cpu().numpy()

    final_probe = probe(trainer.model)
    peak_bytes = int(torch.cuda.max_memory_allocated())
    training_log = list(trainer.state.log_history)
    second_input_shapes = trainer.input_shapes
    second_actual_tokens = trainer.actual_tokens
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = AutoModelForSequenceClassification.from_pretrained(
        export,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    reload_probe = probe(reloaded)
    if not np.array_equal(final_probe, reload_probe):
        raise RuntimeError("TRL export/reload changed reward scores")

    timing_rows = (*first_timer.rows, *second_timer.rows)
    input_shapes = [*first_input_shapes, *second_input_shapes]
    actual_tokens = [*first_actual_tokens, *second_actual_tokens]
    accumulation_steps = contract.global_batch_size // MICRO_BATCH_SIZE
    shutil.rmtree(run_directory / "checkpoints", ignore_errors=True)
    return {
        "schema_version": "representax-outcome-reward-worker-v1",
        "framework": "trl",
        "framework_version": trl.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "global_batch_size": contract.global_batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": (contract.global_batch_size // MICRO_BATCH_SIZE),
        "maximum_length": contract.maximum_length,
        "padding": padding,
        "padding_scope": "microbatch",
        "input_shapes": input_shapes,
        "pair_execution": "one-concatenated-forward",
        "padded_tokens_per_update": optimizer_token_capacities(
            input_shapes,
            gradient_accumulation_steps=accumulation_steps,
        ),
        "actual_tokens_per_update": [
            sum(actual_tokens[start : start + accumulation_steps])
            for start in range(0, len(actual_tokens), accumulation_steps)
        ],
        "precision": "bfloat16",
        "training_seconds": first_training_seconds + second_training_seconds,
        "examples_per_second": (
            contract.global_batch_size
            * steps
            / (first_training_seconds + second_training_seconds)
        ),
        "timing": reference_timing(timing_rows, batch_size=contract.global_batch_size),
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "training_metrics": training_log,
        "resumed": True,
        "checkpoint_iterations": [steps // 2, steps],
        "checkpoint_resume_verified": True,
        "checkpoint_artifacts_retained": False,
        "inference_bundle": str(export),
        "reload_probe_maximum_absolute_difference": float(
            np.max(np.abs(final_probe - reload_probe))
        ),
        "probe_scores": final_probe.tolist(),
        "peak_device_bytes": peak_bytes,
    }


def _verify_huggingface_export(report: Path, output: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    DataCollatorForPreference = import_module(
        "trl.trainer.reward_trainer"
    ).DataCollatorForPreference

    values = _document(report)
    export = Path(values["huggingface_export"])
    tokenizer: Any = AutoTokenizer.from_pretrained(export, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        export, local_files_only=True, dtype=torch.bfloat16
    )
    probe_rows = values["probe_rows"]
    batch = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)(probe_rows)
    model.eval()
    with torch.no_grad():
        scores = model(**batch).logits.squeeze(-1).float().numpy()
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Transformers reload produced non-finite reward scores")
    _write_json(
        output,
        {
            "framework": "representax-huggingface-export",
            "source": str(export),
            "finite": True,
            "scores": scores.tolist(),
        },
    )


def _worker(arguments: argparse.Namespace) -> None:
    if arguments.framework == "representax":
        report = _representax_worker(
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
            padding=arguments.padding,
        )
        report["probe_rows"] = list(
            _read_jsonl(arguments.data_directory / "evaluation.jsonl")[:2]
        )
    else:
        report = _trl_worker(
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
            padding=arguments.padding,
        )
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        write_reference_result(arguments.report, report, reference="trl")


def _probe_worker(arguments: argparse.Namespace) -> None:
    options = {
        "checkpoint": arguments.checkpoint,
        "data_directory": arguments.data_directory,
        "run_directory": arguments.run_directory,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "global_batch_size": arguments.global_batch_size,
        "micro_batch_size": arguments.micro_batch_size,
        "padding": arguments.padding,
    }
    if arguments.framework == "representax":
        report = _representax_probe_worker(**options)
    else:
        report = _trl_probe_worker(**options)
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        report = write_reference_result(arguments.report, report, reference="trl")
    print(json.dumps(report, indent=2, sort_keys=True))


def _kernel_worker(arguments: argparse.Namespace) -> None:
    options = {
        "checkpoint": arguments.checkpoint,
        "data_directory": arguments.data_directory,
        "iterations": arguments.iterations,
        "seed": arguments.seed,
    }
    if arguments.framework == "representax":
        report = _representax_kernel_worker(**options)
    else:
        report = _trl_kernel_worker(**options)
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        report = write_reference_result(arguments.report, report, reference="trl")
    print(json.dumps(report, indent=2, sort_keys=True))


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


def _pair(arguments: argparse.Namespace) -> None:
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        if report.is_file():
            reports[framework] = _document(report)
            continue
        python = sys.executable if framework == "representax" else arguments.trl_python
        command = [
            str(python),
            "-m",
            "experiments.paper.outcome_reward",
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
            "--padding",
            arguments.padding,
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        if framework == "representax":
            environment.update(
                {
                    "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                }
            )
        _run_process(command, environment=environment, log=output / f"{framework}.log")
        reports[framework] = _document(report)

    verification = output / "representax-transformers-reload.json"
    if not verification.is_file():
        verification_environment = os.environ.copy()
        verification_environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(arguments.gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        _run_process(
            (
                str(arguments.trl_python),
                "-m",
                "experiments.paper.outcome_reward",
                "verify-export",
                "--report",
                str(output / "representax.json"),
                "--output",
                str(verification),
            ),
            environment=verification_environment,
            log=output / "representax-transformers-reload.log",
        )
    representax = reports["representax"]
    trl = reports["trl"]
    representax_throughput = float(representax["steady_state"]["examples_per_second"])
    trl_throughput = float(trl["timing"]["warm_examples_per_second"])
    summary = {
        "schema_version": "representax-outcome-reward-preflight-v1",
        "contract": {
            **asdict(frozen_contract()),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "gpu": arguments.gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
        },
        **reports,
        "comparison": {
            "representax_warm_examples_per_second": representax_throughput,
            "trl_warm_examples_per_second": trl_throughput,
            "representax_to_trl_throughput": (representax_throughput / trl_throughput),
            "representax_initial_pairwise_accuracy": representax["initial_evaluation"][
                "valid/ultrafeedback/pairwise_accuracy"
            ],
            "trl_initial_pairwise_accuracy": trl["initial_evaluation"]["eval_accuracy"],
            "representax_final_pairwise_accuracy": representax["final_evaluation"][
                "valid/ultrafeedback/pairwise_accuracy"
            ],
            "trl_final_pairwise_accuracy": trl["final_evaluation"]["eval_accuracy"],
            "representax_training_losses": [
                row["train/loss"] for row in representax["training_metrics"]
            ],
            "trl_training_losses": [
                row["loss"] for row in trl["training_metrics"] if "loss" in row
            ],
        },
        "representax_transformers_reload": _document(verification),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument(
        "--cache-directory", type=Path, default=Path("/raid/.cache/huggingface")
    )
    prepare.add_argument("--train-parquet", type=Path)
    prepare.add_argument("--test-parquet", type=Path)
    prepare.add_argument("--training-rows", type=int, default=512)
    prepare.add_argument("--evaluation-rows", type=int, default=32)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=4)
    worker.add_argument("--seed", type=int, default=7)
    worker.add_argument("--padding", choices=("dynamic", "static"), default="static")

    probe = subparsers.add_parser("probe")
    probe.add_argument("--framework", choices=FRAMEWORKS, required=True)
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--data-directory", type=Path, required=True)
    probe.add_argument("--run-directory", type=Path, required=True)
    probe.add_argument("--report", type=Path, required=True)
    probe.add_argument("--steps", type=int, default=3)
    probe.add_argument("--seed", type=int, default=7)
    probe.add_argument("--global-batch-size", type=int, default=128)
    probe.add_argument("--micro-batch-size", type=int, default=4)
    probe.add_argument("--padding", choices=("dynamic", "static"), default="static")

    kernel = subparsers.add_parser("kernel-probe")
    kernel.add_argument("--framework", choices=FRAMEWORKS, required=True)
    kernel.add_argument("--checkpoint", type=Path, required=True)
    kernel.add_argument("--data-directory", type=Path, required=True)
    kernel.add_argument("--report", type=Path, required=True)
    kernel.add_argument("--iterations", type=int, default=5)
    kernel.add_argument("--seed", type=int, default=7)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--trl-python", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=4)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--gpu", type=int, default=2)
    pair.add_argument("--padding", choices=("dynamic", "static"), default="static")

    verify = subparsers.add_parser("verify-export")
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        manifest = prepare_data(
            arguments.output,
            checkpoint=arguments.checkpoint,
            cache_directory=arguments.cache_directory,
            train_parquet=arguments.train_parquet,
            test_parquet=arguments.test_parquet,
            training_rows=arguments.training_rows,
            evaluation_rows=arguments.evaluation_rows,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif arguments.command == "worker":
        _worker(arguments)
    elif arguments.command == "probe":
        _probe_worker(arguments)
    elif arguments.command == "kernel-probe":
        _kernel_worker(arguments)
    elif arguments.command == "verify-export":
        _verify_huggingface_export(arguments.report, arguments.output)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()
