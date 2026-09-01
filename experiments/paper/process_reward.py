"""Matched Math-Shepherd process-reward preflight against TRL PRMTrainer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
TEXT_REWARD_MANIFEST = ROOT / "benchmarks/configs/paper-text-reward-v1.json"
FRAMEWORKS = ("representax", "trl")
STEPS_PER_TRAJECTORY = 4
EXECUTION_SEQUENCE_LENGTH = 256
SEQUENCE_LENGTH_BUCKETS = (256, 512, 1024, 2048)
MICRO_BATCH_SIZE = 2
EVALUATION_BATCH_SIZE = 8
STEP_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    dataset_id: str
    dataset_revision: str
    batch_size: int
    maximum_length: int
    objective: str
    reference_version: str


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract() -> FrozenContract:
    """Resolve the process-reward identities frozen in the paper manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(TEXT_REWARD_MANIFEST)
    campaign_row = next(
        row for row in campaign["workloads"] if row["name"] == "process-reward"
    )
    panel_row = next(
        row for row in panel["workloads"] if row["name"] == "process-reward"
    )
    if campaign_row["frameworks"] != ["representax", "trl"]:
        raise ValueError("unexpected process-reward frameworks")
    if panel_row["reference"] != "trl":
        raise ValueError("unexpected process-reward reference")
    model = panel["models"][panel_row["model"]]
    dataset = panel["datasets"][panel_row["train"][0]]
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        dataset_id=dataset["repo_id"],
        dataset_revision=dataset["revision"],
        batch_size=int(campaign_row["global_batch"]),
        maximum_length=int(campaign_row["maximum_sequence_length"]),
        objective=str(panel_row["objective"]),
        reference_version=str(panel["references"][panel_row["reference"]]),
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


def tokenize_trajectory(
    tokenizer: Any,
    prompt: str,
    completions: Sequence[str],
    *,
    step_separator: str = STEP_SEPARATOR,
) -> tuple[list[int], list[int]]:
    """Reproduce PRMTrainer tokenization and return supervised token positions."""

    input_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if tokenizer.bos_token_id is not None:
        input_ids.insert(0, int(tokenizer.bos_token_id))
    separator_ids = list(tokenizer.encode(step_separator, add_special_tokens=False))
    if not separator_ids:
        raise ValueError("the process-reward step separator must tokenize")
    step_positions = []
    for completion in completions:
        input_ids.extend(
            tokenizer(str(completion), add_special_tokens=False)["input_ids"]
        )
        input_ids.extend(separator_ids)
        step_positions.append(len(input_ids) - 1)
    return input_ids, step_positions


def _select_rows(
    dataset: Any,
    tokenizer: Any,
    *,
    count: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for source_index, source in enumerate(dataset):
        completions = tuple(str(value) for value in source["completions"])
        labels = tuple(bool(value) for value in source["labels"])
        if len(completions) != STEPS_PER_TRAJECTORY or len(labels) != len(completions):
            continue
        if not any(labels) or all(labels):
            continue
        input_ids, _ = tokenize_trajectory(
            tokenizer, str(source["prompt"]), completions
        )
        if len(input_ids) > EXECUTION_SEQUENCE_LENGTH:
            continue
        rows.append(
            {
                "source_index": source_index,
                "prompt": str(source["prompt"]),
                "completions": list(completions),
                "labels": list(labels),
                "token_count": len(input_ids),
            }
        )
        if len(rows) == count:
            return tuple(rows)
    raise ValueError(f"Math-Shepherd contains only {len(rows)} admitted rows")


def prepare_data(
    output: Path,
    *,
    checkpoint: Path,
    cache_directory: Path,
    training_rows: int,
    evaluation_rows: int,
) -> dict[str, Any]:
    """Materialize deterministic, disjoint official train/test preflight slices."""

    import datasets
    from transformers import AutoTokenizer

    contract = frozen_contract()
    output.mkdir(parents=True, exist_ok=False)
    source = datasets.load_dataset(
        contract.dataset_id,
        revision=contract.dataset_revision,
        cache_dir=str(cache_directory),
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    train = _select_rows(source["train"], tokenizer, count=training_rows)
    evaluation = _select_rows(source["test"], tokenizer, count=evaluation_rows)
    files = {}
    for name, rows in (("train", train), ("evaluation", evaluation)):
        path = output / f"{name}.jsonl"
        count = _write_jsonl(path, rows)
        files[name] = {
            "path": path.name,
            "rows": count,
            "sha256": _sha256(path),
            "source_indices": [row["source_index"] for row in rows],
            "minimum_tokens": min(row["token_count"] for row in rows),
            "maximum_tokens": max(row["token_count"] for row in rows),
        }
    manifest = {
        "schema_version": "representax-process-reward-preflight-data-v1",
        "dataset": {
            "repo_id": contract.dataset_id,
            "revision": contract.dataset_revision,
            "training_split": "train",
            "evaluation_split": "test",
        },
        "selection": {
            "steps_per_trajectory": STEPS_PER_TRAJECTORY,
            "requires_mixed_labels": True,
            "maximum_tokens": EXECUTION_SEQUENCE_LENGTH,
            "order": "first admitted rows in source order",
        },
        "files": files,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def make_process_reward_processor(
    checkpoint: str | Path,
    *,
    sequence_length_buckets: Sequence[int] = SEQUENCE_LENGTH_BUCKETS,
) -> Any:
    """Build the Qwen processor that records every supervised step position."""

    import jax.numpy as jnp
    from transformers import AutoTokenizer

    from representax.core import Route
    from representax.models.processing import Processor, select_static_shape_bucket
    from representax.models.qwen_reranker import QwenRerankerBatch

    root = Path(checkpoint).resolve()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    buckets = tuple(int(value) for value in sequence_length_buckets)

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> Mapping[str, Any]:
        del route, seed
        encoded = []
        positions = []
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise TypeError("process-reward inputs must be trajectory mappings")
            completions = tuple(str(value) for value in artifact["completions"])
            if len(completions) != STEPS_PER_TRAJECTORY:
                raise ValueError("process-reward trajectories must contain four steps")
            ids, step_positions = tokenize_trajectory(
                tokenizer, str(artifact["prompt"]), completions
            )
            encoded.append(ids)
            positions.append(step_positions)
        required = max(len(value) for value in encoded)
        bucket = select_static_shape_bucket(
            (required,), tuple((size,) for size in buckets)
        )[0]
        input_ids = np.full(
            (len(encoded), bucket), tokenizer.pad_token_id, dtype=np.int32
        )
        attention_mask = np.zeros(input_ids.shape, dtype=np.int32)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        return {
            "tokens": QwenRerankerBatch(
                input_ids=jnp.asarray(input_ids),
                attention_mask=jnp.asarray(attention_mask),
            ),
            "step_positions": jnp.asarray(positions, dtype=jnp.int32),
        }

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-qwen-process-reward-processor-v1",
            "checkpoint": str(root),
            "step_separator": STEP_SEPARATOR,
            "steps_per_trajectory": STEPS_PER_TRAJECTORY,
            "sequence_length_buckets": list(buckets),
            "supervision": "last separator token of each step",
        },
    )


def load_process_reward_model(
    model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = True,
    parameter_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    sequence_length_buckets: Sequence[int] = SEQUENCE_LENGTH_BUCKETS,
    rematerialization: str = "full",
) -> tuple[Any, Any]:
    """Load the frozen scalar Qwen head and expose it at step positions."""

    import equinox as eqx
    import jax.numpy as jnp

    from representax.models.qwen_reward import load_qwen_reward_model

    reward, _ = load_qwen_reward_model(
        model_name_or_path,
        revision=revision,
        local_files_only=local_files_only,
        parameter_dtype=jnp.dtype(parameter_dtype),
        compute_dtype=jnp.dtype(compute_dtype),
        sequence_length_buckets=sequence_length_buckets,
        rematerialization=rematerialization,
    )

    class QwenProcessRewardModel(eqx.Module):
        reward: Any

        def logits(self, inputs: Mapping[str, Any], *, key: Any = None) -> Any:
            hidden = self.reward.backbone.hidden_states(inputs["tokens"], key=key)
            scores = self.reward.score_head(
                hidden.astype(self.reward.score_head.weight.dtype)
            )[..., 0].astype(jnp.float32)
            return jnp.take_along_axis(scores, inputs["step_positions"], axis=1)

    return (
        QwenProcessRewardModel(reward),
        make_process_reward_processor(
            model_name_or_path,
            sequence_length_buckets=sequence_length_buckets,
        ),
    )


class ProcessRewardPaperCollator:
    """Collate the fixed four-step paper slice into the native reward batch."""

    def __init__(self, *, processor: Any) -> None:
        self.processor = processor

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-process-reward-paper-collator-v1",
            "processor": self.processor.data_contract(),
            "steps_per_trajectory": STEPS_PER_TRAJECTORY,
        }

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.tasks.reward_modeling import ProcessRewardBatch

        labels = np.asarray([row["labels"] for row in examples], dtype=np.float32)
        if labels.shape != (len(examples), STEPS_PER_TRAJECTORY):
            raise ValueError("paper process-reward labels must have four steps")
        artifacts = [
            {"prompt": row["prompt"], "completions": row["completions"]}
            for row in examples
        ]
        return ProcessRewardBatch(
            inputs=self.processor(artifacts),
            labels=jnp.asarray(labels),
            valid=jnp.ones(labels.shape, dtype=jnp.bool_),
        )


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
        ProcessRewardConfig,
        ProcessRewardLossConfig,
    )

    contract = frozen_contract()
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
    if contract.batch_size % MICRO_BATCH_SIZE:
        raise ValueError("frozen batch must be divisible by the microbatch")

    def data(path: Path, *, evaluation: bool = False) -> DataConfig:
        return DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=ComponentConfig(
                target=("experiments.paper.process_reward:ProcessRewardPaperCollator")
            ),
            drop_remainder=not evaluation,
            num_threads=0,
            prefetch_buffer_size=2,
        )

    return JobConfig(
        name="paper-preflight-process-reward",
        model=ModelConfig(
            target="experiments.paper.process_reward:load_process_reward_model",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": list(SEQUENCE_LENGTH_BUCKETS),
                "rematerialization": "full",
            },
        ),
        task=ProcessRewardConfig(),
        loss=ProcessRewardLossConfig(objective="binary_cross_entropy"),
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
            global_batch_size=contract.batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(
                micro_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=contract.batch_size // MICRO_BATCH_SIZE,
            ),
            activation_rematerialization="full",
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
            batch_size=EVALUATION_BATCH_SIZE,
            evaluators=(RewardEvaluatorConfig(name="math_shepherd", mode="process"),),
            on_start=True,
            on_end=True,
            primary_metric="valid/math_shepherd/accuracy",
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(selection="final"),
    )


def representax_steady_state(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, float]:
    """Compute warm throughput without compilation-and-first-step rows."""

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
        "examples_per_second": batch_size * len(seconds) / sum(seconds),
    }


def _probe_artifacts(data_directory: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"prompt": row["prompt"], "completions": row["completions"]}
        for row in _read_jsonl(data_directory / "evaluation.jsonl")[:2]
    )


def _representax_scores(model: Any, processor: Any, artifacts: Sequence[Any]) -> Any:
    import jax

    from representax.config import PrecisionConfig
    from representax.core import score_logits
    from representax.precision import precision_context, resolve_precision_policy

    with precision_context(resolve_precision_policy(PrecisionConfig.bfloat16_mixed())):
        scores = score_logits(model, processor(artifacts))
    jax.block_until_ready(scores)
    return np.asarray(scores)


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
        raise RuntimeError("Representax did not export a native inference bundle")

    _, processor = load_process_reward_model(
        checkpoint,
        revision=contract.model_revision,
        local_files_only=True,
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        sequence_length_buckets=SEQUENCE_LENGTH_BUCKETS,
        rematerialization="full",
    )
    artifacts = _probe_artifacts(data_directory)
    final_probe = _representax_scores(completed.state.model, processor, artifacts)
    reloaded, restored_job = load_inference_bundle(completed.inference_bundle)
    reload_probe = _representax_scores(reloaded, processor, artifacts)
    if not np.array_equal(final_probe, reload_probe):
        raise RuntimeError("Representax native reload changed step logits")

    rows = _read_jsonl(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    compile_seconds = sum(
        float(row["metrics"].get("perf/compilation_and_first_step_seconds", 0.0))
        for row in training
    )
    return {
        "schema_version": "representax-process-reward-worker-v1",
        "framework": "representax",
        "steps": steps,
        "batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "execution_sequence_length": EXECUTION_SEQUENCE_LENGTH,
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
        "native_reload_exact": True,
        "reload_job_name": restored_job.name,
        "probe_scores": final_probe.tolist(),
        "peak_device_bytes": int(
            (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        ),
    }


def _trl_worker(
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import transformers
    import trl
    from datasets import Dataset
    from safetensors.torch import load_file, save_file
    from torch.utils.data import SequentialSampler
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        TrainerCallback,
    )
    from transformers.modeling_outputs import TokenClassifierOutput
    from trl.experimental.prm import PRMConfig, PRMTrainer

    contract = frozen_contract()
    if trl.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected trl=={contract.reference_version}, found {trl.__version__}"
        )
    run_directory.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    class ScalarStepClassifier(torch.nn.Module):
        """Expose the frozen one-row scalar head as equivalent binary logits."""

        def __init__(self) -> None:
            super().__init__()
            self.sequence = AutoModelForSequenceClassification.from_pretrained(
                checkpoint,
                local_files_only=True,
                dtype=torch.float32,
            )
            self.config = self.sequence.config

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            labels: Any = None,
        ) -> Any:
            outputs = self.sequence.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            scores = self.sequence.score(outputs.last_hidden_state)[..., 0]
            logits = torch.stack((-0.5 * scores, 0.5 * scores), dim=-1)
            loss = None
            if labels is not None:
                loss = functional.cross_entropy(
                    logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100
                )
            return TokenClassifierOutput(loss=loss, logits=logits)

    class SequentialPRMTrainer(PRMTrainer):
        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            return SequentialSampler(train_dataset or self.train_dataset)

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.started: float | None = None
            self.rows: list[dict[str, float | int]] = []

        def on_step_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args, state
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            torch.cuda.synchronize()
            if self.started is None:
                raise RuntimeError("TRL step timer ended without starting")
            self.rows.append(
                {
                    "step": int(state.global_step),
                    "seconds": time.perf_counter() - self.started,
                }
            )
            return control

    class StopAtMidpoint(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            del args
            if state.global_step >= steps // 2:
                control.should_training_stop = True
            return control

    def dataset(path: Path) -> Dataset:
        rows = _read_jsonl(path)
        return Dataset.from_list(
            [
                {
                    "prompt": row["prompt"],
                    "completions": row["completions"],
                    "labels": row["labels"],
                }
                for row in rows
            ]
        )

    train = dataset(data_directory / "train.jsonl")
    evaluation = dataset(data_directory / "evaluation.jsonl")
    collator = DataCollatorForTokenClassification(
        tokenizer,
        padding="max_length",
        max_length=EXECUTION_SEQUENCE_LENGTH,
    )

    def arguments() -> PRMConfig:
        return PRMConfig(
            output_dir=str(run_directory / "checkpoints"),
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            gradient_accumulation_steps=contract.batch_size // MICRO_BATCH_SIZE,
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
            seed=seed,
            data_seed=seed,
            max_length=contract.maximum_length,
            step_separator=STEP_SEPARATOR,
        )

    first_model = ScalarStepClassifier()
    first_timer = StepTimer()
    first = SequentialPRMTrainer(
        model=first_model,
        args=arguments(),
        data_collator=collator,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        callbacks=[first_timer, StopAtMidpoint()],
    )
    if first.model_accepts_loss_kwargs:
        raise RuntimeError("TRL wrapper must use Trainer-owned loss accumulation")
    initial_evaluation = first.evaluate()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    first.train()
    midpoint = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
    if not midpoint.is_dir():
        raise RuntimeError("TRL did not write its midpoint checkpoint")
    del first, first_model
    gc.collect()
    torch.cuda.empty_cache()

    final_model = ScalarStepClassifier()
    final_timer = StepTimer()
    final = SequentialPRMTrainer(
        model=final_model,
        args=arguments(),
        data_collator=collator,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        callbacks=[final_timer],
    )
    if final.model_accepts_loss_kwargs:
        raise RuntimeError("TRL wrapper must use Trainer-owned loss accumulation")
    output = final.train(resume_from_checkpoint=str(midpoint))
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    if final.state.global_step != steps:
        raise RuntimeError("TRL did not resume to the requested update count")
    final_evaluation = final.evaluate()

    export = run_directory / "final-model"
    export.mkdir()
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in final_model.state_dict().items()
    }
    save_file(state, export / "model.safetensors")
    tokenizer.save_pretrained(export)
    _write_json(
        export / "wrapper.json",
        {
            "schema_version": "representax-paper-scalar-step-classifier-v1",
            "source_checkpoint": str(checkpoint),
            "binary_logits": "[-score / 2, score / 2]",
        },
    )

    def probe(model: Any) -> np.ndarray:
        rows = _read_jsonl(data_directory / "evaluation.jsonl")[:2]
        input_ids = np.full(
            (len(rows), EXECUTION_SEQUENCE_LENGTH),
            tokenizer.pad_token_id,
            dtype=np.int64,
        )
        attention = np.zeros(input_ids.shape, dtype=np.int64)
        positions = []
        for index, row in enumerate(rows):
            ids, row_positions = tokenize_trajectory(
                tokenizer, row["prompt"], row["completions"]
            )
            input_ids[index, : len(ids)] = ids
            attention[index, : len(ids)] = 1
            positions.append(row_positions)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=torch.as_tensor(input_ids, device="cuda"),
                attention_mask=torch.as_tensor(attention, device="cuda"),
            ).logits
            scalar = logits[..., 1] - logits[..., 0]
            gathered = torch.gather(
                scalar,
                1,
                torch.as_tensor(positions, device="cuda"),
            )
        return gathered.float().cpu().numpy()

    final_probe = probe(final_model)
    reloaded = ScalarStepClassifier()
    reloaded.load_state_dict(load_file(export / "model.safetensors"), strict=True)
    reloaded.to("cuda")
    reloaded.eval()
    reload_probe = probe(reloaded)
    if not np.array_equal(final_probe, reload_probe):
        raise RuntimeError("TRL export reload changed step logits")

    timing = [*first_timer.rows, *final_timer.rows]
    timing.sort(key=lambda row: int(row["step"]))
    warm = [float(row["seconds"]) for row in timing[1:]]
    return {
        "schema_version": "representax-process-reward-worker-v1",
        "framework": "trl",
        "framework_version": trl.__version__,
        "transformers_version": transformers.__version__,
        "steps": steps,
        "batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "execution_sequence_length": EXECUTION_SEQUENCE_LENGTH,
        "precision": "bfloat16-autocast-float32-parameters",
        "training_seconds": training_seconds,
        "compilation_and_first_step_seconds": 0.0,
        "first_step_seconds": float(timing[0]["seconds"]),
        "steady_state": {
            "measured_steps": len(warm),
            "median_step_seconds": statistics.median(warm),
            "examples_per_second": contract.batch_size * len(warm) / sum(warm),
        },
        "step_timings": timing,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "training_metrics": output.metrics,
        "resumed": True,
        "checkpoint_iterations": [steps // 2, steps],
        "inference_bundle": str(export),
        "reload_parameters_exact": True,
        "reload_probe_exact": True,
        "probe_scores": final_probe.tolist(),
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _worker(arguments: argparse.Namespace) -> None:
    worker = (
        _representax_worker if arguments.framework == "representax" else _trl_worker
    )
    report = worker(
        checkpoint=arguments.checkpoint,
        data_directory=arguments.data_directory,
        run_directory=arguments.run_directory,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    _write_json(arguments.report, report)


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
    executables = {
        "representax": str(arguments.representax_python),
        "trl": str(arguments.reference_python),
    }
    for framework in FRAMEWORKS:
        report = output / f"{framework}.json"
        command = [
            executables[framework],
            "-m",
            "experiments.paper.process_reward",
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
                "PYTHONPATH": str(ROOT),
                "HF_HOME": str(arguments.hf_home),
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
        _run_process(
            command,
            environment=environment,
            log=output / f"{framework}.log",
        )
        reports[framework] = _document(report)
    summary = {
        "schema_version": "representax-process-reward-preflight-v1",
        "contract": {
            **asdict(frozen_contract()),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "gpu": arguments.gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
            "scalar_binary_equivalence": (
                "CE([-score/2, score/2], label) == BCEWithLogits(score, label)"
            ),
        },
        **reports,
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
        "--cache-directory",
        type=Path,
        default=Path("/raid/.cache/huggingface/datasets"),
    )
    prepare.add_argument("--training-rows", type=int, default=256)
    prepare.add_argument("--evaluation-rows", type=int, default=32)

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
    pair.add_argument("--gpu", type=int, default=3)
    pair.add_argument("--representax-python", type=Path, required=True)
    pair.add_argument("--reference-python", type=Path, required=True)
    pair.add_argument("--hf-home", type=Path, default=Path("/raid/.cache/huggingface"))
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(
            json.dumps(
                prepare_data(
                    arguments.output,
                    checkpoint=arguments.checkpoint,
                    cache_directory=arguments.cache_directory,
                    training_rows=arguments.training_rows,
                    evaluation_rows=arguments.evaluation_rows,
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
