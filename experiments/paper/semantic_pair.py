"""Short matched preflights for the paper's labeled-pair workloads."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.paper.provenance import reference_source, write_reference_result

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_MANIFEST = ROOT / "benchmarks/configs/paper-campaign-v1.json"
TEXT_MANIFEST = ROOT / "benchmarks/configs/paper-text-reward-v1.json"

Workload = Literal["semantic-similarity", "pair-classification"]
FRAMEWORKS = ("representax", "sentence-transformers")
MICRO_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class FrozenContract:
    model_id: str
    model_revision: str
    batch_size: int
    maximum_length: int
    reference_version: str
    datasets: Mapping[str, Mapping[str, Any]]


class PairEvaluationCollator:
    """Build fixed-size binary pair batches for configured evaluation."""

    def __init__(
        self,
        *,
        processor: Any,
        pad_to_size: int,
        left_field: str = "sentence1",
        right_field: str = "sentence2",
        label_field: str = "label",
    ) -> None:
        if pad_to_size <= 0:
            raise ValueError("pad_to_size must be positive")
        self.processor = processor
        self.pad_to_size = pad_to_size
        self.left_field = left_field
        self.right_field = right_field
        self.label_field = label_field

    def data_contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "representax-paper-pair-evaluation-v1",
            "processor": self.processor.data_contract(),
            "pad_to_size": self.pad_to_size,
            "left_field": self.left_field,
            "right_field": self.right_field,
            "label_field": self.label_field,
        }

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> Any:
        import jax.numpy as jnp

        from representax.tasks.classification import pair_classification_batch

        if len(rows) > self.pad_to_size:
            raise ValueError("pair evaluation batch exceeds pad_to_size")
        padding = self.pad_to_size - len(rows)
        left = tuple(str(row[self.left_field]) for row in rows) + ("",) * padding
        right = tuple(str(row[self.right_field]) for row in rows) + ("",) * padding
        labels = tuple(int(row[self.label_field]) for row in rows) + (0,) * padding
        return pair_classification_batch(
            left=self.processor(left),
            right=self.processor(right),
            labels=jnp.asarray(labels, dtype=jnp.int32),
            valid=jnp.asarray((True,) * len(rows) + (False,) * padding),
        )


def _document(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_contract(workload: Workload) -> FrozenContract:
    """Resolve and validate the identities frozen across the two manifests."""

    campaign = _document(CAMPAIGN_MANIFEST)
    panel = _document(TEXT_MANIFEST)
    campaign_row = next(row for row in campaign["workloads"] if row["name"] == workload)
    panel_row = next(row for row in panel["workloads"] if row["name"] == workload)
    if campaign_row["frameworks"] != ["representax", "sentence-transformers"]:
        raise ValueError(f"unexpected frameworks for {workload}")
    if panel_row["reference"] != "sentence-transformers":
        raise ValueError(f"unexpected reference for {workload}")
    model = panel["models"][panel_row["model"]]
    reference = reference_source(panel_row["reference"])
    if reference.release is None:
        raise ValueError("the labeled-pair reference requires a release")
    dataset_names = tuple(dict.fromkeys((*panel_row["train"], *panel_row["evaluate"])))
    return FrozenContract(
        model_id=model["repo_id"],
        model_revision=model["revision"],
        batch_size=int(campaign_row["global_batch"]),
        maximum_length=int(campaign_row["maximum_sequence_length"]),
        reference_version=reference.release,
        datasets={name: panel["datasets"][name] for name in dataset_names},
    )


def execution_batch_size(workload: Workload, contract: FrozenContract) -> int:
    """Use the frozen scientific batch for both labeled-pair workloads."""

    del workload
    return contract.batch_size


def normalize_sts_score(score: float) -> float:
    """Map the frozen STS 0-5 scale onto the cosine regression target range."""

    value = float(score)
    if not 0.0 <= value <= 5.0:
        raise ValueError(f"STS score is outside [0, 5]: {value}")
    return value / 5.0


def sprint_preflight_rows(
    packed: Mapping[str, Sequence[Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build deterministic, disjoint train and readiness-evaluation subsets."""

    left = packed["sent1"]
    right = packed["sent2"]
    labels = packed["labels"]
    if not (len(left) == len(right) == len(labels)):
        raise ValueError("Sprint packed columns must have equal lengths")
    positive = [index for index, label in enumerate(labels) if int(label) == 1]
    negative = [index for index, label in enumerate(labels) if int(label) == 0]
    if len(positive) < 1_000 or len(negative) < 3_248:
        raise ValueError("Sprint source does not contain the frozen class populations")

    train_positive = positive[:800]
    train_negative = negative[:1_248]
    evaluation_positive = positive[800:1_000]
    evaluation_negative = negative[1_248:3_248]

    def interleave(first: Sequence[int], second: Sequence[int]) -> tuple[int, ...]:
        output: list[int] = []
        for offset in range(max(len(first), len(second))):
            if offset < len(first):
                output.append(first[offset])
            if offset < len(second):
                output.append(second[offset])
        return tuple(output)

    def records(indices: Sequence[int]) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "source_index": int(index),
                "sentence1": str(left[index]),
                "sentence2": str(right[index]),
                "label": int(labels[index]),
            }
            for index in indices
        )

    return (
        records(interleave(train_positive, train_negative)),
        records(interleave(evaluation_positive, evaluation_negative)),
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


def prepare_data(output: Path) -> dict[str, Any]:
    """Materialize small line-oriented views of the pinned source datasets."""

    import datasets

    output.mkdir(parents=True, exist_ok=False)
    semantic = frozen_contract("semantic-similarity")
    pair = frozen_contract("pair-classification")
    files: dict[str, Any] = {}

    stsb = semantic.datasets["stsb-v2"]
    for split in ("train", "validation"):
        source = datasets.load_dataset(
            stsb["repo_id"], revision=stsb["revision"], split=split
        )
        path = output / f"semantic-{split}.jsonl"
        count = _write_jsonl(
            path,
            (
                {
                    "source_index": index,
                    "sentence1": str(row["sentence1"]),
                    "sentence2": str(row["sentence2"]),
                    "label": normalize_sts_score(row["score"]),
                }
                for index, row in enumerate(source)
            ),
        )
        files[path.name] = {"rows": count, "sha256": _sha256(path)}

    sick = semantic.datasets["sick-r"]
    source = datasets.load_dataset(
        sick["repo_id"], revision=sick["revision"], split="test"
    )
    path = output / "semantic-sick-r.jsonl"
    count = _write_jsonl(
        path,
        (
            {
                "source_index": index,
                "sentence1": str(row["sentence1"]),
                "sentence2": str(row["sentence2"]),
                "label": normalize_sts_score(row["score"]),
            }
            for index, row in enumerate(source)
        ),
    )
    files[path.name] = {"rows": count, "sha256": _sha256(path)}

    sprint = pair.datasets["sprint-duplicate"]
    packed = datasets.load_dataset(
        sprint["repo_id"], revision=sprint["revision"], split="test"
    )[0]
    training_rows, evaluation_rows = sprint_preflight_rows(packed)
    for name, rows in (
        ("pair-train", training_rows),
        ("pair-evaluation", evaluation_rows),
    ):
        path = output / f"{name}.jsonl"
        count = _write_jsonl(path, rows)
        files[path.name] = {"rows": count, "sha256": _sha256(path)}

    manifest = {
        "schema_version": "representax-semantic-pair-preflight-data-v1",
        "contracts": {
            "semantic-similarity": {
                "model": semantic.model_id,
                "datasets": semantic.datasets,
                "sts_label_transform": "score / 5.0",
            },
            "pair-classification": {
                "model": pair.model_id,
                "datasets": pair.datasets,
                "training_source_indices": "800 positive + 1248 negative, interleaved",
                "evaluation_source_indices": (
                    "held-out 200 positive + 2000 negative, interleaved"
                ),
            },
        },
        "files": files,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def _data_paths(workload: Workload, data_directory: Path) -> tuple[Path, Path]:
    if workload == "semantic-similarity":
        return (
            data_directory / "semantic-train.jsonl",
            data_directory / "semantic-validation.jsonl",
        )
    return (
        data_directory / "pair-train.jsonl",
        data_directory / "pair-evaluation.jsonl",
    )


def _representax_job(
    workload: Workload,
    *,
    checkpoint: Path,
    data_directory: Path,
    steps: int,
    seed: int,
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
        JobConfig,
        LoggingConfig,
        ModelConfig,
        OptimizationConfig,
        PairClassificationEvaluatorConfig,
        PrecisionConfig,
        SimilarityEvaluatorConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.pairwise import (
        ContrastiveConfig,
        CosineRegressionConfig,
        PairwiseConfig,
    )

    contract = frozen_contract(workload)
    training_batch_size = execution_batch_size(workload, contract)
    train_path, evaluation_path = _data_paths(workload, data_directory)
    if training_batch_size % MICRO_BATCH_SIZE:
        raise ValueError("global batch must be divisible by the preflight microbatch")

    def data(path: Path, *, evaluation: bool = False) -> DataConfig:
        if evaluation and workload == "pair-classification":
            collator = ComponentConfig(
                target="experiments.paper.semantic_pair:PairEvaluationCollator",
                parameters={"pad_to_size": MICRO_BATCH_SIZE},
            )
        else:
            collator = ComponentConfig(
                target="representax.tasks.pairwise:PairwiseCollator",
                parameters={
                    "label_field": "label",
                    **({"pad_to_size": MICRO_BATCH_SIZE} if evaluation else {}),
                },
            )
        return DataConfig(
            distribution=mix(source(str(path), map=identity), shuffle=False),
            collate=collator,
            drop_remainder=not evaluation,
            num_threads=0,
            prefetch_buffer_size=2,
        )

    schedule = ComponentConfig(
        target="optax.warmup_cosine_decay_schedule",
        parameters={
            "init_value": 0.0,
            "peak_value": 2e-5,
            "warmup_steps": 1,
            "decay_steps": steps,
            "end_value": 0.0,
        },
    )
    evaluator: Any
    primary_metric: str
    if workload == "semantic-similarity":
        loss = CosineRegressionConfig()
        evaluator = SimilarityEvaluatorConfig(
            name="stsb",
            similarity_functions=("cosine",),
            main_similarity="cosine",
        )
        primary_metric = "valid/stsb/spearman_cosine"
    else:
        loss = ContrastiveConfig(distance="cosine", margin=0.5, mining="all")
        evaluator = PairClassificationEvaluatorConfig(
            name="sprint",
            similarity_functions=("cosine",),
        )
        primary_metric = "valid/sprint/average_precision_max"

    return JobConfig(
        name=f"paper-preflight-{workload}",
        model=ModelConfig(
            target="representax.models:SentenceEncoder.load_from_hf",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": contract.model_revision,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": (contract.maximum_length,),
            },
        ),
        task=PairwiseConfig(),
        loss=loss,
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
            schedule=schedule,
            max_gradient_norm=1.0,
        ),
        data=data(train_path),
        training=TrainingConfig(
            global_batch_size=training_batch_size,
            max_steps=steps,
            seed=seed,
            batch=BatchConfig(
                micro_batch_size=MICRO_BATCH_SIZE,
                gradient_accumulation_steps=training_batch_size // MICRO_BATCH_SIZE,
            ),
            activation_rematerialization="none",
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
            data=data(evaluation_path, evaluation=True),
            batch_size=MICRO_BATCH_SIZE,
            evaluators=(evaluator,),
            on_start=True,
            on_end=True,
            primary_metric=primary_metric,
            primary_metric_mode="max",
            save_best=False,
        ),
        export=ExportConfig(selection="final"),
    )


def _model_and_processor(checkpoint: Path, contract: FrozenContract) -> tuple[Any, Any]:
    from representax.models import SentenceEncoder

    return SentenceEncoder.load_from_hf(
        checkpoint,
        revision=contract.model_revision,
        local_files_only=True,
        parameter_dtype="float32",
        compute_dtype="bfloat16",
        sequence_length_buckets=(contract.maximum_length,),
    )


def _pair_probe(
    model: Any, processor: Any, rows: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    import jax

    from representax.config import PrecisionConfig
    from representax.core import Route, encode
    from representax.precision import precision_context, resolve_precision_policy
    from representax.tasks.pairwise import PairwiseCollator

    batch = PairwiseCollator(processor=processor, label_field="label")(rows)
    precision = resolve_precision_policy(
        PrecisionConfig.bfloat16_mixed(),
    )
    with precision_context(precision):
        values = (
            encode(model, batch.left, route=Route.GENERIC),
            encode(model, batch.right, route=Route.GENERIC),
        )
    jax.block_until_ready(values)
    return np.concatenate([np.asarray(value).reshape(-1) for value in values])


def _similarity_metrics(
    model: Any,
    processor: Any,
    path: Path,
    batch_size: int,
    *,
    name: str,
) -> dict[str, float]:
    from representax.evaluation import SimilarityEvaluator
    from representax.tasks.pairwise import PairwiseCollator
    from representax.train import EvaluationRunner

    rows = _read_jsonl(path)
    collator = PairwiseCollator(
        processor=processor,
        label_field="label",
        pad_to_size=batch_size,
    )

    def batches() -> Iterable[Any]:
        for start in range(0, len(rows), batch_size):
            yield collator(rows[start : start + batch_size])

    result = EvaluationRunner(
        SimilarityEvaluator(
            name=name,
            similarity_functions=("cosine",),
            main_similarity="cosine",
        )
    ).run(model, batches())
    return {name: float(value) for name, value in result.metrics.items()}


def _representax_similarity_point(
    update: int,
    stsb: Mapping[str, float],
    sick_r: Mapping[str, float],
) -> dict[str, float | int]:
    return {
        "update": update,
        "stsb_spearman_cosine": stsb["valid/stsb/spearman_cosine"],
        "sick_r_spearman_cosine": sick_r["valid/sick_r/spearman_cosine"],
    }


def _reference_similarity_point(
    update: int,
    metrics: Mapping[str, float],
) -> dict[str, float | int]:
    return {
        "update": update,
        "stsb_spearman_cosine": metrics["stsb_spearman_cosine"],
        "sick_r_spearman_cosine": metrics["sick_r_spearman_cosine"],
    }


def _metric_rows(path: Path) -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def representax_steady_state(
    rows: Sequence[Mapping[str, Any]], batch_size: int
) -> dict[str, float]:
    """Derive warm throughput from actual completed-step metrics."""

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
        "aggregate_examples_per_second": batch_size * len(seconds) / sum(seconds),
    }


def _representax_worker(
    workload: Workload,
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

    contract = frozen_contract(workload)
    training_batch_size = execution_batch_size(workload, contract)
    job = _representax_job(
        workload,
        checkpoint=checkpoint,
        data_directory=data_directory,
        steps=steps,
        seed=seed,
    )
    sick_initial = None
    similarity_curve = None
    processor = None
    if workload == "semantic-similarity":
        initial_model, processor = _model_and_processor(checkpoint, contract)
        stsb_initial = _similarity_metrics(
            initial_model,
            processor,
            data_directory / "semantic-validation.jsonl",
            MICRO_BATCH_SIZE,
            name="stsb",
        )
        sick_initial = _similarity_metrics(
            initial_model,
            processor,
            data_directory / "semantic-sick-r.jsonl",
            MICRO_BATCH_SIZE,
            name="sick_r",
        )
        similarity_curve = [
            _representax_similarity_point(0, stsb_initial, sick_initial)
        ]
        del initial_model
        gc.collect()
        jax.clear_caches()

    started = time.perf_counter()
    paused = run_job(job, run_directory, stop_after=steps // 2)
    if paused.completed_iterations != steps // 2:
        raise RuntimeError("Representax did not stop at the resumable midpoint")
    if workload == "semantic-similarity":
        assert processor is not None
        assert similarity_curve is not None
        midpoint_stsb = _similarity_metrics(
            paused.state.model,
            processor,
            data_directory / "semantic-validation.jsonl",
            MICRO_BATCH_SIZE,
            name="stsb",
        )
        midpoint_sick = _similarity_metrics(
            paused.state.model,
            processor,
            data_directory / "semantic-sick-r.jsonl",
            MICRO_BATCH_SIZE,
            name="sick_r",
        )
        similarity_curve.append(
            _representax_similarity_point(
                steps // 2,
                midpoint_stsb,
                midpoint_sick,
            )
        )
    del paused
    gc.collect()
    jax.clear_caches()
    completed = run_job(job, run_directory, resume=True)
    jax.block_until_ready(completed.state)
    elapsed = time.perf_counter() - started
    if not completed.resumed or completed.completed_iterations != steps:
        raise RuntimeError("Representax did not resume to the requested update count")
    if completed.inference_bundle is None:
        raise RuntimeError("Representax did not export the final inference bundle")

    if processor is None:
        _, processor = _model_and_processor(checkpoint, contract)
    evaluation_rows = _read_jsonl(_data_paths(workload, data_directory)[1])
    final_probe = _pair_probe(completed.state.model, processor, evaluation_rows[:8])
    reloaded_model, restored_job = load_inference_bundle(completed.inference_bundle)
    reload_probe = _pair_probe(reloaded_model, processor, evaluation_rows[:8])
    reload_maximum_difference = float(np.max(np.abs(final_probe - reload_probe)))
    reload_cosine = float(
        final_probe
        @ reload_probe
        / (np.linalg.norm(final_probe) * np.linalg.norm(reload_probe))
    )
    if not np.allclose(final_probe, reload_probe, rtol=2e-2, atol=2e-3):
        raise RuntimeError("Representax inference reload changed probe embeddings")

    sick_final = None
    if workload == "semantic-similarity":
        assert similarity_curve is not None
        stsb_final = _similarity_metrics(
            completed.state.model,
            processor,
            data_directory / "semantic-validation.jsonl",
            MICRO_BATCH_SIZE,
            name="stsb",
        )
        sick_final = _similarity_metrics(
            completed.state.model,
            processor,
            data_directory / "semantic-sick-r.jsonl",
            MICRO_BATCH_SIZE,
            name="sick_r",
        )
        similarity_curve.append(
            _representax_similarity_point(steps, stsb_final, sick_final)
        )

    rows = _metric_rows(run_directory / "metrics.jsonl")
    training = [row for row in rows if row.get("event") == "training_step"]
    evaluations = [row for row in rows if row.get("event") == "evaluation"]
    compile_seconds = sum(
        float(row["metrics"].get("perf/compilation_and_first_step_seconds", 0.0))
        for row in training
    )
    return {
        "schema_version": "representax-semantic-pair-worker-v1",
        "framework": "representax",
        "workload": workload,
        "steps": steps,
        "batch_size": training_batch_size,
        "campaign_batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "precision": "bfloat16-compute-float32-master",
        "elapsed_seconds": elapsed,
        "compilation_and_first_step_seconds": compile_seconds,
        "steady_state": representax_steady_state(training, training_batch_size),
        "initial_evaluation": evaluations[0]["metrics"],
        "final_evaluation": evaluations[-1]["metrics"],
        "sick_r_initial": sick_initial,
        "sick_r_final": sick_final,
        "evaluation_curve": similarity_curve,
        "final_training": training[-1]["metrics"],
        "resumed": completed.resumed,
        "checkpoint_iterations": [steps // 2, steps],
        "inference_bundle": str(completed.inference_bundle),
        "reload_job_name": restored_job.name,
        "reload_probe_maximum_absolute_difference": reload_maximum_difference,
        "reload_probe_cosine": reload_cosine,
        "peak_device_bytes": int(
            (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use", 0)
        ),
    }


def _fixed_length_collator(model: Any, maximum_length: int) -> Any:
    import torch
    import torch.nn.functional as functional
    from sentence_transformers.sentence_transformer.data_collator import (
        SentenceTransformerDataCollator,
    )

    class FixedLengthCollator(SentenceTransformerDataCollator):
        def __call__(self, features: Any) -> dict[str, Any]:
            batch = super().__call__(features)
            output = {}
            for name, value in batch.items():
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    output[name] = value
                    continue
                value = value[:, :maximum_length]
                padding = maximum_length - value.shape[1]
                fill = model.tokenizer.pad_token_id if name.endswith("input_ids") else 0
                output[name] = functional.pad(value, (0, padding), value=fill)
            return output

    return FixedLengthCollator(
        preprocess_fn=model.tokenize,
        valid_label_columns=["label", "score"],
    )


def _reference_evaluators(
    workload: Workload, data_directory: Path, batch_size: int
) -> tuple[Any, ...]:
    from sentence_transformers.sentence_transformer.evaluation import (
        BinaryClassificationEvaluator,
        EmbeddingSimilarityEvaluator,
    )

    if workload == "semantic-similarity":
        evaluators = []
        for name, path in (
            ("stsb", data_directory / "semantic-validation.jsonl"),
            ("sick_r", data_directory / "semantic-sick-r.jsonl"),
        ):
            rows = _read_jsonl(path)
            evaluators.append(
                EmbeddingSimilarityEvaluator(
                    [row["sentence1"] for row in rows],
                    [row["sentence2"] for row in rows],
                    [float(row["label"]) for row in rows],
                    batch_size=batch_size,
                    main_similarity="cosine",
                    similarity_fn_names=["cosine"],
                    name=name,
                    show_progress_bar=False,
                    write_csv=False,
                )
            )
        return tuple(evaluators)
    rows = _read_jsonl(data_directory / "pair-evaluation.jsonl")
    return (
        BinaryClassificationEvaluator(
            [row["sentence1"] for row in rows],
            [row["sentence2"] for row in rows],
            [int(row["label"]) for row in rows],
            name="sprint",
            batch_size=batch_size,
            show_progress_bar=False,
            write_csv=False,
            similarity_fn_names=["cosine"],
        ),
    )


def _run_reference_evaluation(
    evaluators: Sequence[Any], model: Any
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for evaluator in evaluators:
        metrics.update({name: float(value) for name, value in evaluator(model).items()})
    return metrics


def _sentence_transformers_worker(
    workload: Workload,
    *,
    checkpoint: Path,
    data_directory: Path,
    run_directory: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    if steps < 4 or steps % 2:
        raise ValueError("steps must be an even integer of at least four")
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
        ContrastiveLoss,
        CosineSimilarityLoss,
    )
    from sentence_transformers.sentence_transformer.losses.contrastive import (
        SiameseDistanceMetric,
    )

    contract = frozen_contract(workload)
    training_batch_size = execution_batch_size(workload, contract)
    if sentence_transformers.__version__ != contract.reference_version:
        raise RuntimeError(
            f"expected sentence-transformers=={contract.reference_version}, "
            f"found {sentence_transformers.__version__}"
        )
    model = SentenceTransformer(str(checkpoint), local_files_only=True)
    model.max_seq_length = contract.maximum_length
    train_path, _ = _data_paths(workload, data_directory)
    train_dataset = datasets.Dataset.from_json(str(train_path))
    removable = [
        name for name in ("source_index",) if name in train_dataset.column_names
    ]
    if removable:
        train_dataset = train_dataset.remove_columns(removable)
    evaluators = _reference_evaluators(workload, data_directory, MICRO_BATCH_SIZE)
    initial_evaluation = _run_reference_evaluation(evaluators, model)
    similarity_curve = (
        [_reference_similarity_point(0, initial_evaluation)]
        if workload == "semantic-similarity"
        else None
    )
    loss = (
        CosineSimilarityLoss(model)
        if workload == "semantic-similarity"
        else ContrastiveLoss(
            model,
            distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
            margin=0.5,
        )
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=training_batch_size // MICRO_BATCH_SIZE,
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
        data_collator=_fixed_length_collator(model, contract.maximum_length),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    final_evaluation = _run_reference_evaluation(evaluators, model)
    export = run_directory / "final-model"
    trainer.save_model(str(export))
    probe_rows = _read_jsonl(_data_paths(workload, data_directory)[1])[:8]
    final_probe = model.encode(
        [row["sentence1"] for row in probe_rows]
        + [row["sentence2"] for row in probe_rows],
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).reshape(-1)
    reloaded = SentenceTransformer(str(export), local_files_only=True)
    reload_probe = reloaded.encode(
        [row["sentence1"] for row in probe_rows]
        + [row["sentence2"] for row in probe_rows],
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).reshape(-1)
    reload_maximum_difference = float(np.max(np.abs(final_probe - reload_probe)))
    reload_cosine = float(
        final_probe
        @ reload_probe
        / (np.linalg.norm(final_probe) * np.linalg.norm(reload_probe))
    )
    if not np.allclose(final_probe, reload_probe, rtol=2e-2, atol=2e-3):
        raise RuntimeError("Sentence Transformers reload changed probe embeddings")
    checkpoints = sorted((run_directory / "checkpoints").glob("checkpoint-*"))
    if workload == "semantic-similarity":
        midpoint_path = run_directory / "checkpoints" / f"checkpoint-{steps // 2}"
        if not midpoint_path.is_dir():
            raise RuntimeError("Sentence Transformers did not save the midpoint")
        midpoint_model = SentenceTransformer(str(midpoint_path), local_files_only=True)
        midpoint_evaluation = _run_reference_evaluation(evaluators, midpoint_model)
        assert similarity_curve is not None
        similarity_curve.append(
            _reference_similarity_point(steps // 2, midpoint_evaluation)
        )
        similarity_curve.append(_reference_similarity_point(steps, final_evaluation))
    return {
        "schema_version": "representax-semantic-pair-worker-v1",
        "framework": "sentence-transformers",
        "framework_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "workload": workload,
        "steps": steps,
        "batch_size": training_batch_size,
        "campaign_batch_size": contract.batch_size,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "maximum_length": contract.maximum_length,
        "precision": "bfloat16-compute-float32-master",
        "training_seconds": training_seconds,
        "examples_per_second": training_batch_size * steps / training_seconds,
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "evaluation_curve": similarity_curve,
        "training_metrics": output.metrics,
        "checkpoint_directories": [str(path) for path in checkpoints],
        "inference_bundle": str(export),
        "reload_probe_maximum_absolute_difference": reload_maximum_difference,
        "reload_probe_cosine": reload_cosine,
        "peak_device_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _worker(arguments: argparse.Namespace) -> None:
    report = (
        _representax_worker(
            arguments.workload,
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
        if arguments.framework == "representax"
        else _sentence_transformers_worker(
            arguments.workload,
            checkpoint=arguments.checkpoint,
            data_directory=arguments.data_directory,
            run_directory=arguments.run_directory,
            steps=arguments.steps,
            seed=arguments.seed,
        )
    )
    if arguments.framework == "representax":
        _write_json(arguments.report, report)
    else:
        write_reference_result(
            arguments.report,
            report,
            reference="sentence-transformers",
        )


def _pair(arguments: argparse.Namespace) -> None:
    if arguments.representax_gpu == arguments.reference_gpu:
        raise ValueError("Representax and the reference must use different GPUs")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    running = []
    for framework, gpu in (
        ("representax", arguments.representax_gpu),
        ("sentence-transformers", arguments.reference_gpu),
    ):
        report = output / f"{framework}.json"
        log = output / f"{framework}.log"
        command = [
            sys.executable,
            "-m",
            "experiments.paper.semantic_pair",
            "worker",
            "--workload",
            arguments.workload,
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
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_COMPILATION_CACHE_DIR": str(output / "jax-cache"),
            }
        )
        if framework == "representax":
            environment.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
            environment.update(
                {
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
                }
            )
        stream = log.open("x", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        running.append((framework, process, stream, log, report, gpu))
    failures = []
    for framework, process, stream, log, _report, gpu in running:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append(
                {
                    "framework": framework,
                    "gpu": gpu,
                    "return_code": return_code,
                    "log": str(log),
                }
            )
    if failures:
        _write_json(output / "failures.json", failures)
        raise RuntimeError(f"preflight workers failed: {failures}")
    reports = {
        framework: _document(report)
        for framework, _process, _stream, _log, report, _gpu in running
    }
    summary = {
        "schema_version": "representax-semantic-pair-preflight-v1",
        "workload": arguments.workload,
        "contract": {
            **asdict(frozen_contract(arguments.workload)),
            "steps": arguments.steps,
            "seed": arguments.seed,
            "representax_gpu": arguments.representax_gpu,
            "reference_gpu": arguments.reference_gpu,
            "data_manifest": _document(arguments.data_directory / "manifest.json"),
        },
        **reports,
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({name: reports[name] for name in FRAMEWORKS}, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument(
        "--workload",
        choices=("semantic-similarity", "pair-classification"),
        required=True,
    )
    worker.add_argument("--framework", choices=FRAMEWORKS, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--data-directory", type=Path, required=True)
    worker.add_argument("--run-directory", type=Path, required=True)
    worker.add_argument("--report", type=Path, required=True)
    worker.add_argument("--steps", type=int, default=8)
    worker.add_argument("--seed", type=int, default=7)

    pair = subparsers.add_parser("pair")
    pair.add_argument(
        "--workload",
        choices=("semantic-similarity", "pair-classification"),
        required=True,
    )
    pair.add_argument("--checkpoint", type=Path, required=True)
    pair.add_argument("--data-directory", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--steps", type=int, default=8)
    pair.add_argument("--seed", type=int, default=7)
    pair.add_argument("--representax-gpu", type=int, default=2)
    pair.add_argument("--reference-gpu", type=int, default=3)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "prepare":
        print(json.dumps(prepare_data(arguments.output), indent=2, sort_keys=True))
    elif arguments.command == "worker":
        _worker(arguments)
    else:
        _pair(arguments)


if __name__ == "__main__":
    main()
