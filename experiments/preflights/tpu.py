"""Run the bounded Representax TPU acceptance matrix used by the Colab notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import numpy as np

from representax import load_inference_bundle
from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    DDPConfig,
    EvaluationConfig,
    EvaluatorConfig,
    ExportConfig,
    FSDPConfig,
    GradCacheConfig,
    JobConfig,
    LoggingConfig,
    MeshConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.data import ArtifactSpec, RandomAccessSource, mix, source
from representax.tasks.pairwise import (
    CosineRegressionConfig,
    PairwiseConfig,
    pairwise_batch,
)
from representax.tasks.retrieval import MNRConfig, RetrievalConfig, retrieval_batch
from representax.train import run_job

TOY_BATCH_SIZE = 16
TOY_DIMENSION = 64
TOY_RECORDS = 512
MAPPER = "experiments.preflights.tpu.identity"


def identity(record: Any) -> Any:
    return record


@dataclass(frozen=True)
class ToySource:
    records: tuple[dict[str, Any], ...]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        return self.records[index]


def _record(index: int) -> dict[str, Any]:
    concept = index % TOY_BATCH_SIZE
    view = index // TOY_BATCH_SIZE
    angle = 2.0 * math.pi * (concept + 0.5) / TOY_BATCH_SIZE
    semantic = [
        function(frequency * angle)
        for frequency in range(1, 17)
        for function in (math.sin, math.cos)
    ]
    left_noise = [
        0.2 * math.sin((concept + 1) * (view + 1) * (axis + 1) * 0.13)
        for axis in range(32)
    ]
    right_noise = [
        0.2 * math.cos((concept + 3) * (view + 2) * (axis + 1) * 0.11)
        for axis in range(32)
    ]
    left = semantic + left_noise
    right = semantic + right_noise
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return {"left": left, "right": right, "label": cosine}


def resolve_toy_records(_artifact: ArtifactSpec) -> RandomAccessSource:
    return ToySource(tuple(_record(index) for index in range(TOY_RECORDS)))


def collate_pairwise(examples: Sequence[dict[str, Any]]) -> Any:
    return pairwise_batch(
        left=np.asarray([example["left"] for example in examples], dtype=np.float32),
        right=np.asarray([example["right"] for example in examples], dtype=np.float32),
        labels=np.asarray([example["label"] for example in examples], dtype=np.float32),
    )


def collate_retrieval(examples: Sequence[dict[str, Any]]) -> Any:
    size = len(examples)
    return retrieval_batch(
        query=np.asarray([example["left"] for example in examples], dtype=np.float32),
        document=np.asarray(
            [example["right"] for example in examples], dtype=np.float32
        ),
        positive_mask=np.eye(size, dtype=np.bool_),
    )


@dataclass(frozen=True)
class Variant:
    name: str
    task: Literal["pairwise", "retrieval"]
    precision: Literal["fp32", "bf16"] = "fp32"
    gradient_accumulation_steps: int = 1
    grad_cache: Literal["rematerialized", "custom_vjp"] | None = None
    sharding: Literal["single", "ddp", "fsdp"] = "single"
    telemetry: bool = False


def variants() -> tuple[Variant, ...]:
    return (
        Variant("pairwise-direct", "pairwise", telemetry=True),
        Variant("pairwise-bf16", "pairwise", precision="bf16"),
        Variant(
            "pairwise-gradient-accumulation",
            "pairwise",
            gradient_accumulation_steps=4,
        ),
        Variant("pairwise-ddp", "pairwise", sharding="ddp"),
        Variant("pairwise-fsdp", "pairwise", sharding="fsdp"),
        Variant("retrieval-direct", "retrieval"),
        Variant(
            "retrieval-grad-cache-rematerialized",
            "retrieval",
            grad_cache="rematerialized",
        ),
        Variant(
            "retrieval-grad-cache-custom-vjp",
            "retrieval",
            grad_cache="custom_vjp",
        ),
    )


def _available_variants(device_count: int) -> tuple[Variant, ...]:
    return tuple(
        variant
        for variant in variants()
        if device_count > 1 or variant.sharding == "single"
    )


def _data(task: Literal["pairwise", "retrieval"]) -> DataConfig:
    return DataConfig(
        distribution=mix(
            source("memory://representax-tpu-toy-v1", map=MAPPER),
            shuffle=False,
            seed=7,
        ),
        collate=ComponentConfig(
            target=(
                "experiments.preflights.tpu.collate_pairwise"
                if task == "pairwise"
                else "experiments.preflights.tpu.collate_retrieval"
            )
        ),
        num_threads=0,
        prefetch_buffer_size=0,
    )


def _job(variant: Variant, *, device_count: int, steps: int) -> JobConfig:
    if variant.sharding == "single":
        mesh = MeshConfig()
        sharding: DDPConfig | FSDPConfig = DDPConfig()
        data_replicas = 1
    elif variant.sharding == "ddp":
        mesh = MeshConfig(axis_shapes=(device_count,), axis_names=("data",))
        sharding = DDPConfig(axis="data")
        data_replicas = device_count
    else:
        mesh = MeshConfig(axis_shapes=(device_count,), axis_names=("model",))
        sharding = FSDPConfig(
            data_axis=None,
            parameter_axis="model",
            minimum_parameter_elements=1,
        )
        data_replicas = 1
    accumulation = variant.gradient_accumulation_steps
    micro_batch_size = TOY_BATCH_SIZE // (data_replicas * accumulation)
    if micro_batch_size * data_replicas * accumulation != TOY_BATCH_SIZE:
        raise ValueError(
            f"toy batch {TOY_BATCH_SIZE} is not divisible by "
            f"{data_replicas} replicas and {accumulation} accumulation steps"
        )
    task = PairwiseConfig() if variant.task == "pairwise" else RetrievalConfig()
    loss = (
        CosineRegressionConfig()
        if variant.task == "pairwise"
        else MNRConfig(scale=5.0, symmetric=True)
    )
    grad_cache = (
        None
        if variant.grad_cache is None
        else GradCacheConfig(
            micro_batch_size=4,
            implementation=variant.grad_cache,
        )
    )
    precision = (
        PrecisionConfig.bfloat16_mixed()
        if variant.precision == "bf16"
        else PrecisionConfig()
    )
    data = _data(variant.task)
    return JobConfig(
        name=f"tpu-acceptance-{variant.name}",
        model=ModelConfig(
            target="representax.models.DenseEncoder",
            parameters={
                "input_dimension": TOY_DIMENSION,
                "output_dimension": TOY_DIMENSION,
            },
        ),
        task=task,
        loss=loss,
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 0.01, "weight_decay": 0.0},
            ),
            max_gradient_norm=None,
        ),
        data=data,
        training=TrainingConfig(
            global_batch_size=TOY_BATCH_SIZE,
            max_steps=steps,
            seed=7,
            mesh=mesh,
            sharding=sharding,
            batch=BatchConfig(
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=accumulation,
            ),
            grad_cache=grad_cache,
            precision=precision,
            activation_rematerialization="none",
        ),
        checkpointing=CheckpointConfig(
            every=max(1, steps // 2),
            keep=2,
            asynchronous=True,
        ),
        logging=LoggingConfig(
            timing=True,
            accelerator=(variant.telemetry and jax.default_backend() in {"gpu", "tpu"}),
        ),
        evaluation=EvaluationConfig(
            data=data,
            batch_size=TOY_BATCH_SIZE,
            evaluators=(EvaluatorConfig(),),
            on_start=True,
            on_end=True,
        ),
        export=ExportConfig(enabled=True, selection="final"),
    )


def _model_vector(model: eqx.Module) -> np.ndarray:
    arrays = [
        np.asarray(jax.device_get(value), dtype=np.float64).reshape(-1)
        for value in jax.tree.leaves(eqx.filter(model, eqx.is_array))
        if value is not None
    ]
    return np.concatenate(arrays)


def _digest(vector: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(vector.tobytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _metric_rows(run_directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]


def _run_variant(
    variant: Variant,
    *,
    output: Path,
    device_count: int,
    steps: int,
) -> tuple[dict[str, Any], np.ndarray]:
    run_directory = output / "runs" / variant.name
    job = _job(variant, device_count=device_count, steps=steps)
    resolver = {"memory": resolve_toy_records}
    mapper = {MAPPER: identity}
    paused = run_job(
        job,
        run_directory,
        resolvers=resolver,
        mappers=mapper,
        stop_after=max(1, steps // 2),
    )
    resumed = run_job(
        job,
        run_directory,
        resume=True,
        resolvers=resolver,
        mappers=mapper,
    )
    if paused.completed_iterations != max(1, steps // 2):
        raise RuntimeError(f"{variant.name} did not stop at its resume boundary")
    paused_vector = _model_vector(paused.selected_model)
    if resumed.completed_iterations != steps or not resumed.resumed:
        raise RuntimeError(f"{variant.name} did not complete through resume")
    if resumed.inference_bundle is None:
        raise RuntimeError(f"{variant.name} did not export an inference bundle")
    exported_model, exported_job = load_inference_bundle(resumed.inference_bundle)
    if exported_job != job:
        raise RuntimeError(f"{variant.name} exported a different job contract")
    vector = _model_vector(resumed.selected_model)
    exported_vector = _model_vector(exported_model)
    if not np.array_equal(vector, exported_vector):
        raise RuntimeError(f"{variant.name} export/reload changed model parameters")
    resume_update_l2 = float(np.linalg.norm(vector - paused_vector))
    if resume_update_l2 == 0.0:
        raise RuntimeError(f"{variant.name} parameters did not update after resume")
    rows = _metric_rows(run_directory)
    training = [row for row in rows if row["event"] == "training_step"]
    losses = np.asarray(
        [row["metrics"]["train/loss"] for row in training], dtype=np.float64
    )
    if len(losses) != steps or not np.all(np.isfinite(losses)):
        raise RuntimeError(f"{variant.name} did not report {steps} finite updates")
    accelerator_rows = [row for row in rows if row["event"] == "accelerator"]
    if job.logging.accelerator and not accelerator_rows:
        raise RuntimeError(f"{variant.name} did not publish accelerator telemetry")
    warm_rates = [
        float(row["metrics"]["perf/examples_per_second"])
        for row in training
        if "perf/examples_per_second" in row["metrics"]
        and "perf/compilation_and_first_step_seconds" not in row["metrics"]
    ]
    return (
        {
            "name": variant.name,
            "task": variant.task,
            "precision": variant.precision,
            "sharding": variant.sharding,
            "device_count": device_count if variant.sharding != "single" else 1,
            "steps": steps,
            "first_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "parameter_digest": _digest(vector),
            "parameter_norm": float(np.linalg.norm(vector)),
            "resume_update_l2": resume_update_l2,
            "export_reload_exact": True,
            "accelerator_samples": len(accelerator_rows),
            "warm_examples_per_second": (
                float(np.median(warm_rates)) if warm_rates else None
            ),
        },
        vector,
    )


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.linalg.norm(actual - expected)
        / max(np.linalg.norm(expected), np.finfo(np.float64).tiny)
    )


def _cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.dot(actual, expected)
        / max(
            np.linalg.norm(actual) * np.linalg.norm(expected),
            np.finfo(np.float64).tiny,
        )
    )


def _git_revision() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run(output: Path, *, steps: int, device_count: int) -> dict[str, Any]:
    if jax.process_count() != 1:
        raise RuntimeError("the Colab acceptance runner supports one host process")
    visible = len(jax.devices())
    if not 1 <= device_count <= visible:
        raise ValueError(
            f"device_count must be between one and {visible}; got {device_count}"
        )
    if TOY_BATCH_SIZE % device_count:
        raise ValueError(
            f"device_count {device_count} must divide toy batch {TOY_BATCH_SIZE}"
        )
    output.mkdir(parents=True, exist_ok=False)
    reports: dict[str, dict[str, Any]] = {}
    vectors: dict[str, np.ndarray] = {}
    available_variants = _available_variants(device_count)
    skipped_variants = tuple(
        variant.name for variant in variants() if variant not in available_variants
    )
    for variant in available_variants:
        report, vector = _run_variant(
            variant,
            output=output,
            device_count=device_count,
            steps=steps,
        )
        reports[variant.name] = report
        vectors[variant.name] = vector

    parity_groups = {
        "pairwise": (
            "pairwise-gradient-accumulation",
            "pairwise-ddp",
            "pairwise-fsdp",
        ),
        "retrieval": (
            "retrieval-grad-cache-rematerialized",
            "retrieval-grad-cache-custom-vjp",
        ),
    }
    parity: dict[str, Any] = {}
    passed = True
    for group, candidates in parity_groups.items():
        baseline_name = f"{group}-direct"
        baseline = vectors[baseline_name]
        for candidate_name in candidates:
            if candidate_name not in vectors:
                continue
            candidate = vectors[candidate_name]
            relative_l2 = _relative_l2(candidate, baseline)
            cosine = _cosine(candidate, baseline)
            accepted = relative_l2 <= 5e-4 and cosine >= 0.999999
            parity[candidate_name] = {
                "baseline": baseline_name,
                "relative_l2": relative_l2,
                "cosine": cosine,
                "accepted": accepted,
            }
            passed &= accepted

    result = {
        "schema_version": "representax-tpu-acceptance-v1",
        "accepted": passed,
        "environment": {
            "platform": jax.default_backend(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": _package_version("jaxlib"),
            "libtpu": _package_version("libtpu"),
            "process_count": jax.process_count(),
            "process_index": jax.process_index(),
            "visible_device_count": visible,
            "tested_device_count": device_count,
            "devices": [str(device) for device in jax.devices()],
            "representax_revision": _git_revision(),
            "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
            "colab_backend_version": os.environ.get("COLAB_BACKEND_VERSION"),
        },
        "variants": reports,
        "skipped_variants": {
            name: "requires at least two JAX devices" for name in skipped_variants
        },
        "parity": parity,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--device-count",
        type=int,
        help="number of local TPU devices; defaults to at most eight",
    )
    parser.add_argument(
        "--allow-non-tpu",
        action="store_true",
        help="allow local CPU/GPU validation of the runner",
    )
    arguments = parser.parse_args()
    if arguments.steps < 2:
        parser.error("--steps must be at least two")
    if jax.default_backend() != "tpu" and not arguments.allow_non_tpu:
        parser.error(f"TPU backend required; JAX selected {jax.default_backend()!r}")
    device_count = arguments.device_count or min(8, len(jax.devices()))
    result = run(
        arguments.output.expanduser().resolve(),
        steps=arguments.steps,
        device_count=device_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
