"""Run the paper's matched framework comparisons on GPU or TPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform as platform_module
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DATASET_ID = "sentence-transformers/msmarco-msmarco-MiniLM-L6-v3"
DATASET_REVISION = "0d54352548089199bde15ad7e06efe895dc80b56"
SEEDS = (7, 42, 773, 1234, 2026)
STEPS = 20
GLOBAL_BATCH_SIZE = 2048
QUERY_LENGTH = 32
DOCUMENT_LENGTH = 256
DATA_POOL_SIZE = 65_536
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.06
REPRESENTAX_CHUNK_SIZE = 32
REFERENCE_CHUNK_SIZE = 128
EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
DATA_MANIFEST = EXPERIMENT_DIRECTORY / "data-manifest.json"
MODEL_MANIFEST = EXPERIMENT_DIRECTORY / "model-manifest.json"

Variant = Literal[
    "representax-local",
    "representax-global",
    "sentence-transformers-local",
]
Platform = Literal["gpu", "tpu"]

RECIPES = (
    "dense-retrieval",
    "semantic-similarity-mpnet-base",
    "semantic-similarity-bert-base",
    "pair-classification-mpnet-base",
    "pair-classification-bert-base",
    "cross-encoder",
    "late-interaction",
    "outcome-reward",
    "process-reward",
    "image-text",
    "audio-text",
    "video-text",
    "v-jepa",
)
NEGATIVE_SCOPE_RECIPES = frozenset(
    {
        "dense-retrieval",
        "late-interaction",
        "image-text",
        "audio-text",
        "video-text",
    }
)

REFERENCE_FRAMEWORKS = {
    "dense-retrieval": "sentence-transformers",
    "semantic-similarity-mpnet-base": "sentence-transformers",
    "semantic-similarity-bert-base": "sentence-transformers",
    "pair-classification-mpnet-base": "sentence-transformers",
    "pair-classification-bert-base": "sentence-transformers",
    "cross-encoder": "sentence-transformers",
    "late-interaction": "pylate",
    "outcome-reward": "trl",
    "process-reward": "trl",
    "image-text": "sentence-transformers",
    "audio-text": "sentence-transformers",
    "video-text": "sentence-transformers",
    "v-jepa": "facebookresearch-vjepa2",
}

RECIPE_ASSETS = {
    "dense-retrieval": ("all-mpnet-base-v2", "dense-msmarco-unique-v1", None),
    "semantic-similarity-mpnet-base": ("all-mpnet-base-v2", "pairs", None),
    "semantic-similarity-bert-base": ("bert-base", "pairs", None),
    "pair-classification-mpnet-base": ("all-mpnet-base-v2", "pairs", None),
    "pair-classification-bert-base": ("bert-base", "pairs", None),
    "cross-encoder": ("cross-checkpoint", "cross-data", None),
    "late-interaction": ("late-checkpoint", "late-data", None),
    "outcome-reward": ("qwen3-0.6b", "outcome-data", None),
    "process-reward": ("qwen3-0.6b", "process-data", None),
    "image-text": ("clip-vit-b-32", "image-data", None),
    "audio-text": ("omni-3b", "audio-data", None),
    "video-text": ("omni-3b", "video-data", None),
    "v-jepa": (None, "vjepa-data", "vjepa2-reference"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    patch = subprocess.run(
        ("git", "diff", "--binary"),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": head,
        "working_tree_patch_sha256": ("sha256:" + hashlib.sha256(patch).hexdigest()),
        "working_tree_clean": not bool(patch),
    }


def _environment_state(
    worker_python: Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if worker_python is not None:
        result = subprocess.run(
            (str(worker_python), str(Path(__file__).resolve()), "environment"),
            cwd=Path(__file__).resolve().parents[2],
            env=dict(environment) if environment is not None else None,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    accelerator_variables = (
        "CUDA_VISIBLE_DEVICES",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "PJRT_DEVICE",
        "XLA_PERSISTENT_CACHE_PATH",
        "XLA_PERSISTENT_CACHE_READ_ONLY",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "python": platform_module.python_version(),
        "executable": sys.executable,
        "platform": platform_module.platform(),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "accelerator_environment": {
            name: os.environ[name]
            for name in accelerator_variables
            if name in os.environ
        },
    }


def _contract(
    *,
    seed: int,
    variant: Variant,
    steps: int,
    platform: Platform,
    torch_compile: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": "representax-cross-accelerator-dense-v1",
        "variant": variant,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "manifest_sha256": _sha256(MODEL_MANIFEST),
        },
        "data": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "manifest_sha256": _sha256(DATA_MANIFEST),
            "order_seed": seed,
            "duplicate_queries": 0,
            "duplicate_positives": 0,
        },
        "training": {
            "steps": steps,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "query_length": QUERY_LENGTH,
            "document_length": DOCUMENT_LENGTH,
            "parameter_dtype": "float32",
            "compute_dtype": "bfloat16",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "warmup_steps": round(steps * WARMUP_RATIO),
            "schedule": "cosine",
            "weight_decay": 0.0,
            "max_gradient_norm": 1.0,
            "mnr_scale": 20.0,
            "mnr_symmetric": False,
            "negative_scope": (
                "global" if variant == "representax-global" else "local"
            ),
        },
        "execution": {
            "platform": platform,
            "representax_grad_cache_chunk": REPRESENTAX_CHUNK_SIZE,
            "sentence_transformers_loss": (
                "multiple_negatives_ranking"
                if platform == "tpu"
                else "cached_multiple_negatives_ranking"
            ),
            "sentence_transformers_grad_cache_chunk": (
                None if platform == "tpu" else REFERENCE_CHUNK_SIZE
            ),
            "torch_compile": torch_compile,
        },
    }


def _training_path(data: Path, seed: int) -> Path:
    path = data / f"seed-{seed}.parquet"
    manifest_path = data / "manifest.json"
    if _sha256(manifest_path) != _sha256(DATA_MANIFEST):
        raise ValueError(f"prepared data manifest changed: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    record = manifest["seed_files"][str(seed)]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"prepared training data hash changed: {path}")
    return path


def _verify_checkpoint(checkpoint: Path) -> None:
    manifest = json.loads(MODEL_MANIFEST.read_text())
    if manifest["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("tracked model manifest does not match the frozen model")
    for relative, expected in manifest["files"].items():
        path = checkpoint / relative
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file is missing: {path}")
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"checkpoint file size changed: {path}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"checkpoint file hash changed: {path}")


def _prepare_data(output: Path, checkpoint: Path, rows: int) -> None:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"prepared data directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    selected: list[dict[str, Any]] = []
    queries: set[str] = set()
    positives: set[str] = set()
    source_files: list[dict[str, Any]] = []

    for shard in range(39):
        filename = f"triplet-all/train-{shard:05d}-of-00039.parquet"
        path = Path(
            hf_hub_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=DATASET_REVISION,
                filename=filename,
                local_dir=output / "source",
            )
        )
        source_files.append(
            {"file": filename, "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
        source_row = 0
        source = parquet.ParquetFile(path, memory_map=True)
        for batch in source.iter_batches(
            batch_size=4096,
            columns=("query", "positive"),
        ):
            values = batch.to_pylist()
            candidates = [
                (source_row + index, str(value["query"]), str(value["positive"]))
                for index, value in enumerate(values)
                if value["query"] not in queries and value["positive"] not in positives
            ]
            source_row += len(values)
            if not candidates:
                continue
            query_tokens = tokenizer(
                [value[1] for value in candidates],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            positive_tokens = tokenizer(
                [value[2] for value in candidates],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            for candidate, query_ids, positive_ids in zip(
                candidates,
                query_tokens,
                positive_tokens,
                strict=True,
            ):
                row_index, query, positive = candidate
                if (
                    query in queries
                    or positive in positives
                    or len(query_ids) > QUERY_LENGTH
                    or len(positive_ids) > DOCUMENT_LENGTH
                ):
                    continue
                queries.add(query)
                positives.add(positive)
                selected.append(
                    {
                        "query": query,
                        "positive": positive,
                        "source_shard": shard,
                        "source_row": row_index,
                    }
                )
                if len(selected) == rows:
                    break
            if len(selected) == rows:
                break
        if len(selected) == rows:
            break
    if len(selected) != rows:
        raise ValueError(f"only found {len(selected)} eligible unique pairs")

    table = pa.Table.from_pylist(selected)
    seed_files = {}
    for seed in SEEDS:
        permutation = np.random.default_rng(seed).permutation(rows)
        seeded = table.take(pa.array(permutation))
        path = output / f"seed-{seed}.parquet"
        parquet.write_table(seeded, path, compression="zstd")
        seed_files[str(seed)] = {
            "file": path.name,
            "rows": seeded.num_rows,
            "sha256": _sha256(path),
        }
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "representax-unique-msmarco-v1",
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION},
            "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "selection": {
                "rows": rows,
                "query_length_at_most": QUERY_LENGTH,
                "document_length_at_most": DOCUMENT_LENGTH,
                "unique_query": True,
                "unique_positive": True,
                "order": "first eligible source occurrence",
            },
            "source_files": source_files,
            "seed_files": seed_files,
        },
    )


def _training_rows(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(path, columns=["query", "positive"])
    return table.to_pylist()


def _representax(
    *,
    checkpoint: Path,
    data: Path,
    output: Path,
    seed: int,
    steps: int,
    scope: Literal["local", "global"],
    platform: Platform,
) -> None:
    import jax
    import numpy as np

    if platform == "tpu":
        jax.distributed.initialize()

    from representax.config import (
        BatchConfig,
        ComponentConfig,
        DataConfig,
        DDPConfig,
        ExportConfig,
        GradCacheConfig,
        JobConfig,
        LoggingConfig,
        MeshConfig,
        ModelConfig,
        OptimizationConfig,
        PrecisionConfig,
        TrainingConfig,
    )
    from representax.data import identity, mix, source
    from representax.tasks.retrieval import MNRConfig, RetrievalConfig
    from representax.train import run_job

    if GLOBAL_BATCH_SIZE % jax.device_count():
        raise ValueError("global batch must divide the device count")
    variant = cast(Variant, f"representax-{scope}")
    run_directory = output / f"process-{jax.process_index()}"
    job = JobConfig(
        name=f"{platform}-dense-{scope}-seed-{seed}",
        model=ModelConfig(
            target="representax.models:SentenceEncoder.load_from_hf",
            parameters={
                "model_name_or_path": str(checkpoint),
                "revision": MODEL_REVISION,
                "local_files_only": True,
                "parameter_dtype": "float32",
                "compute_dtype": "bfloat16",
                "sequence_length_buckets": (QUERY_LENGTH, DOCUMENT_LENGTH),
            },
        ),
        task=RetrievalConfig(),
        loss=MNRConfig(
            scale=20.0,
            symmetric=False,
            negative_scope=scope,
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
                    "peak_value": LEARNING_RATE,
                    "warmup_steps": round(steps * WARMUP_RATIO),
                    "decay_steps": steps,
                    "end_value": 0.0,
                },
            ),
            max_gradient_norm=1.0,
        ),
        data=DataConfig(
            distribution=mix(
                source(str(_training_path(data, seed)), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target="representax.tasks.retrieval.RetrievalCollator"
            ),
            num_threads=4,
            prefetch_buffer_size=8,
        ),
        training=TrainingConfig(
            global_batch_size=GLOBAL_BATCH_SIZE,
            max_steps=steps,
            seed=seed,
            mesh=MeshConfig(axis_shapes=(jax.device_count(),), axis_names=("data",)),
            sharding=DDPConfig(axis="data"),
            batch=BatchConfig(micro_batch_size=GLOBAL_BATCH_SIZE // jax.device_count()),
            grad_cache=GradCacheConfig(
                micro_batch_size=REPRESENTAX_CHUNK_SIZE,
                implementation="rematerialized",
            ),
            precision=PrecisionConfig.bfloat16_mixed(),
            donate_buffers=True,
        ),
        checkpointing=None,
        logging=LoggingConfig(console_every=steps, timing=True, accelerator=False),
        evaluation=None,
        export=ExportConfig(enabled=False),
    )
    result = run_job(job, run_directory)
    jax.block_until_ready(result.state)
    parameter = next(
        value
        for value in jax.tree.leaves(result.state.model)
        if hasattr(value, "dtype") and np.issubdtype(value.dtype, np.inexact)
    )
    local_probe = np.asarray(parameter.addressable_data(0)).reshape(-1)[:1]
    if jax.process_count() == 1:
        probes = local_probe
    else:
        from jax.experimental import multihost_utils

        probes = np.asarray(multihost_utils.process_allgather(local_probe)).reshape(-1)
    if jax.process_index() != 0:
        return

    rows = [
        json.loads(line)
        for line in (run_directory / "metrics.jsonl").read_text().splitlines()
    ]
    training = [row for row in rows if row.get("event") == "training_step"]
    warm = [
        row
        for row in training
        if "perf/step_seconds" in row["metrics"]
        and "perf/compilation_and_first_step_seconds" not in row["metrics"]
    ]
    summary = {
        "contract": _contract(
            seed=seed,
            variant=variant,
            steps=steps,
            platform=platform,
        ),
        "framework": "representax",
        "negative_scope": scope,
        "jax_version": jax.__version__,
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "first_loss": float(training[0]["metrics"]["train/loss"]),
        "final_loss": float(training[-1]["metrics"]["train/loss"]),
        "warm_observations": len(warm),
        "median_step_seconds": statistics.median(
            float(row["metrics"]["perf/step_seconds"]) for row in warm
        ),
        "median_examples_per_second": statistics.median(
            float(row["metrics"]["perf/examples_per_second"]) for row in warm
        ),
        "parameter_probe_spread": float(probes.max() - probes.min()),
        "source": _git_state(),
        "command": sys.argv,
    }
    _write_json(run_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


class _FixedPairCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.valid_label_columns: list[str] = []
        self.valid_tokens = 0

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        output = {}
        self.valid_tokens = 0
        for field, length in (("query", QUERY_LENGTH), ("positive", DOCUMENT_LENGTH)):
            encoded = self.tokenizer(
                [str(row[field]) for row in features],
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            )
            self.valid_tokens += int(encoded["attention_mask"].sum())
            output.update({f"{field}_{name}": value for name, value in encoded.items()})
        return output


def _sequential_sentence_transformers_batches(
    dataset: Any,
    batch_size: int,
    drop_last: bool,
    valid_label_columns: list[str] | None = None,
    generator: Any = None,
    seed: int = 0,
) -> Any:
    """Preserve the prepared seed file's exact row order."""
    del generator
    from sentence_transformers.base.sampler import DefaultBatchSampler
    from torch.utils.data import SequentialSampler

    return DefaultBatchSampler(
        SequentialSampler(dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        valid_label_columns=valid_label_columns,
        seed=seed,
    )


def _sentence_transformers_worker(
    index: int,
    checkpoint: str,
    data: str,
    output: str,
    seed: int,
    steps: int,
    platform: Platform,
    torch_compile: bool,
) -> None:
    del index
    import datasets
    import sentence_transformers
    import torch
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        CachedMultipleNegativesRankingLoss,
        MultipleNegativesRankingLoss,
    )
    from transformers import TrainerCallback

    if platform == "tpu":
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        world_size = xr.world_size()
        process_count = xr.process_count()
        rank = xr.global_ordinal()

        def synchronize() -> None:
            torch_xla.sync(wait=True)

        def reduce_measurements(values: Any) -> Any:
            return xm.all_reduce(xm.REDUCE_SUM, values)

        def parameter_probes(value: Any) -> Any:
            return xm.all_gather(value)

        torch_xla_version: str | None = torch_xla.__version__
    else:
        world_size = 1
        process_count = 1
        rank = 0

        def synchronize() -> None:
            torch.cuda.synchronize()

        def reduce_measurements(values: Any) -> Any:
            return values

        def parameter_probes(value: Any) -> Any:
            return value

        torch_xla_version = None
    if GLOBAL_BATCH_SIZE % world_size:
        raise ValueError("global batch must divide the device count")
    local_batch_size = GLOBAL_BATCH_SIZE // world_size
    run_directory = Path(output) / f"process-{rank}"
    run_directory.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(checkpoint, local_files_only=True)
    model.max_seq_length = DOCUMENT_LENGTH
    collator = _FixedPairCollator(model.tokenizer)
    table = datasets.Dataset.from_list(_training_rows(Path(data)))
    if platform == "tpu":
        loss = MultipleNegativesRankingLoss(
            model,
            scale=20.0,
            gather_across_devices=False,
        )
    else:
        loss = CachedMultipleNegativesRankingLoss(
            model,
            scale=20.0,
            mini_batch_size=REFERENCE_CHUNK_SIZE,
            gather_across_devices=False,
            show_progress_bar=False,
        )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(run_directory / "checkpoints"),
        per_device_train_batch_size=local_batch_size,
        max_steps=steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=round(steps * WARMUP_RATIO),
        optim="adamw_torch",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        logging_strategy="no",
        report_to="none",
        disable_tqdm=True,
        save_strategy="no",
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        torch_compile=torch_compile,
        torch_compile_backend="inductor" if torch_compile else None,
        batch_sampler=_sequential_sentence_transformers_batches,
        seed=seed,
        data_seed=seed,
    )

    class RecordingTrainer(SentenceTransformerTrainer):
        current_loss: Any = None

        def compute_loss(self, *args: Any, **kwargs: Any) -> Any:
            value = super().compute_loss(*args, **kwargs)
            loss_value = value[0] if isinstance(value, tuple) else value
            self.current_loss = loss_value.detach()
            return value

    class StepRecorder(TrainerCallback):
        trainer: RecordingTrainer | None = None

        def __init__(self) -> None:
            self.previous: float | None = None
            self.rows: list[dict[str, Any]] = []

        def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
            synchronize()
            self.previous = time.perf_counter()

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if self.previous is None or self.trainer is None:
                raise AssertionError("training timing callback was not initialized")
            if self.trainer.current_loss is None:
                raise AssertionError("trainer did not expose the current loss")
            loss_scale = world_size if platform == "tpu" else 1
            values = torch.stack(
                (
                    self.trainer.current_loss.float() / loss_scale,
                    torch.tensor(
                        float(collator.valid_tokens),
                        device=self.trainer.current_loss.device,
                    ),
                )
            )
            values = reduce_measurements(values)
            synchronize()
            now = time.perf_counter()
            elapsed = now - self.previous
            self.previous = now
            if rank == 0:
                self.rows.append(
                    {
                        "event": "training_step",
                        "iteration": int(state.global_step),
                        "metrics": {
                            "train/loss": float(values[0].cpu()),
                            "perf/step_seconds": elapsed,
                            "perf/examples": GLOBAL_BATCH_SIZE,
                            "perf/examples_per_second": GLOBAL_BATCH_SIZE / elapsed,
                            "perf/tokens": int(values[1].cpu()),
                            "perf/tokens_per_second": float(values[1].cpu()) / elapsed,
                            "perf/padded_positions": GLOBAL_BATCH_SIZE
                            * (QUERY_LENGTH + DOCUMENT_LENGTH),
                        },
                    }
                )
            return control

    recorder = StepRecorder()
    trainer = RecordingTrainer(
        model=model,
        args=arguments,
        train_dataset=table,
        loss=loss,
        data_collator=collator,
        callbacks=[recorder],
    )
    recorder.trainer = trainer
    trainer.train()
    synchronize()
    probe = next(model.parameters()).detach().reshape(-1)[:1]
    probes = parameter_probes(probe)
    synchronize()
    if rank != 0:
        return

    metrics_path = run_directory / "metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in recorder.rows)
    )
    warm = recorder.rows[2:] if platform == "tpu" else recorder.rows[1:]
    variant: Variant = "sentence-transformers-local"
    summary = {
        "contract": _contract(
            seed=seed,
            variant=variant,
            steps=steps,
            platform=platform,
            torch_compile=torch_compile,
        ),
        "framework": "sentence-transformers",
        "negative_scope": "local",
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "torch_xla_version": torch_xla_version,
        "distributed_type": str(trainer.accelerator.state.distributed_type),
        "process_count": process_count,
        "device_count": world_size,
        "first_loss": recorder.rows[0]["metrics"]["train/loss"],
        "final_loss": recorder.rows[-1]["metrics"]["train/loss"],
        "warm_observations": len(warm),
        "median_step_seconds": statistics.median(
            row["metrics"]["perf/step_seconds"] for row in warm
        ),
        "median_examples_per_second": statistics.median(
            row["metrics"]["perf/examples_per_second"] for row in warm
        ),
        "parameter_probe_spread": float((probes.max() - probes.min()).cpu()),
        "source": _git_state(),
        "command": sys.argv,
    }
    _write_json(run_directory / "run.json", summary["contract"])
    _write_json(run_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _sentence_transformers(
    *,
    checkpoint: Path,
    data: Path,
    output: Path,
    seed: int,
    steps: int,
    platform: Platform,
    torch_compile: bool,
) -> None:
    worker_arguments = (
        str(checkpoint),
        str(_training_path(data, seed)),
        str(output),
        seed,
        steps,
        platform,
        torch_compile,
    )
    if platform == "tpu":
        import torch_xla

        torch_xla.launch(_sentence_transformers_worker, args=worker_arguments)
    else:
        _sentence_transformers_worker(0, *worker_arguments)


def _recipe_command(arguments: argparse.Namespace) -> list[str]:
    worker_python = str(arguments.worker_python or Path(sys.executable))
    framework = (
        "representax"
        if arguments.framework == "representax"
        else REFERENCE_FRAMEWORKS[arguments.recipe]
    )
    if arguments.recipe == "dense-retrieval":
        if arguments.checkpoint is None:
            raise ValueError("dense retrieval requires --checkpoint")
        command = [
            worker_python,
            str(Path(__file__).resolve()),
            "representax" if arguments.framework == "representax" else framework,
            "--checkpoint",
            str(arguments.checkpoint),
            "--data",
            str(arguments.data),
            "--output",
            str(arguments.output),
            "--seed",
            str(arguments.seed),
            "--steps",
            str(arguments.steps),
            "--platform",
            arguments.platform,
        ]
        if arguments.platform == "gpu":
            command.extend(("--gpu", str(arguments.gpu)))
        if arguments.framework == "representax":
            command.extend(("--negative-scope", arguments.negative_scope))
        elif arguments.torch_compile:
            command.append("--torch-compile")
        return command

    if arguments.checkpoint is None and arguments.recipe != "v-jepa":
        raise ValueError(f"{arguments.recipe} requires --checkpoint")
    module = {
        "semantic-similarity-mpnet-base": "semantic_pair",
        "semantic-similarity-bert-base": "semantic_pair",
        "pair-classification-mpnet-base": "semantic_pair",
        "pair-classification-bert-base": "semantic_pair",
        "cross-encoder": "cross_encoder",
        "late-interaction": "late_interaction",
        "outcome-reward": "outcome_reward",
        "process-reward": "process_reward",
        "image-text": "image_text",
        "audio-text": "audio_text",
        "video-text": "video_text",
        "v-jepa": "vjepa",
    }[arguments.recipe]
    command = [
        worker_python,
        "-m",
        f"experiments.preflights.{module}",
        "worker",
        "--framework",
        framework,
        "--data-directory",
        str(arguments.data),
        "--run-directory",
        str(arguments.output / "run"),
        "--report",
        str(arguments.output / "summary.json"),
        "--steps",
        str(arguments.steps),
        "--seed",
        str(arguments.seed),
        "--platform",
        arguments.platform,
    ]
    if arguments.checkpoint is not None:
        command.extend(("--checkpoint", str(arguments.checkpoint)))
    if arguments.recipe.startswith(("semantic-similarity-", "pair-classification-")):
        workload = (
            "semantic-similarity"
            if arguments.recipe.startswith("semantic-similarity-")
            else "pair-classification"
        )
        model = arguments.recipe.removeprefix(f"{workload}-")
        command.extend(("--workload", workload, "--model", model, "--serious"))
        return command
    if (
        arguments.recipe in NEGATIVE_SCOPE_RECIPES
        and arguments.framework == "representax"
    ):
        command.extend(("--negative-scope", arguments.negative_scope))
    if arguments.recipe == "outcome-reward":
        command.extend(("--padding", "static"))
    if arguments.recipe == "audio-text":
        command.extend(
            ("--batch-size", "256", "--sharding", "ddp", "--continuous")
        )
    if arguments.recipe == "video-text":
        command.extend(("--batch-size", "128"))
    if arguments.recipe == "v-jepa":
        if arguments.reference is None:
            raise ValueError("V-JEPA requires --reference")
        command.extend(
            ("--reference", str(arguments.reference), "--batch-size", "128")
        )
    return command


def _run_recipe(arguments: argparse.Namespace) -> None:
    if arguments.platform == "gpu":
        if arguments.gpu is None or arguments.gpu < 0:
            raise ValueError("GPU runs require one non-negative --gpu index")
    elif arguments.gpu is not None:
        raise ValueError("TPU runs do not accept --gpu")
    if arguments.torch_compile and not (
        arguments.recipe == "dense-retrieval"
        and arguments.framework == "reference"
        and arguments.platform == "gpu"
    ):
        raise ValueError("--torch-compile is only the dense GPU reference control")
    arguments.output.mkdir(parents=True, exist_ok=False)
    command = _recipe_command(arguments)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if arguments.platform == "tpu" and arguments.framework == "reference":
        environment["PJRT_DEVICE"] = "TPU"
        environment.setdefault(
            "XLA_PERSISTENT_CACHE_PATH",
            str(Path.home() / ".cache/representax/torch-xla"),
        )
        environment.setdefault("XLA_PERSISTENT_CACHE_READ_ONLY", "0")
        Path(environment["XLA_PERSISTENT_CACHE_PATH"]).mkdir(
            parents=True,
            exist_ok=True,
        )
    if arguments.platform == "gpu":
        environment["CUDA_VISIBLE_DEVICES"] = str(arguments.gpu)
    invocation = {
        "schema_version": "representax-cross-accelerator-invocation-v1",
        "recipe": arguments.recipe,
        "framework": arguments.framework,
        "reference_framework": REFERENCE_FRAMEWORKS[arguments.recipe],
        "platform": arguments.platform,
        "seed": arguments.seed,
        "steps": arguments.steps,
        "negative_scope": (
            arguments.negative_scope
            if arguments.framework == "representax"
            and arguments.recipe in NEGATIVE_SCOPE_RECIPES
            else "local"
            if arguments.framework == "reference"
            and arguments.recipe in NEGATIVE_SCOPE_RECIPES
            else None
        ),
        "command": command,
        "source": _git_state(),
    }
    _write_json(arguments.output / "invocation.json", invocation)
    _write_json(
        arguments.output / "environment.json",
        _environment_state(Path(command[0]), environment=environment),
    )
    with (arguments.output / "worker.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _run_suite(arguments: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    if arguments.platform == "tpu":
        jax_python = arguments.jax_python or (
            root / "experiments/tpu/.venv-jax/bin/python"
        )
        reference_python = arguments.reference_python or (
            root / "experiments/tpu/.venv-torch-xla/bin/python"
        )
        late_python = arguments.late_reference_python or (
            root / "experiments/tpu/.venv-torch-xla-late/bin/python"
        )
    else:
        jax_python = arguments.jax_python or Path(sys.executable)
        reference_python = arguments.reference_python or Path(sys.executable)
        late_python = arguments.late_reference_python or reference_python

    for recipe in arguments.recipes or RECIPES:
        checkpoint_name, data_name, reference_name = RECIPE_ASSETS[recipe]
        worker_python = (
            jax_python
            if arguments.framework == "representax"
            else late_python
            if recipe == "late-interaction"
            else reference_python
        )
        variant = (
            f"representax-{arguments.negative_scope}"
            if arguments.framework == "representax"
            and recipe in NEGATIVE_SCOPE_RECIPES
            else arguments.framework
        )
        _run_recipe(
            argparse.Namespace(
                recipe=recipe,
                framework=arguments.framework,
                checkpoint=(
                    arguments.asset_root / checkpoint_name
                    if checkpoint_name is not None
                    else None
                ),
                data=arguments.asset_root / data_name,
                reference=(
                    arguments.asset_root / reference_name
                    if reference_name is not None
                    else None
                ),
                output=arguments.output / recipe / variant,
                seed=arguments.seed,
                steps=arguments.steps,
                platform=arguments.platform,
                gpu=arguments.gpu,
                worker_python=worker_python,
                negative_scope=arguments.negative_scope,
                torch_compile=arguments.torch_compile,
            )
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment", help=argparse.SUPPRESS)
    prepare = commands.add_parser("prepare-data")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--rows", type=int, default=DATA_POOL_SIZE)
    suite = commands.add_parser("suite")
    suite.add_argument(
        "--framework",
        choices=("representax", "reference"),
        required=True,
    )
    suite.add_argument("--asset-root", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    suite.add_argument("--steps", type=int, default=4)
    suite.add_argument("--platform", choices=("gpu", "tpu"), required=True)
    suite.add_argument("--gpu", type=int)
    suite.add_argument("--recipe", dest="recipes", action="append", choices=RECIPES)
    suite.add_argument("--jax-python", type=Path)
    suite.add_argument("--reference-python", type=Path)
    suite.add_argument("--late-reference-python", type=Path)
    suite.add_argument(
        "--negative-scope",
        choices=("local", "global"),
        default="local",
    )
    suite.add_argument("--torch-compile", action="store_true")
    recipe = commands.add_parser("recipe")
    recipe.add_argument("--recipe", choices=RECIPES, required=True)
    recipe.add_argument(
        "--framework",
        choices=("representax", "reference"),
        required=True,
    )
    recipe.add_argument("--checkpoint", type=Path)
    recipe.add_argument("--data", type=Path, required=True)
    recipe.add_argument("--reference", type=Path)
    recipe.add_argument("--output", type=Path, required=True)
    recipe.add_argument("--seed", type=int, choices=SEEDS, required=True)
    recipe.add_argument("--steps", type=int, default=STEPS)
    recipe.add_argument("--platform", choices=("gpu", "tpu"), required=True)
    recipe.add_argument("--gpu", type=int)
    recipe.add_argument("--worker-python", type=Path)
    recipe.add_argument(
        "--negative-scope",
        choices=("local", "global"),
        default="local",
    )
    recipe.add_argument("--torch-compile", action="store_true")
    for name in ("representax", "sentence-transformers"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, choices=SEEDS, required=True)
        command.add_argument("--steps", type=int, default=STEPS)
        command.add_argument("--platform", choices=("gpu", "tpu"), required=True)
        command.add_argument("--gpu", type=int)
        if name == "representax":
            command.add_argument(
                "--negative-scope", choices=("local", "global"), required=True
            )
        else:
            command.add_argument("--torch-compile", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    parsed = _parser().parse_args(arguments)
    if parsed.command == "environment":
        print(json.dumps(_environment_state(), sort_keys=True))
        return
    if parsed.command == "prepare-data":
        _prepare_data(parsed.output, parsed.checkpoint, parsed.rows)
        return
    if parsed.command == "suite":
        _run_suite(parsed)
        return
    if parsed.command == "recipe":
        _run_recipe(parsed)
        return
    if parsed.platform == "gpu":
        if parsed.gpu is None or parsed.gpu < 0:
            raise ValueError("GPU runs require one non-negative --gpu index")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(parsed.gpu)
    elif parsed.gpu is not None:
        raise ValueError("TPU runs do not accept --gpu")
    _verify_checkpoint(parsed.checkpoint)
    if parsed.command == "representax":
        _representax(
            checkpoint=parsed.checkpoint,
            data=parsed.data,
            output=parsed.output,
            seed=parsed.seed,
            steps=parsed.steps,
            scope=parsed.negative_scope,
            platform=parsed.platform,
        )
    else:
        if parsed.platform == "tpu" and parsed.torch_compile:
            raise ValueError("TorchInductor is a GPU-only reference control")
        _sentence_transformers(
            checkpoint=parsed.checkpoint,
            data=parsed.data,
            output=parsed.output,
            seed=parsed.seed,
            steps=parsed.steps,
            platform=parsed.platform,
            torch_compile=parsed.torch_compile,
        )


if __name__ == "__main__":
    main()
