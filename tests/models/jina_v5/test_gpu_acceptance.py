"""Physical training, resume, and export acceptance for Jina v5 Omni."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest

from representax import load_inference_bundle
from representax.config import (
    BatchConfig,
    CheckpointConfig,
    ComponentConfig,
    DataConfig,
    ExportConfig,
    HuggingFaceExportConfig,
    JobConfig,
    LoggingConfig,
    LoRAConfig,
    ModelConfig,
    OptimizationConfig,
    PrecisionConfig,
    TrainingConfig,
)
from representax.core import Route
from representax.data import identity, mix, source
from representax.models.jina_v5 import (
    JinaV5OmniCheckpointAdapter,
    load_jina_v5_omni,
)
from representax.tasks.pairwise import (
    CosineRegressionConfig,
    PairwiseConfig,
    pairwise_batch,
)
from representax.train import run_job

pytestmark = pytest.mark.performance

_CHECKPOINT_ENVIRONMENTS = {
    "nano": "REPRESENTAX_JINA_V5_OMNI_NANO_CHECKPOINT",
    "small": "REPRESENTAX_JINA_V5_OMNI_SMALL_CHECKPOINT",
}


def collate_pairs(examples: Sequence[dict], *, processor):
    return pairwise_batch(
        left=processor([example["left"] for example in examples], route=Route.QUERY),
        right=processor(
            [example["right"] for example in examples], route=Route.DOCUMENT
        ),
        labels=np.asarray([example["label"] for example in examples], np.float32),
    )


def _checkpoint(variant: str) -> Path:
    environment = _CHECKPOINT_ENVIRONMENTS[variant]
    value = os.environ.get(environment)
    if value is None:
        pytest.skip(f"set {environment} for Jina v5 Omni GPU acceptance")
    return Path(value)


def _write_data(path: Path) -> None:
    rows = (
        {"left": "a quiet harbor", "right": "calm water", "label": 0.8},
        {"left": "living cells", "right": "a rainy street", "label": 0.1},
        {"left": "a red flower", "right": "a blooming rose", "label": 0.9},
        {"left": "a mountain", "right": "computer source code", "label": 0.0},
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _job(checkpoint: Path, data: Path) -> JobConfig:
    return JobConfig(
        name="jina-v5-omni-lifecycle",
        model=ModelConfig(
            target="representax.models.jina_v5.load_jina_v5_omni",
            parameters={
                "model_name_or_path": str(checkpoint),
                "local_files_only": True,
                "sequence_length_buckets": [32],
                "patch_count_buckets": [8],
                "audio_chunk_count_buckets": [1],
                "audio_token_count_buckets": [32],
            },
        ),
        task=PairwiseConfig(
            left_route=Route.QUERY,
            right_route=Route.DOCUMENT,
        ),
        loss=CosineRegressionConfig(),
        optimization=OptimizationConfig(
            optimizer=ComponentConfig(
                target="optax.adamw",
                parameters={"learning_rate": 1e-4, "weight_decay": 0.0},
            )
        ),
        data=DataConfig(
            distribution=mix(
                source(str(data), map=identity),
                shuffle=False,
            ),
            collate=ComponentConfig(
                target=("tests.models.jina_v5.test_gpu_acceptance.collate_pairs")
            ),
            num_threads=0,
            prefetch_buffer_size=0,
        ),
        training=TrainingConfig(
            global_batch_size=2,
            max_steps=2,
            seed=7,
            batch=BatchConfig(micro_batch_size=2),
            adapter=LoRAConfig(rank=4, alpha=8.0, target_pattern="text"),
            precision=PrecisionConfig.bfloat16_mixed(),
            activation_rematerialization="full",
        ),
        checkpointing=CheckpointConfig(
            every=1,
            keep=2,
            asynchronous=False,
        ),
        logging=LoggingConfig(timing=True),
        export=ExportConfig(
            huggingface=HuggingFaceExportConfig(
                source_checkpoint=str(checkpoint),
                adapter=ComponentConfig(
                    target=("representax.models.jina_v5.JinaV5OmniCheckpointAdapter")
                ),
            )
        ),
    )


@pytest.mark.parametrize("variant", ["nano", "small"])
def test_released_checkpoint_updates_resumes_and_reloads(
    variant: str, tmp_path: Path
) -> None:
    if jax.default_backend() != "gpu":
        pytest.skip("Jina v5 Omni lifecycle acceptance requires a GPU")
    checkpoint = _checkpoint(variant)
    data = tmp_path / "pairs.jsonl"
    _write_data(data)
    job = _job(checkpoint, data)
    run_directory = tmp_path / "run"

    paused = run_job(job, run_directory, stop_after=1)
    assert paused.completed_iterations == 1
    resumed = run_job(job, run_directory, resume=True)
    assert resumed.completed_iterations == 2
    assert resumed.resumed
    assert resumed.inference_bundle is not None

    native, restored_job = load_inference_bundle(resumed.inference_bundle)
    assert restored_job == job
    expected_leaves = [
        value
        for value in jax.tree.leaves(resumed.selected_model)
        if eqx.is_array(value)
    ]
    actual_leaves = [value for value in jax.tree.leaves(native) if eqx.is_array(value)]
    assert len(actual_leaves) == len(expected_leaves)
    for actual, expected in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    exported = resumed.inference_bundle / "huggingface"
    restored_model, processor = load_jina_v5_omni(
        exported,
        local_files_only=True,
        sequence_length_buckets=(32,),
        patch_count_buckets=(8,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(32,),
    )
    batch = processor(["a quiet harbor"], route=Route.QUERY)
    embedding = restored_model.encode(batch, route=Route.QUERY)
    assert bool(np.all(np.isfinite(np.asarray(embedding))))
    assert JinaV5OmniCheckpointAdapter().state_dict(restored_model)
    subprocess.run(
        [
            sys.executable,
            "tests/models/jina_v5/huggingface_reload.py",
            str(exported),
        ],
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd() / "src")},
    )
