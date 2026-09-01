from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from experiments.preflights.audio_text import (
    GRAD_CACHE_MICRO_BATCH,
    PREFLIGHT_BATCH_SIZE,
    SAMPLE_RATE,
    AudioTextEvaluationCollator,
    AudioTextRetrievalCollator,
    _normalize_audio,
    _parser,
    _representax_job,
    frozen_contract,
)

from representax.config import DDPConfig, FSDPConfig, LoRAConfig


class _Processor:
    def data_contract(self):
        return {"processor": "test"}

    def __call__(self, values, *, route):
        del route
        return jnp.zeros((len(values), 3), dtype=jnp.float32)


def test_frozen_audio_text_contract() -> None:
    contract = frozen_contract()

    assert contract.model_id == "LCO-Embedding/LCO-Embedding-Omni-3B-2605"
    assert contract.model_revision == "5f6b5329da5141367da30e06a9826d1322d6c9b2"
    assert contract.train_dataset["repo_id"] == "OpenSound/AudioCaps"
    assert contract.evaluation_dataset["repo_id"] == "mteb/audiocaps_a2t"
    assert contract.reference_version == "5.6.1"
    assert contract.global_batch_size == 256
    assert contract.audio_seconds == 10


def test_audio_normalization_is_mono_resampled_and_fixed_length() -> None:
    stereo = np.stack(
        (
            np.arange(24_000, dtype=np.int16),
            np.arange(24_000, dtype=np.int16),
        ),
        axis=1,
    )

    actual = _normalize_audio(stereo, 24_000, seconds=2)

    assert actual.shape == (2 * SAMPLE_RATE,)
    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    assert np.any(actual[:SAMPLE_RATE] != 0)
    assert np.all(actual[-SAMPLE_RATE:] == 0)


def test_audio_text_collators_preserve_routes_and_validity(tmp_path) -> None:
    audio = np.linspace(-1, 1, SAMPLE_RATE, dtype=np.float32)
    np.save(tmp_path / "audio.npy", audio)
    processor = _Processor()
    training = AudioTextRetrievalCollator(
        processor=processor,
        root_directory=tmp_path,
    )(
        (
            {"audio": "audio.npy", "caption": "first"},
            {"audio": "audio.npy", "caption": "second"},
        )
    )
    assert training.query.shape == training.document.shape == (2, 3)
    np.testing.assert_array_equal(training.positive_mask, np.eye(2, dtype=bool))

    collator = AudioTextEvaluationCollator(
        processor=processor,
        root_directory=tmp_path,
    )
    queries = collator(
        (
            {
                "kind": "query",
                "identifier": 10,
                "audio": "audio.npy",
                "text": "",
                "valid": True,
            },
        )
    )
    documents = collator(
        (
            {
                "kind": "document",
                "identifier": 20,
                "audio": "",
                "text": "caption",
                "valid": True,
            },
        )
    )
    assert queries.kind == "query"
    assert documents.kind == "document"
    np.testing.assert_array_equal(queries.ids, [10])
    np.testing.assert_array_equal(documents.ids, [20])


def test_representax_audio_job_uses_ddp_grad_cache_and_verified_export(
    tmp_path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 128,
                "relevant_documents": {"0": [0]},
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data,
        steps=4,
        seed=7,
        batch_size=PREFLIGHT_BATCH_SIZE,
    )

    assert job.model.target == ("representax.models.qwen2_5_omni:load_qwen2_5_omni")
    assert job.training.global_batch_size == PREFLIGHT_BATCH_SIZE
    assert job.training.mesh.axis_shapes == (2,)
    assert type(job.training.sharding) is DDPConfig
    assert job.training.batch.micro_batch_size == PREFLIGHT_BATCH_SIZE // 2
    assert job.training.grad_cache is not None
    assert job.training.grad_cache.micro_batch_size == GRAD_CACHE_MICRO_BATCH
    assert job.training.adapter is not None
    assert type(job.training.adapter) is LoRAConfig
    assert job.training.adapter.target_pattern == "text"
    assert job.checkpointing is not None and job.checkpointing.every == 2
    assert job.evaluation is not None and job.evaluation.on_start
    assert job.evaluation.on_end
    assert job.export.huggingface is not None
    assert job.export.huggingface.source_checkpoint == str(checkpoint)
    assert job.export.huggingface.verify_reload


def test_representax_audio_scaling_probe_can_use_one_gpu_without_export(
    tmp_path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 128,
                "relevant_documents": {"0": [0]},
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data,
        steps=4,
        seed=7,
        batch_size=PREFLIGHT_BATCH_SIZE,
        world_size=1,
        export_enabled=False,
    )

    assert job.training.mesh.axis_shapes == (1,)
    assert type(job.training.sharding) is DDPConfig
    assert job.training.batch.micro_batch_size == PREFLIGHT_BATCH_SIZE
    assert not job.export.enabled
    assert job.export.huggingface is None


def test_representax_audio_job_retains_fsdp_capacity_fallback(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 128,
                "relevant_documents": {"0": [0]},
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data,
        steps=4,
        seed=7,
        batch_size=PREFLIGHT_BATCH_SIZE,
        sharding="fsdp",
    )

    assert type(job.training.sharding) is FSDPConfig
    assert job.training.batch.micro_batch_size == PREFLIGHT_BATCH_SIZE


def test_pair_command_defaults_to_gpus_zero_and_one() -> None:
    arguments = _parser().parse_args(
        [
            "pair",
            "--checkpoint",
            "/checkpoint",
            "--data-directory",
            "/data",
            "--output",
            "/output",
        ]
    )

    assert arguments.representax_gpus == "0,1"
    assert arguments.representax_sharding == "ddp"
    assert arguments.reference_gpu == 1
    assert arguments.batch_size == PREFLIGHT_BATCH_SIZE
