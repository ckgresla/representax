from __future__ import annotations

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from experiments.paper.video_text import (
    GRAD_CACHE_MICRO_BATCH,
    PREFLIGHT_BATCH_SIZE,
    VIDEO_FPS,
    VideoTextEvaluationCollator,
    VideoTextRetrievalCollator,
    _parser,
    _reference_video,
    _representax_job,
    frozen_contract,
)
from tests.models.qwen2_5_omni.test_model import tiny_config

from representax.config import DDPConfig, LoRAConfig
from representax.models.qwen2_5_omni import (
    Qwen2_5OmniEncoder,
    batch_from_processor_output,
)
from representax.tasks.retrieval import MNRTask, retrieval_batch
from representax.train import GradCache, build_train_step, init_train_state


class _Processor:
    def data_contract(self):
        return {"processor": "test"}

    def __call__(self, values, *, route):
        del route
        return jnp.zeros((len(values), 3), dtype=jnp.float32)


def _video(path) -> None:
    np.save(path, np.zeros((16, 8, 8, 3), dtype=np.uint8))


def test_frozen_video_text_contract() -> None:
    contract = frozen_contract()

    assert contract.model_id == "LCO-Embedding/LCO-Embedding-Omni-3B-2605"
    assert contract.model_revision == "5f6b5329da5141367da30e06a9826d1322d6c9b2"
    assert contract.train_dataset["repo_id"] == "VLM2Vec/MSR-VTT"
    assert contract.train_dataset["subset"] == "train_7k"
    assert contract.evaluation_dataset["repo_id"] == "mteb/MSR-VTT"
    assert contract.reference_version == "5.6.1"
    assert contract.global_batch_size == 128
    assert contract.video_frames == 16


def test_video_text_collators_preserve_routes_and_validity(tmp_path) -> None:
    _video(tmp_path / "video.npy")
    processor = _Processor()
    training = VideoTextRetrievalCollator(
        processor=processor,
        root_directory=tmp_path,
    )(
        (
            {"video": "video.npy", "caption": "first"},
            {"video": "video.npy", "caption": "second"},
        )
    )
    assert training.query.shape == training.document.shape == (2, 3)
    np.testing.assert_array_equal(training.positive_mask, np.eye(2, dtype=bool))

    collator = VideoTextEvaluationCollator(
        processor=processor,
        root_directory=tmp_path,
    )
    queries = collator(
        (
            {
                "kind": "query",
                "identifier": 10,
                "video": "video.npy",
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
                "video": "",
                "text": "caption",
                "valid": True,
            },
        )
    )
    assert queries.kind == "query"
    assert documents.kind == "document"
    np.testing.assert_array_equal(queries.ids, [10])
    np.testing.assert_array_equal(documents.ids, [20])


def test_reference_video_preserves_frozen_frames_and_metadata(tmp_path) -> None:
    _video(tmp_path / "video.npy")

    value = _reference_video(tmp_path, "video.npy")

    assert value["array"].shape == (16, 8, 8, 3)
    assert value["video_metadata"] == {
        "fps": VIDEO_FPS,
        "total_num_frames": 16,
        "frames_indices": list(range(16)),
    }


def test_video_batch_is_row_major_and_grad_cache_matches_direct_update() -> None:
    config = tiny_config()
    patch_count = 2 * 16
    queries = batch_from_processor_output(
        {
            "input_ids": np.asarray(
                (
                    (8, 4, 2, 2, 2, 2, 5, 9),
                    (8, 4, 2, 2, 2, 2, 5, 9),
                )
            ),
            "attention_mask": np.ones((2, 8), dtype=np.int32),
            "pixel_values_videos": (
                np.arange(
                    patch_count * config.vision.patch_dimension,
                    dtype=np.float32,
                ).reshape((patch_count, config.vision.patch_dimension))
                / 100
            ),
            "video_grid_thw": np.asarray(((1, 4, 4), (1, 4, 4))),
            "video_second_per_grid": np.asarray((0.5, 0.5), dtype=np.float32),
        },
        config,
        sequence_length_buckets=(8,),
        patch_count_buckets=(16,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(2,),
    )
    documents = batch_from_processor_output(
        {
            "input_ids": np.asarray(((8, 10, 9), (8, 11, 9))),
            "attention_mask": np.ones((2, 3), dtype=np.int32),
        },
        config,
        sequence_length_buckets=(8,),
        patch_count_buckets=(16,),
        audio_chunk_count_buckets=(1,),
        audio_token_count_buckets=(2,),
    )
    assert queries.pixel_values is not None
    assert queries.pixel_values.shape == (
        2,
        16,
        config.vision.patch_dimension,
    )
    assert all(
        leaf.shape[0] == 2 for leaf in jax.tree.leaves(queries) if eqx.is_array(leaf)
    )

    model = Qwen2_5OmniEncoder.init(
        config,
        key=jax.random.key(37),
        rematerialization="none",
    )
    optimizer = optax.adamw(1e-3)
    state = init_train_state(model, optimizer)
    batch = retrieval_batch(
        query=queries,
        document=documents,
        positive_mask=np.eye(2, dtype=np.bool_),
    )
    task = MNRTask(scale=7.0)
    direct = build_train_step(task, optimizer, donate_state=False)
    cached = build_train_step(
        task,
        optimizer,
        execution=GradCache(query_chunk_size=1, document_chunk_size=1),
        donate_state=False,
    )
    expected = direct(state, batch, None)
    actual = cached(state, batch, None)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if eqx.is_array(actual_leaf):
            np.testing.assert_allclose(
                actual_leaf,
                expected_leaf,
                rtol=1e-4,
                atol=1e-5,
            )


def test_representax_video_job_uses_one_gpu_grad_cache_and_verified_export(
    tmp_path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 8,
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
    )

    assert job.model.target == "representax.models.qwen2_5_omni:load_qwen2_5_omni"
    assert job.model.parameters["sequence_length_buckets"] == [256]
    assert job.model.parameters["patch_count_buckets"] == [800]
    assert job.model.parameters["video_min_pixels"] == 32 * 28 * 28
    assert job.model.parameters["video_max_pixels"] == 32 * 28 * 28
    assert job.training.global_batch_size == PREFLIGHT_BATCH_SIZE
    assert job.training.mesh.axis_shapes == (1,)
    assert type(job.training.sharding) is DDPConfig
    assert job.training.batch.micro_batch_size == PREFLIGHT_BATCH_SIZE
    assert job.training.grad_cache is not None
    assert job.training.grad_cache.micro_batch_size == GRAD_CACHE_MICRO_BATCH
    assert type(job.training.adapter) is LoRAConfig
    assert job.training.adapter.target_pattern == "text"
    assert job.checkpointing is not None and job.checkpointing.every == 2
    assert job.evaluation is not None and job.evaluation.on_start
    assert job.evaluation.on_end
    assert job.export.huggingface is not None
    assert job.export.huggingface.source_checkpoint == str(checkpoint)
    assert job.export.huggingface.verify_reload


def test_pair_command_defaults_to_assigned_gpu_zero() -> None:
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

    assert arguments.gpu == 0
    assert arguments.batch_size == PREFLIGHT_BATCH_SIZE
