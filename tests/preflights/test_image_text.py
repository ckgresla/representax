from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
from experiments.preflights.image_text import (
    GRAD_CACHE_MICRO_BATCH,
    ImageTextEvaluationCollator,
    ImageTextRetrievalCollator,
    _batch_unique_caption_order,
    _distinct_captions,
    _parser,
    _representax_job,
    _select_coco_rows,
    frozen_contract,
)
from PIL import Image

from representax.tasks.retrieval import ProcessLocalRetrievalBatch


class _Processor:
    def data_contract(self):
        return {"processor": "test"}

    def __call__(self, values, *, route):
        del route
        return jnp.zeros((len(values), 3), dtype=jnp.float32)


def _image(path) -> None:
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(path)


def test_frozen_image_text_contract() -> None:
    contract = frozen_contract()

    assert contract.model_id == "sentence-transformers/clip-ViT-B-32"
    assert contract.model_revision == "327ab6726d33c0e22f920c83f2ff9e4bd38ca37f"
    assert contract.train_dataset["repo_id"] == "phiyodr/coco2017"
    assert contract.evaluation_dataset["repo_id"] == "mteb/flickr30kt2i"
    assert contract.reference_version == "5.6.1"
    assert contract.global_batch_size == 512
    assert contract.image_shape == (3, 224, 224)


def test_training_captions_are_distinct_and_nonempty() -> None:
    assert _distinct_captions(
        (" first ", "", "first", "second", "third", "fourth"), count=4
    ) == ("first", "second", "third", "fourth")


def test_coco_selection_skips_duplicate_media_and_local_captions() -> None:
    rows = (
        {"image_id": 1, "captions": ("a", "b", "c", "d")},
        {"image_id": 1, "captions": ("e", "f", "g", "h")},
        {"image_id": 2, "captions": ("a", "i", "j", "k")},
        {"image_id": 3, "captions": ("e", "f", "g", "h")},
    )

    selected = _select_coco_rows(rows, count=2)

    assert [(index, captions) for index, _, captions in selected] == [
        (0, ("a", "b", "c", "d")),
        (2, ("a", "i", "j", "k")),
    ]


def test_repeated_captions_are_distributed_across_complete_batches() -> None:
    rows = [
        {"image_id": index, "caption": caption}
        for index, caption in enumerate(("same", "same", "a", "b", "c", "d"))
    ]

    ordered = _batch_unique_caption_order(rows, batch_size=3, seed=7)

    batches = (ordered[:3], ordered[3:])
    assert all(len({row["image_id"] for row in batch}) == 3 for batch in batches)
    assert all(len({row["caption"] for row in batch}) == 3 for batch in batches)


def test_image_text_collators_preserve_modalities_and_validity(tmp_path) -> None:
    _image(tmp_path / "image.jpg")
    processor = _Processor()
    training = ImageTextRetrievalCollator(
        processor=processor,
        root_directory=tmp_path,
    )(
        (
            {"caption": "first", "image": "image.jpg"},
            {"caption": "second", "image": "image.jpg"},
        )
    )
    assert training.query.shape == training.document.shape == (2, 3)
    np.testing.assert_array_equal(training.positive_mask, np.eye(2, dtype=bool))

    collator = ImageTextEvaluationCollator(
        processor=processor,
        root_directory=tmp_path,
    )
    queries = collator(
        (
            {
                "kind": "query",
                "identifier": 10,
                "text": "caption",
                "image": "",
                "valid": True,
            },
            {
                "kind": "query",
                "identifier": -1,
                "text": "",
                "image": "",
                "valid": False,
            },
        )
    )
    documents = collator(
        (
            {
                "kind": "document",
                "identifier": 20,
                "text": "",
                "image": "image.jpg",
                "valid": True,
            },
        )
    )
    assert queries.kind == "query"
    assert documents.kind == "document"
    np.testing.assert_array_equal(queries.valid, [True, False])
    np.testing.assert_array_equal(documents.ids, [20])


def test_image_collator_preprocesses_only_process_local_rows(
    tmp_path, monkeypatch
) -> None:
    _image(tmp_path / "image.jpg")
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(jax, "process_index", lambda: 0)

    batch = ImageTextRetrievalCollator(
        processor=_Processor(),
        root_directory=tmp_path,
    )(
        tuple(
            {"caption": f"caption-{index}", "image": "image.jpg"} for index in range(4)
        )
    )

    assert isinstance(batch, ProcessLocalRetrievalBatch)
    assert batch.query.shape == batch.document.shape == (2, 3)
    np.testing.assert_array_equal(
        batch.positive_mask,
        [[True, False, False, False], [False, True, False, False]],
    )


def test_representax_job_uses_run_job_grad_cache_and_verified_export(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 2048,
                "unique_image_ids": True,
                "unique_captions": True,
                "relevant_documents": {"1000": [0]},
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    source = checkpoint / "0_CLIPModel"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")

    job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data,
        steps=4,
        seed=7,
        symmetric=True,
        warmup_steps=2,
        training_file="train-seed-7.jsonl",
    )

    assert job.model.target == "representax.models.clip:load_clip"
    assert job.training.global_batch_size == 512
    assert job.training.grad_cache is not None
    assert job.training.grad_cache.micro_batch_size == GRAD_CACHE_MICRO_BATCH
    assert job.loss.negative_scope == "global"
    assert job.loss.symmetric
    assert job.optimization.schedule.parameters["warmup_steps"] == 2
    assert job.data.distribution.sources[0].uri.endswith("train-seed-7.jsonl")
    assert job.checkpointing is not None and job.checkpointing.every == 2
    assert job.evaluation is not None and job.evaluation.on_start
    assert job.evaluation.on_end
    assert job.export.huggingface is not None
    assert job.export.huggingface.source_checkpoint == str(source)
    assert job.export.huggingface.verify_reload

    local_job = _representax_job(
        checkpoint=checkpoint,
        data_directory=data,
        steps=4,
        seed=7,
        negative_scope="local",
    )
    assert local_job.loss.negative_scope == "local"


def test_pair_command_defaults_to_gpu_one() -> None:
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

    assert arguments.gpu == 1
