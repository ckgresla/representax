from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
from experiments.preflights.image_text import (
    GRAD_CACHE_MICRO_BATCH,
    ImageTextEvaluationCollator,
    ImageTextRetrievalCollator,
    _parser,
    _representax_job,
    frozen_contract,
)
from PIL import Image


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


def test_representax_job_uses_run_job_grad_cache_and_verified_export(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(
        json.dumps(
            {
                "training_presentations": 2048,
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
    )

    assert job.model.target == "representax.models.clip:load_clip"
    assert job.training.global_batch_size == 512
    assert job.training.grad_cache is not None
    assert job.training.grad_cache.micro_batch_size == GRAD_CACHE_MICRO_BATCH
    assert job.checkpointing is not None and job.checkpointing.every == 2
    assert job.evaluation is not None and job.evaluation.on_start
    assert job.evaluation.on_end
    assert job.export.huggingface is not None
    assert job.export.huggingface.source_checkpoint == str(source)
    assert job.export.huggingface.verify_reload


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
