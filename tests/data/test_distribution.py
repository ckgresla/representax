"""Data distribution contract tests."""

from dataclasses import dataclass
from operator import setitem
from typing import Any, cast

import pytest

from representax import data
from representax.core import Modality


def named_mapper(record):
    return record


def test_mixture_is_normalized_and_fingerprint_is_stable():
    first = data.source(
        "hf://organization/one",
        revision="abc123",
        map=named_mapper,
        name="one",
    )
    second = data.source(
        "s3://bucket/two/*.parquet",
        revision="version-2",
        map="project.mappers.to_example",
        name="two",
    )
    distribution = data.mix(first, second, weights=(7.0, 3.0), seed=11)

    assert distribution.normalized_weights == pytest.approx((0.7, 0.3))
    assert distribution.fingerprint() == distribution.fingerprint()
    assert distribution.sources[0].scheme == "hf"
    assert distribution.sources[1].scheme == "s3"


def test_distribution_rejects_anonymous_mapper():
    with pytest.raises(ValueError, match="named importable"):
        data.source("file:///tmp/example", map=lambda value: value)

    def nested_mapper(value):
        return value

    with pytest.raises(ValueError, match="named importable"):
        data.source("file:///tmp/example", map=nested_mapper)


def test_mix_rehydrates_config_mappings():
    distribution = data.mix(
        {
            "uri": "file:///data/examples.jsonl",
            "mapper": "project.mappers.to_example",
            "revision": "v1",
        }
    )

    assert isinstance(distribution.sources[0], data.DataSourceConfig)


def test_distribution_config_defaults_to_equal_source_weights():
    distribution = data.DataDistributionConfig.model_validate(
        {
            "sources": [
                {
                    "uri": "file:///data/examples.jsonl",
                    "mapper": "project.mappers.to_example",
                },
                {
                    "uri": "file:///data/more.jsonl",
                    "mapper": "project.mappers.to_example",
                },
            ]
        }
    )

    assert distribution.weights == (1.0, 1.0)
    assert distribution.normalized_weights == (0.5, 0.5)


def test_mapper_import_and_subset_are_part_of_the_distribution_contract():
    mapper = data.load_mapper("tests.data.test_distribution.named_mapper")
    first = data.source(
        "hf://organization/dataset",
        revision="abc123",
        split="train",
        subset="english",
        map=mapper,
    )
    second = data.source(
        "hf://organization/dataset",
        revision="abc123",
        split="train",
        subset="german",
        map=mapper,
    )

    assert mapper is named_mapper
    assert data.mix(first).fingerprint() != data.mix(second).fingerprint()


def test_source_uri_parsing_is_explicit():
    assert data.huggingface_dataset_id("hf://organization/dataset") == (
        "organization/dataset"
    )
    assert data.local_path("file:///tmp/example%20data.jsonl") == data.local_path(
        "/tmp/example data.jsonl"
    )
    with pytest.raises(ValueError, match="query strings"):
        data.huggingface_dataset_id("hf://organization/dataset?split=train")


def test_artifacts_compose_into_task_specific_samples_without_a_shared_schema():
    @dataclass(frozen=True)
    class RetrievalSample:
        query: dict[str, data.Artifact]
        document: data.Artifact

    sample = RetrievalSample(
        query={"instruction": data.Artifact.text("find the matching image")},
        document=data.Artifact.ref(
            Modality.IMAGE,
            source="images",
            key="00042.jpg",
            metadata={"width": 1024, "height": 768},
        ),
    )

    assert sample.query["instruction"].data == "find the matching image"
    assert sample.document.modality == Modality.IMAGE
    assert sample.document.source == "images"
    assert sample.document.key == "00042.jpg"
    assert sample.document.metadata == {"width": 1024, "height": 768}
    with pytest.raises(TypeError):
        setitem(cast(Any, sample.document.metadata), "width", 512)


def test_artifact_requires_exactly_one_inline_value_or_lazy_reference():
    with pytest.raises(ValueError, match="inline data or one reference"):
        data.Artifact(modality=Modality.AUDIO)
    with pytest.raises(ValueError, match="inline data or one reference"):
        data.Artifact(
            modality=Modality.AUDIO,
            data=b"wav",
            source="audio",
            key="clip.wav",
        )
    with pytest.raises(ValueError, match="both source and key"):
        data.Artifact(modality=Modality.VIDEO, source="video")
