"""Data recipe contract tests."""

import pytest

from representax import data


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
    recipe = data.mix(first, second, weights=(7.0, 3.0), seed=11)

    assert recipe.normalized_weights == pytest.approx((0.7, 0.3))
    assert recipe.fingerprint() == recipe.fingerprint()
    assert recipe.sources[0].scheme == "hf"
    assert recipe.sources[1].scheme == "s3"


def test_recipe_rejects_anonymous_mapper():
    with pytest.raises(ValueError, match="named importable"):
        data.source("file:///tmp/example", map=lambda value: value)

    def nested_mapper(value):
        return value

    with pytest.raises(ValueError, match="named importable"):
        data.source("file:///tmp/example", map=nested_mapper)


def test_mix_rehydrates_config_mappings():
    recipe = data.mix(
        {
            "uri": "file:///data/examples.jsonl",
            "mapper": "project.mappers.to_example",
            "revision": "v1",
        }
    )

    assert isinstance(recipe.sources[0], data.ArtifactSource)


def test_mapper_import_and_subset_are_part_of_the_recipe_contract():
    mapper = data.load_mapper("tests.data.test_recipe.named_mapper")
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


def test_recipe_uri_parsing_is_explicit():
    assert data.huggingface_dataset_id("hf://organization/dataset") == (
        "organization/dataset"
    )
    assert data.local_path("file:///tmp/example%20data.jsonl") == data.local_path(
        "/tmp/example data.jsonl"
    )
    with pytest.raises(ValueError, match="query strings"):
        data.huggingface_dataset_id("hf://organization/dataset?split=train")
