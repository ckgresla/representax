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


def test_mix_rehydrates_config_mappings():
    recipe = data.mix(
        {
            "uri": "file:///data/examples.jsonl",
            "mapper": "project.mappers.to_example",
            "revision": "v1",
        }
    )

    assert isinstance(recipe.sources[0], data.ArtifactSource)
