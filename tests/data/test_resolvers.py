"""Built-in Hugging Face and local artifact resolver tests."""

from __future__ import annotations

import json

import pytest

from representax import data

datasets = pytest.importorskip("datasets")
grain = pytest.importorskip("grain")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

MAPPED_RECORDS: list[int] = []


def project_record(record):
    value = int(record["value"])
    MAPPED_RECORDS.append(value)
    return {"value": value * 10}


def _write_local_artifact(path, kind: str) -> None:
    rows = [{"value": 1}, {"value": 2}, {"value": 3}]
    if kind == "jsonl":
        path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    elif kind == "parquet":
        pq.write_table(pa.Table.from_pylist(rows), path)
    elif kind == "arrow":
        table = pa.Table.from_pylist(rows)
        with (
            pa.OSFile(str(path), "wb") as sink,
            pa.ipc.new_stream(sink, table.schema) as writer,
        ):
            writer.write_table(table)
    else:  # pragma: no cover - test helper contract
        raise AssertionError(kind)


@pytest.mark.parametrize(
    ("kind", "suffix"),
    (("jsonl", ".jsonl"), ("parquet", ".parquet"), ("arrow", ".arrow")),
)
def test_local_artifacts_map_lazily_and_shuffle_deterministically(
    tmp_path, kind, suffix
):
    artifact_path = tmp_path / f"records{suffix}"
    _write_local_artifact(artifact_path, kind)
    recipe = data.mix(
        data.source(artifact_path.as_uri(), map=project_record),
        seed=19,
    )

    MAPPED_RECORDS.clear()
    first = data.build_grain_dataset(recipe)
    second = data.build_grain_dataset(recipe)
    assert isinstance(first, grain.MapDataset)
    assert MAPPED_RECORDS == []

    first_epoch = [first[index] for index in range(len(first))]
    second_epoch = [second[index] for index in range(len(second))]
    assert first_epoch == second_epoch
    assert sorted(record["value"] for record in first_epoch) == [10, 20, 30]


def test_huggingface_resolver_forwards_pinned_identity(monkeypatch):
    calls = []
    source_dataset = datasets.Dataset.from_dict({"value": [4, 5]})

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return source_dataset

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    recipe = data.mix(
        data.source(
            "hf://organization/dataset",
            revision="0123456789abcdef",
            split="validation",
            subset="english",
            map=project_record,
        ),
        shuffle=False,
    )

    resolved = data.build_grain_dataset(recipe)

    assert resolved[0] == {"value": 40}
    assert calls == [
        (
            "organization/dataset",
            {
                "name": "english",
                "revision": "0123456789abcdef",
                "split": "validation",
                "streaming": False,
                "keep_in_memory": False,
            },
        )
    ]


def test_local_dataset_directory_resolves_named_split(tmp_path):
    dataset_path = tmp_path / "records"
    dataset_path.mkdir()
    _write_local_artifact(dataset_path / "train.jsonl", "jsonl")
    artifact = data.source(
        dataset_path.as_uri(),
        split="train",
        map=project_record,
    )

    resolved = data.build_grain_dataset(data.mix(artifact, shuffle=False))

    assert [resolved[index]["value"] for index in range(len(resolved))] == [
        10,
        20,
        30,
    ]


@pytest.mark.parametrize("missing", ("revision", "split"))
def test_huggingface_resolver_requires_pinned_revision_and_split(missing):
    values = {
        "revision": "0123456789abcdef",
        "split": "train",
    }
    values[missing] = None
    artifact = data.source(
        "hf://organization/dataset",
        revision=values["revision"],
        split=values["split"],
        map=project_record,
    )

    with pytest.raises(ValueError, match=missing):
        data.resolve_huggingface(artifact)


def test_custom_resolver_and_mapper_can_override_builtins():
    artifact = data.source("hf://ignored/source", map=project_record)
    recipe = data.mix(artifact, shuffle=False)
    mapper_path = artifact.mapper_id

    resolved = data.build_grain_dataset(
        recipe,
        resolvers={"hf": lambda _artifact: [{"value": 7}]},
        mappers={mapper_path: lambda record: {"value": record["value"] + 1}},
    )

    assert resolved[0] == {"value": 8}


def test_local_resolver_rejects_unknown_formats(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("value\n1\n")
    artifact = data.source(path.as_uri(), map=project_record)

    with pytest.raises(ValueError, match="unsupported local artifact"):
        data.resolve_local(artifact)
