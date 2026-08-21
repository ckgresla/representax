"""Built-in Hugging Face and local artifact resolver tests."""

from __future__ import annotations

import json
from typing import cast

import pytest

from representax import data

datasets = pytest.importorskip("datasets")
grain = pytest.importorskip("grain")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

MAPPED_RECORDS: list[int] = []


@pytest.fixture(autouse=True)
def _isolated_datasets_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        datasets.config,
        "HF_DATASETS_CACHE",
        str(tmp_path / "datasets-cache"),
    )


def project_record(record):
    value = int(record["value"])
    MAPPED_RECORDS.append(value)
    return {"value": value * 10}


def alternative_project_record(record):
    return {"value": int(record["value"]) * 100}


def collate_records(records):
    return {"values": [record["value"] for record in records]}


def collate_values(values):
    return tuple(values)


class StatefulCollator:
    def __init__(self, width):
        self.width = width

    def __call__(self, records):
        return collate_records(records)

    def data_contract(self):
        return {"width": self.width}


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
    distribution = data.mix(
        data.source(artifact_path.as_uri(), map=project_record),
        seed=19,
    )

    MAPPED_RECORDS.clear()
    first = data.build_dataset(distribution)
    second = data.build_dataset(distribution)
    assert isinstance(first, grain.MapDataset)
    assert MAPPED_RECORDS == []

    first_epoch = [first[index] for index in range(len(first))]
    second_epoch = [second[index] for index in range(len(second))]
    assert first_epoch == second_epoch
    assert sorted(record["value"] for record in first_epoch) == [10, 20, 30]


def test_local_parquet_uses_row_groups_without_a_datasets_cache(tmp_path, monkeypatch):
    artifact_path = tmp_path / "records.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"value": index} for index in range(7)]),
        artifact_path,
        row_group_size=3,
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "local Parquet must not construct a Hugging Face Arrow cache"
        ),
    )

    source = data.resolve_local(data.source(artifact_path.as_uri(), map=project_record))

    assert isinstance(source, data.ParquetSource)
    assert len(source) == 7
    assert source[0] == {"value": 0}
    assert source[4] == {"value": 4}
    assert source[-1] == {"value": 6}
    with pytest.raises(IndexError):
        source[7]


def test_huggingface_resolver_forwards_pinned_identity(monkeypatch):
    calls = []
    source_dataset = datasets.Dataset.from_dict({"value": [4, 5]})

    def fake_load_dataset(path, **kwargs):
        calls.append((path, kwargs))
        return source_dataset

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    distribution = data.mix(
        data.source(
            "hf://organization/dataset",
            revision="0123456789abcdef",
            split="validation",
            subset="english",
            map=project_record,
        ),
        shuffle=False,
    )

    resolved = data.build_dataset(distribution)

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

    resolved = data.build_dataset(data.mix(artifact, shuffle=False))

    rows = [resolved[index] for index in range(len(resolved))]
    assert all(row is not None for row in rows)
    assert [row["value"] for row in rows if row is not None] == [10, 20, 30]


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
    distribution = data.mix(artifact, shuffle=False)
    mapper_path = artifact.mapper_id

    resolved = data.build_dataset(
        distribution,
        resolvers={"hf": cast(data.ArtifactResolver, lambda _artifact: [{"value": 7}])},
        mappers={mapper_path: lambda record: {"value": record["value"] + 1}},
    )

    assert resolved[0] == {"value": 8}


def test_local_resolver_rejects_unknown_formats(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("value\n1\n")
    artifact = data.source(path.as_uri(), map=project_record)

    with pytest.raises(ValueError, match="unsupported local artifact"):
        data.resolve_local(artifact)


def test_data_loader_fingerprints_the_resume_data_contract(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_local_artifact(path, "jsonl")
    distribution = data.mix(
        data.source(path.as_uri(), revision="v1", map=project_record),
        shuffle=False,
        seed=11,
    )

    first = data.build_data_loader(
        distribution,
        batch_size=2,
        batch_fn=collate_records,
        num_threads=0,
        prefetch_buffer_size=0,
    )
    same = data.build_data_loader(
        distribution,
        batch_size=2,
        batch_fn=collate_records,
        num_threads=0,
        prefetch_buffer_size=0,
    )
    different_batching = data.build_data_loader(
        distribution,
        batch_size=3,
        batch_fn=collate_records,
        num_threads=0,
        prefetch_buffer_size=0,
    )
    different_mapper = data.build_data_loader(
        distribution,
        batch_size=2,
        batch_fn=collate_records,
        num_threads=0,
        prefetch_buffer_size=0,
        mappers={distribution.sources[0].mapper: alternative_project_record},
    )

    assert first.data_contract["loader"] == "grain"
    assert first.data_contract["grain_version"] == grain.__version__
    source_contract = first.data_contract["source"]
    assert source_contract["kind"] == "configured-distribution"
    assert source_contract["distribution_fingerprint"] == distribution.fingerprint()
    assert source_contract["distribution"]["sources"][0]["mapper"].endswith(
        ".project_record"
    )
    implementations = source_contract["implementations"]
    assert implementations["batch_mapper"]["declared"].endswith(".collate_records")
    mapper_contract = implementations["mappers"][distribution.sources[0].mapper]
    assert mapper_contract["callable"].endswith(".project_record")
    assert implementations["resolvers"]["file"]["callable"].endswith(".resolve_local")
    assert first.data_fingerprint == same.data_fingerprint
    assert first.data_fingerprint != different_batching.data_fingerprint
    assert first.data_fingerprint != different_mapper.data_fingerprint


def test_data_loader_fingerprints_callable_configuration(tmp_path):
    path = tmp_path / "records.jsonl"
    _write_local_artifact(path, "jsonl")
    distribution = data.mix(
        data.source(path.as_uri(), revision="v1", map=project_record),
        shuffle=False,
    )

    first = data.build_data_loader(
        distribution,
        batch_size=2,
        batch_fn=StatefulCollator(width=8),
        num_threads=0,
        prefetch_buffer_size=0,
    )
    second = data.build_data_loader(
        distribution,
        batch_size=2,
        batch_fn=StatefulCollator(width=16),
        num_threads=0,
        prefetch_buffer_size=0,
    )

    assert first.data_fingerprint != second.data_fingerprint


@pytest.mark.parametrize("dataset_kind", ("map", "iter"))
def test_data_loader_accepts_native_grain_datasets_directly(dataset_kind):
    dataset = grain.MapDataset.source((1, 2, 3, 4))
    if dataset_kind == "iter":
        dataset = dataset.to_iter_dataset(
            grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
        )
    loader = data.build_data_loader(
        dataset,
        batch_size=2,
        batch_fn=collate_values,
        num_threads=0,
        prefetch_buffer_size=0,
        data_contract={"name": "numbers", "revision": "1"},
    )

    iterator = iter(loader)
    assert next(iterator) == (1, 2)
    assert loader.data_contract["source"]["kind"] == (f"grain-{dataset_kind}-dataset")
    assert callable(getattr(iterator, "get_state", None))
    assert callable(getattr(iterator, "set_state", None))


def test_direct_grain_dataset_requires_a_reproducibility_contract():
    dataset = grain.MapDataset.source((1, 2))

    with pytest.raises(ValueError, match="require data_contract"):
        data.build_data_loader(
            dataset,
            batch_size=2,
            batch_fn=collate_values,
            num_threads=0,
            prefetch_buffer_size=0,
        )
