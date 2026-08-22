"""Built-in Hugging Face and local artifact resolver tests."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from typing import cast

import numpy as np
import pytest

from representax import data
from representax.data import resolvers as data_resolvers

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


def collate_numpy_values(values):
    return np.asarray(values, dtype=np.float32)


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


def test_lazy_local_artifact_reads_only_selected_verified_bytes(tmp_path):
    path = tmp_path / "audio.pcm"
    path.write_bytes(b"header:0123456789:footer")
    selected = b"23456"
    artifact = data.Artifact.ref(
        "audio",
        uri=path.as_uri(),
        revision="manifest-v3",
        byte_range=(9, 14),
        checksum="sha256:" + hashlib.sha256(selected).hexdigest(),
        metadata={"sample_rate": 16_000, "samples": 5},
    )

    assert artifact.read_bytes() == selected
    assert artifact.revision == "manifest-v3"
    assert artifact.metadata["samples"] == 5


def test_lazy_artifact_checksums_fail_closed(tmp_path):
    path = tmp_path / "image.raw"
    path.write_bytes(b"pixels")
    artifact = data.Artifact.ref(
        "image",
        uri=path.as_uri(),
        checksum="sha256:" + "0" * 64,
    )

    with pytest.raises(OSError, match="checksum mismatch"):
        artifact.read_bytes()
    with pytest.raises(ValueError, match="sha256"):
        data.Artifact.ref(
            "image",
            uri=path.as_uri(),
            checksum="md5:invalid",
        )


@pytest.mark.parametrize("archive_kind", ("zip", "tar"))
def test_lazy_artifact_reads_one_archive_member_range(tmp_path, archive_kind):
    path = tmp_path / f"shard.{archive_kind}"
    payload = b"0123456789"
    if archive_kind == "zip":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("clips/0001.raw", payload)
    else:
        member = tarfile.TarInfo("clips/0001.raw")
        member.size = len(payload)
        with tarfile.open(path, "w") as archive:
            archive.addfile(member, io.BytesIO(payload))
    artifact = data.Artifact.ref(
        "video",
        uri=path.as_uri(),
        archive_member="clips/0001.raw",
        byte_range=(3, 7),
    )

    assert data.read_artifact(artifact) == b"3456"


def test_lazy_artifact_reader_registry_is_scheme_extensible():
    observed = []

    def read_object(artifact):
        observed.append(
            (
                artifact.uri,
                artifact.revision,
                artifact.etag,
                artifact.byte_range,
            )
        )
        return b"selected"

    artifact = data.Artifact.ref(
        "image",
        uri="s3://bucket/images/42.jpg",
        revision="version-7",
        etag='"object-etag"',
        byte_range=(1024, 2048),
    )

    assert data.read_artifact(artifact, readers={"s3": read_object}) == b"selected"
    assert observed == [
        (
            "s3://bucket/images/42.jpg",
            "version-7",
            '"object-etag"',
            (1024, 2048),
        )
    ]


def test_http_artifact_requires_range_and_etag_to_be_honored(monkeypatch):
    requests = []

    class Response:
        status = 206
        headers = {
            "ETag": '"immutable"',
            "Content-Range": "bytes 2-5/32",
        }

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b"cdef"

    def open_request(request):
        requests.append(request)
        return Response()

    monkeypatch.setattr(data_resolvers, "urlopen", open_request)
    artifact = data.Artifact.ref(
        "audio",
        uri="https://example.test/audio.raw",
        etag='"immutable"',
        byte_range=(2, 6),
    )

    assert artifact.read_bytes() == b"cdef"
    headers = {name.lower(): value for name, value in requests[0].header_items()}
    assert headers == {"if-match": '"immutable"', "range": "bytes=2-5"}


def test_http_artifact_refuses_server_that_ignores_range(monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            pytest.fail("ignored range must fail before reading the response body")

    monkeypatch.setattr(data_resolvers, "urlopen", lambda _request: Response())
    artifact = data.Artifact.ref(
        "video",
        uri="https://example.test/video.mp4",
        byte_range=(0, 1024),
    )

    with pytest.raises(OSError, match="ignored byte-range"):
        artifact.read_bytes()


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


def test_data_loader_bounds_model_ready_host_memory_and_reports_prefetch():
    dataset = grain.MapDataset.range(16)
    loader = data.build_data_loader(
        dataset,
        batch_size=4,
        batch_fn=collate_numpy_values,
        num_threads=2,
        prefetch_buffer_size=2,
        host_memory_budget_bytes=48,
        data_contract={"name": "bounded-values", "revision": "1"},
    )
    iterator = iter(loader)

    batch = next(iterator)

    assert batch.shape == (4,)
    assert iterator.last_telemetry is not None
    assert iterator.last_telemetry.host_batch_bytes == 16
    assert iterator.last_telemetry.preprocess_seconds is not None
    assert iterator.last_telemetry.prefetch_capacity == 2
    assert iterator.last_telemetry.prefetch_ready_batches == 0
    iterator.close()

    too_small = data.build_data_loader(
        grain.MapDataset.range(16),
        batch_size=4,
        batch_fn=collate_numpy_values,
        num_threads=2,
        prefetch_buffer_size=2,
        host_memory_budget_bytes=47,
        data_contract={"name": "bounded-values", "revision": "1"},
    )
    with pytest.raises(MemoryError, match="per-slot limit"):
        next(iter(too_small))


def test_grain_cursor_resumes_identically_across_worker_and_prefetch_settings():
    def dataset():
        return grain.MapDataset.range(40).seed(31).shuffle()

    contract = {"name": "resume-values", "revision": "1"}
    serial = data.build_data_loader(
        dataset(),
        batch_size=4,
        batch_fn=collate_values,
        num_threads=0,
        prefetch_buffer_size=0,
        data_contract=contract,
    )
    parallel = data.build_data_loader(
        dataset(),
        batch_size=4,
        batch_fn=collate_values,
        num_threads=4,
        prefetch_buffer_size=4,
        data_contract=contract,
    )
    reference = data.build_data_loader(
        dataset(),
        batch_size=4,
        batch_fn=collate_values,
        num_threads=2,
        prefetch_buffer_size=2,
        data_contract=contract,
    )

    assert serial.data_fingerprint == parallel.data_fingerprint
    serial_iterator = iter(serial)
    prefix = [next(serial_iterator), next(serial_iterator), next(serial_iterator)]
    state = serial_iterator.get_state()
    serial_iterator.close()
    parallel_iterator = iter(parallel)
    parallel_iterator.set_state(state)
    resumed = list(parallel_iterator)
    uninterrupted = list(reference)

    assert [*prefix, *resumed] == uninterrupted
