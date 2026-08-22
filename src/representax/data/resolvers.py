"""Built-in random-access resolvers for immutable data artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
import threading
import zipfile
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


class RandomAccessSource(Protocol):
    """The source boundary required by Grain's ``MapDataset``."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class ArtifactSpec(Protocol):
    """Artifact fields consumed by built-in and custom resolvers."""

    @property
    def uri(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def split(self) -> str | None: ...

    @property
    def subset(self) -> str | None: ...


ArtifactResolver = Callable[[ArtifactSpec], RandomAccessSource]


class LazyArtifactSpec(Protocol):
    """The immutable byte identity carried through task samples."""

    @property
    def data(self) -> Any | None: ...

    @property
    def uri(self) -> str | None: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def etag(self) -> str | None: ...

    @property
    def archive_member(self) -> str | None: ...

    @property
    def byte_range(self) -> tuple[int, int] | None: ...

    @property
    def checksum(self) -> str | None: ...


ArtifactReader = Callable[[LazyArtifactSpec], bytes]


def _read_span(stream: Any, span: tuple[int, int] | None) -> bytes:
    if span is None:
        return stream.read()
    start, stop = span
    stream.seek(start)
    value = stream.read(stop - start)
    if len(value) != stop - start:
        raise EOFError(f"artifact byte range [{start}, {stop}) exceeds its payload")
    return value


def _read_archive_member(
    path: Path,
    member: str,
    span: tuple[int, int] | None,
) -> bytes:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
            return _read_span(stream, span)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            stream = archive.extractfile(member)
            if stream is None:
                raise FileNotFoundError(
                    f"archive member {member!r} is not a regular file in {path}"
                )
            with stream:
                return _read_span(stream, span)
    raise ValueError(f"artifact {path} is not a supported ZIP or TAR archive")


def read_local_artifact(artifact: LazyArtifactSpec) -> bytes:
    """Read one local artifact, archive member, or exact byte range lazily."""

    if artifact.uri is None:
        raise ValueError("referenced artifact requires a uri")
    path = local_path(artifact.uri)
    if not path.is_file():
        raise FileNotFoundError(f"local artifact does not exist: {path}")
    if artifact.archive_member is not None:
        return _read_archive_member(
            path,
            artifact.archive_member,
            artifact.byte_range,
        )
    with path.open("rb") as stream:
        return _read_span(stream, artifact.byte_range)


def read_http_artifact(artifact: LazyArtifactSpec) -> bytes:
    """Read one HTTP(S) object or server-honored byte range."""

    if artifact.uri is None:
        raise ValueError("referenced artifact requires a uri")
    if artifact.archive_member is not None:
        raise ValueError(
            "remote archive members require a scheme reader with an index; "
            "Representax will not download the complete archive implicitly"
        )
    headers = {}
    if artifact.etag is not None:
        headers["If-Match"] = artifact.etag
    if artifact.byte_range is not None:
        start, stop = artifact.byte_range
        headers["Range"] = f"bytes={start}-{stop - 1}"
    request = Request(artifact.uri, headers=headers)
    with urlopen(request) as response:  # noqa: S310 - explicit user data URI
        if artifact.byte_range is not None and response.status != 206:
            raise OSError(
                f"server ignored byte-range request for {artifact.uri!r}; "
                "refusing to download the complete object"
            )
        if artifact.byte_range is not None:
            start, stop = artifact.byte_range
            content_range = response.headers.get("Content-Range")
            expected_prefix = f"bytes {start}-{stop - 1}/"
            if content_range is None or not content_range.startswith(expected_prefix):
                raise OSError(
                    f"server returned unexpected Content-Range {content_range!r}; "
                    f"expected {expected_prefix!r}"
                )
        if artifact.etag is not None:
            observed_etag = response.headers.get("ETag")
            if observed_etag != artifact.etag:
                raise OSError(
                    f"artifact ETag changed: expected {artifact.etag!r}, "
                    f"received {observed_etag!r}"
                )
        value = response.read()
    if artifact.byte_range is not None:
        start, stop = artifact.byte_range
        if len(value) != stop - start:
            raise EOFError(
                f"HTTP artifact byte range [{start}, {stop}) returned "
                f"{len(value)} bytes"
            )
    return value


BUILTIN_ARTIFACT_READERS: Mapping[str, ArtifactReader] = {
    "file": read_local_artifact,
    "http": read_http_artifact,
    "https": read_http_artifact,
}


def read_artifact(
    artifact: LazyArtifactSpec,
    *,
    readers: Mapping[str, ArtifactReader] | None = None,
) -> bytes:
    """Resolve exactly one artifact payload and verify its declared checksum.

    The checksum is over the bytes returned after archive-member and byte-range
    selection. URI schemes remain extensible through ``readers``; unsupported
    schemes fail rather than silently materializing an intermediate copy.
    """

    if artifact.data is not None:
        if not isinstance(artifact.data, (bytes, bytearray, memoryview)):
            raise TypeError("inline artifact bytes must be bytes-like")
        value = bytes(artifact.data)
    else:
        if artifact.uri is None:
            raise ValueError("artifact has neither inline bytes nor a uri")
        registry = dict(BUILTIN_ARTIFACT_READERS)
        if readers is not None:
            registry.update(readers)
        scheme = urlparse(artifact.uri).scheme or "file"
        try:
            reader = registry[scheme]
        except KeyError as error:
            raise ValueError(
                f"no artifact reader registered for uri scheme {scheme!r}"
            ) from error
        value = reader(artifact)
        if not isinstance(value, bytes):
            raise TypeError("artifact readers must return bytes")
    if artifact.checksum is not None:
        algorithm, _, expected = artifact.checksum.partition(":")
        if algorithm != "sha256" or len(expected) != 64:
            raise ValueError("artifact checksum must use sha256:<64 hex digits>")
        observed = hashlib.sha256(value).hexdigest()
        if observed != expected.lower():
            raise OSError(
                f"artifact checksum mismatch: expected {expected.lower()}, "
                f"received {observed}"
            )
    return value


@dataclass(frozen=True)
class JsonLinesSource:
    """Offset-indexed local JSONL without a materialized dataset dependency."""

    path: Path
    offsets: tuple[int, ...]

    @classmethod
    def open(cls, path: Path) -> JsonLinesSource:
        offsets = []
        with path.open("rb") as stream:
            while line := stream.readline():
                if line.strip():
                    offsets.append(stream.tell() - len(line))
        return cls(path=path, offsets=tuple(offsets))

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return json.loads(stream.readline())


@dataclass
class ParquetSource:
    """Memory-mapped, row-group-cached Parquet without an Arrow cache rewrite."""

    path: Path
    row_group_offsets: tuple[int, ...]
    columns: tuple[str, ...]
    _file: Any = field(repr=False)
    _cache_size: int = field(default=2, repr=False)
    _cache: OrderedDict[int, Any] = field(
        default_factory=OrderedDict,
        init=False,
        repr=False,
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    @classmethod
    def open(cls, path: Path, *, cache_size: int = 2) -> ParquetSource:
        if cache_size <= 0:
            raise ValueError("Parquet row-group cache size must be positive")
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:  # pragma: no cover - optional HF extra
            raise ImportError(
                "local Parquet resolution requires `pip install representax[hf]`"
            ) from error
        parquet_file = parquet.ParquetFile(path, memory_map=True)
        offsets = []
        position = 0
        for index in range(parquet_file.num_row_groups):
            offsets.append(position)
            position += parquet_file.metadata.row_group(index).num_rows
        return cls(
            path=path,
            row_group_offsets=tuple(offsets),
            columns=tuple(parquet_file.schema_arrow.names),
            _file=parquet_file,
            _cache_size=cache_size,
        )

    def __len__(self) -> int:
        return self._file.metadata.num_rows

    def __getitem__(self, index: int) -> Any:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row_group = bisect_right(self.row_group_offsets, index) - 1
        offset = index - self.row_group_offsets[row_group]
        with self._lock:
            table = self._cache.get(row_group)
            if table is None:
                table = self._file.read_row_group(row_group)
                self._cache[row_group] = table
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(row_group)
        return {
            name: table.column(column)[offset].as_py()
            for column, name in enumerate(self.columns)
        }


def _datasets_module():
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "artifact resolution through Hugging Face Datasets requires the "
            "'hf' extra: "
            "pip install representax[hf]"
        ) from error
    return datasets


def _require_random_access(dataset: Any, *, source: str) -> RandomAccessSource:
    if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
        raise TypeError(
            f"resolved source {source!r} is not random-access; "
            "streaming artifacts require a future IterDataset resolver"
        )
    return dataset


def huggingface_dataset_id(uri: str) -> str:
    """Parse a configured ``hf://namespace/dataset`` source URI."""

    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"expected an hf:// URI, received {uri!r}")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("HF source URIs do not support query strings or fragments")
    dataset_id = "/".join(
        part for part in (parsed.netloc, parsed.path.strip("/")) if part
    )
    if not dataset_id:
        raise ValueError("HF source URI must identify a dataset")
    return unquote(dataset_id)


def resolve_huggingface(artifact: ArtifactSpec) -> RandomAccessSource:
    """Load one revision-pinned Hugging Face dataset split."""

    if not artifact.revision:
        raise ValueError("HF sources require an explicit revision")
    if not artifact.split:
        raise ValueError("HF sources require an explicit split")
    datasets = _datasets_module()
    resolved = datasets.load_dataset(
        huggingface_dataset_id(artifact.uri),
        name=artifact.subset,
        revision=artifact.revision,
        split=artifact.split,
        streaming=False,
        keep_in_memory=False,
    )
    return _require_random_access(resolved, source=artifact.uri)


_LOCAL_BUILDERS = {
    ".arrow": "arrow",
    ".jsonl": "json",
    ".ndjson": "json",
    ".parquet": "parquet",
}


def local_path(uri: str) -> Path:
    """Resolve a plain path or local ``file://`` URI without touching storage."""

    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"expected a local path or file:// URI, received {uri!r}")
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost"):
            raise ValueError("file:// sources must refer to the local host")
        value = unquote(parsed.path)
    else:
        value = uri
    if not value:
        raise ValueError("local source path must be non-empty")
    return Path(value).expanduser()


def resolve_local(artifact: ArtifactSpec) -> RandomAccessSource:
    """Resolve local JSONL, Parquet, Arrow, or HF dataset directories."""

    path = local_path(artifact.uri)
    if not path.exists():
        raise FileNotFoundError(f"local artifact does not exist: {path}")
    if path.is_dir():
        datasets = _datasets_module()
        split = artifact.split or "train"
        resolved = datasets.load_dataset(
            str(path),
            name=artifact.subset,
            split=split,
            streaming=False,
            keep_in_memory=False,
        )
        return _require_random_access(resolved, source=artifact.uri)

    try:
        builder = _LOCAL_BUILDERS[path.suffix.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(_LOCAL_BUILDERS))
        raise ValueError(
            f"unsupported local artifact extension {path.suffix!r}; "
            f"expected one of {supported}"
        ) from error

    if builder == "json":
        return JsonLinesSource.open(path)
    if builder == "parquet":
        return ParquetSource.open(path)
    datasets = _datasets_module()
    if builder == "arrow":
        resolved = datasets.Dataset.from_file(str(path), in_memory=False)
    else:
        split = artifact.split or "train"
        resolved = datasets.load_dataset(
            builder,
            data_files={split: str(path)},
            split=split,
            streaming=False,
            keep_in_memory=False,
        )
    return _require_random_access(resolved, source=artifact.uri)


BUILTIN_RESOLVERS: Mapping[str, ArtifactResolver] = {
    "file": resolve_local,
    "hf": resolve_huggingface,
}


__all__ = [
    "ArtifactReader",
    "ArtifactResolver",
    "ArtifactSpec",
    "BUILTIN_ARTIFACT_READERS",
    "BUILTIN_RESOLVERS",
    "JsonLinesSource",
    "LazyArtifactSpec",
    "ParquetSource",
    "RandomAccessSource",
    "huggingface_dataset_id",
    "local_path",
    "read_artifact",
    "read_http_artifact",
    "read_local_artifact",
    "resolve_huggingface",
    "resolve_local",
]
