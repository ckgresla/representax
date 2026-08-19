"""Built-in random-access resolvers for immutable data artifacts."""

from __future__ import annotations

import json
import threading
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse


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
    """Parse Representax's ``hf://namespace/dataset`` recipe URI."""

    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"expected an hf:// URI, received {uri!r}")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("HF recipe URIs do not support query strings or fragments")
    dataset_id = "/".join(
        part for part in (parsed.netloc, parsed.path.strip("/")) if part
    )
    if not dataset_id:
        raise ValueError("HF recipe URI must identify a dataset")
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
    "ArtifactResolver",
    "ArtifactSpec",
    "BUILTIN_RESOLVERS",
    "JsonLinesSource",
    "ParquetSource",
    "RandomAccessSource",
    "huggingface_dataset_id",
    "local_path",
    "resolve_huggingface",
    "resolve_local",
]
