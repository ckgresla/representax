"""Built-in random-access resolvers for immutable data artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse


class RandomAccessSource(Protocol):
    """The source boundary required by Grain's ``MapDataset``."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class ArtifactSpec(Protocol):
    """Artifact fields consumed by built-in and custom resolvers."""

    uri: str
    revision: str | None
    split: str | None
    subset: str | None


ArtifactResolver = Callable[[ArtifactSpec], RandomAccessSource]


def _datasets_module():
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "artifact resolution requires the 'data' extra: "
            "pip install representax[data]"
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
    datasets = _datasets_module()
    if path.is_dir():
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
    "RandomAccessSource",
    "huggingface_dataset_id",
    "local_path",
    "resolve_huggingface",
    "resolve_local",
]
