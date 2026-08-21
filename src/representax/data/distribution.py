"""Git-trackable data distributions over immutable upstream artifacts."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self
from urllib.parse import urlparse

import grain
from pydantic import model_validator

from representax._config import FrozenConfig
from representax.core import Modality

from .resolvers import BUILTIN_RESOLVERS, ArtifactResolver

Mapper = str | Callable[[Any], Any]


def identity(record: Any) -> Any:
    """Preserve an artifact record while retaining a stable mapper identity."""

    return record


@dataclass(frozen=True, slots=True)
class Artifact:
    """One raw input leaf in a mapped training sample.

    An artifact is either inline data or a lazy ``source``/``key`` reference.
    One source row may map to several artifacts, and task-specific sample
    dataclasses compose them naturally::

        RetrievalSample(
            query={"text": Artifact.text("find this")},
            document=Artifact.ref(
                Modality.IMAGE,
                source="images",
                key="00042.jpg",
            ),
        )

    The model-associated processor interprets the resulting artifact tree and
    emits its native fixed-shape array batch before the compiled step.
    """

    modality: Modality
    data: Any | None = None
    source: str | None = None
    key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "modality", Modality(self.modality))
        inline = self.data is not None
        referenced = self.source is not None or self.key is not None
        if inline == referenced:
            raise ValueError("an artifact must contain inline data or one reference")
        if referenced and (not self.source or not self.key):
            raise ValueError("referenced artifacts require both source and key")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def inline(
        cls,
        modality: Modality | str,
        data: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Construct an artifact whose raw value is already in the source row."""

        return cls(
            modality=Modality(modality),
            data=data,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def text(cls, value: str) -> Artifact:
        """Construct one inline text artifact."""

        if not isinstance(value, str):
            raise TypeError("text artifacts require a string")
        return cls.inline(Modality.TEXT, value)

    @classmethod
    def ref(
        cls,
        modality: Modality | str,
        *,
        source: str,
        key: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Artifact:
        """Construct a lazy reference into one named data source."""

        return cls(
            modality=Modality(modality),
            source=source,
            key=key,
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class DataLoader:
    """Thin iterable metadata wrapper around a native Grain ``IterDataset``.

    Representax does not define a dataset implementation. Configured sources
    resolve into Grain, transformations remain native Grain operations, and
    this wrapper only carries the batch-size and reproducibility contracts the
    trainer needs for validation and checkpoint resume.
    """

    dataset: grain.IterDataset[Any]
    global_batch_size: int | None
    data_contract: Mapping[str, Any]

    def __iter__(self):
        return iter(self.dataset)

    @property
    def data_fingerprint(self) -> str:
        return _json_fingerprint(self.data_contract)


def _json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapper_id(mapper: Mapper) -> str:
    if isinstance(mapper, str):
        if not mapper:
            raise ValueError("mapper import path must be non-empty")
        return mapper
    callable_value = mapper.func if isinstance(mapper, partial) else mapper
    if not inspect.isfunction(callable_value) and not inspect.isclass(callable_value):
        callable_value = type(callable_value)
    module = getattr(callable_value, "__module__", "")
    qualname = getattr(callable_value, "__qualname__", "")
    if (
        not module
        or module == "__main__"
        or not qualname
        or "<lambda>" in qualname
        or "<locals>" in qualname
    ):
        raise ValueError("data mappers must be named importable callables")
    identity = f"{module}.{qualname}"
    if isinstance(mapper, partial):
        bindings = {
            "args": mapper.args,
            "keywords": mapper.keywords or {},
        }
        identity += ":partial:" + _json_fingerprint(bindings)
    return identity


def _implementation_contract(function: Callable[..., Any]) -> dict[str, str]:
    """Identify callable code strongly enough to reject changed preprocessing."""

    callable_value = function.func if isinstance(function, partial) else function
    if not inspect.isfunction(callable_value) and not inspect.isclass(callable_value):
        callable_value = type(callable_value)
    module = getattr(callable_value, "__module__", "")
    qualname = getattr(callable_value, "__qualname__", "")
    if not module or not qualname:
        raise ValueError("data callables must expose a stable module and qualname")
    source_path = inspect.getsourcefile(callable_value)
    if source_path is not None:
        implementation = Path(source_path).read_bytes()
        digest_kind = "module_file_sha256"
    else:
        try:
            implementation = inspect.getsource(callable_value).encode()
        except (OSError, TypeError) as error:
            raise ValueError(
                f"cannot fingerprint data callable {module}.{qualname}"
            ) from error
        digest_kind = "callable_source_sha256"
    contract = {
        "callable": f"{module}.{qualname}",
        digest_kind: hashlib.sha256(implementation).hexdigest(),
    }
    if isinstance(function, partial):
        contract["bindings_sha256"] = _json_fingerprint(
            {"args": function.args, "keywords": function.keywords or {}}
        )
    state = getattr(function, "data_contract", None)
    if callable(state):
        contract["state_sha256"] = _json_fingerprint(state())
    return contract


def _data_implementations(
    distribution: DataDistributionConfig,
    *,
    batch_fn: Callable[[Sequence[Any]], Any] | None,
    resolvers: Mapping[str, ArtifactResolver] | None,
    mappers: Mapping[str, Callable[[Any], Any]] | None,
) -> dict[str, Any]:
    resolver_registry = dict(BUILTIN_RESOLVERS)
    if resolvers is not None:
        resolver_registry.update(resolvers)
    mapper_registry = {} if mappers is None else dict(mappers)
    resolver_contracts = {}
    mapper_contracts = {}
    for source_config in distribution.sources:
        try:
            resolver = resolver_registry[source_config.scheme]
        except KeyError as error:
            raise ValueError(
                f"no resolver registered for source scheme {source_config.scheme!r}"
            ) from error
        resolver_contracts[source_config.scheme] = _implementation_contract(resolver)
        mapper = mapper_registry.get(source_config.mapper)
        if mapper is None:
            mapper = load_mapper(source_config.mapper)
        mapper_contracts[source_config.mapper] = _implementation_contract(mapper)
    return {
        "resolvers": resolver_contracts,
        "mappers": mapper_contracts,
        "batch_mapper": _batch_implementation(batch_fn),
    }


def _batch_implementation(
    batch_fn: Callable[[Sequence[Any]], Any] | None,
) -> Mapping[str, Any] | None:
    if batch_fn is None:
        return None
    return {
        "declared": _mapper_id(batch_fn),
        "implementation": _implementation_contract(batch_fn),
    }


def load_mapper(path: str) -> Callable[[Any], Any]:
    """Import a data mapper from its stable dotted path."""

    parts = path.split(".")
    module = None
    boundary = 0
    for boundary in range(len(parts), 0, -1):
        try:
            module = importlib.import_module(".".join(parts[:boundary]))
        except ModuleNotFoundError as error:
            missing = error.name or ""
            candidate = ".".join(parts[:boundary])
            if missing != candidate and not candidate.startswith(f"{missing}."):
                raise
        else:
            break
    if module is None:
        raise ImportError(f"could not import mapper {path!r}")
    value: Any = module
    try:
        for attribute in parts[boundary:]:
            value = getattr(value, attribute)
    except AttributeError as error:
        raise ImportError(f"could not import mapper {path!r}") from error
    if not callable(value):
        raise TypeError(f"data mapper {path!r} is not callable")
    return value


class DataSourceConfig(FrozenConfig):
    """One immutable dataset source resolved directly into Grain records."""

    uri: str
    mapper: str
    revision: str | None = None
    split: str | None = None
    subset: str | None = None
    name: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if not self.uri:
            raise ValueError("source uri must be non-empty")
        _mapper_id(self.mapper)
        return self

    @property
    def scheme(self) -> str:
        return urlparse(self.uri).scheme or "file"

    @property
    def mapper_id(self) -> str:
        return self.mapper


class DataDistributionConfig(FrozenConfig):
    """A deterministic sampling policy over one or more data sources.

    The one-dataset case is a distribution with one source and implicit weight
    one. Mixtures and single sources are not separate concepts, so there is no
    second recipe abstraction.
    """

    sources: tuple[DataSourceConfig, ...]
    weights: tuple[float, ...]
    seed: int = 0
    shuffle: bool = True

    @model_validator(mode="before")
    @classmethod
    def default_weights(cls, value: object) -> object:
        """Make the ordinary one/equal-source distribution concise in config."""

        if isinstance(value, Mapping) and "weights" not in value:
            sources = value.get("sources")
            if isinstance(sources, Sequence):
                return {**value, "weights": [1.0] * len(sources)}
        return value

    @model_validator(mode="after")
    def validate_mixture(self) -> Self:
        if not self.sources:
            raise ValueError("a distribution must contain at least one source")
        if len(self.sources) != len(self.weights):
            raise ValueError("weights must match sources")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("sampling weights must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        names = [source.name for source in self.sources if source.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("named sources must be unique")
        return self

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)

    def fingerprint(self) -> str:
        """Hash distribution semantics without hashing the referenced contents."""

        return _json_fingerprint(self.model_dump(mode="json"))


def source(
    uri: str,
    *,
    map: Mapper,
    revision: str | None = None,
    split: str | None = None,
    subset: str | None = None,
    name: str | None = None,
) -> DataSourceConfig:
    """Declare one upstream data source without reading or copying it."""

    return DataSourceConfig(
        uri=uri,
        mapper=_mapper_id(map),
        revision=revision,
        split=split,
        subset=subset,
        name=name,
    )


def mix(
    *sources: DataSourceConfig | Mapping[str, Any],
    weights: Sequence[float] | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> DataDistributionConfig:
    """Declare a sampling policy; one source is the ordinary dataset case."""

    resolved_sources = tuple(
        item
        if isinstance(item, DataSourceConfig)
        else DataSourceConfig.model_validate(item)
        for item in sources
    )
    resolved_weights = (
        tuple(1.0 for _ in resolved_sources) if weights is None else tuple(weights)
    )
    return DataDistributionConfig(
        sources=resolved_sources,
        weights=resolved_weights,
        seed=seed,
        shuffle=shuffle,
    )


def build_dataset(
    distribution: DataDistributionConfig,
    *,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
) -> grain.MapDataset[Any]:
    """Resolve a configured distribution as a lazy native Grain MapDataset.

    Representax resolves ``hf://`` and local sources by default. Additional
    schemes can be registered without changing distribution or task semantics.
    """

    resolver_registry = dict(BUILTIN_RESOLVERS)
    if resolvers is not None:
        resolver_registry.update(resolvers)
    mapper_registry = {} if mappers is None else dict(mappers)
    datasets = []
    for index, source_config in enumerate(distribution.sources):
        try:
            resolver = resolver_registry[source_config.scheme]
        except KeyError as error:
            raise ValueError(
                f"no resolver registered for source scheme {source_config.scheme!r}"
            ) from error
        raw_source = resolver(source_config)
        mapper = mapper_registry.get(source_config.mapper)
        if mapper is None:
            mapper = load_mapper(source_config.mapper)
        dataset = grain.MapDataset.source(raw_source).seed(distribution.seed + index)
        dataset = dataset.map(mapper)
        if distribution.shuffle:
            dataset = dataset.shuffle()
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return grain.MapDataset.mix(datasets, weights=distribution.normalized_weights)


def build_data_loader(
    distribution: (
        DataDistributionConfig | grain.MapDataset[Any] | grain.IterDataset[Any]
    ),
    *,
    batch_size: int,
    batch_fn: Callable[[Sequence[Any]], Any] | None = None,
    drop_remainder: bool = True,
    num_threads: int = 16,
    prefetch_buffer_size: int = 2,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
    data_contract: Mapping[str, Any] | None = None,
) -> DataLoader:
    """Build a native Grain pipeline yielding static, model-ready batches.

    ``distribution`` is normally a serializable :class:`DataDistributionConfig`.
    Advanced Python callers may instead supply an existing Grain ``MapDataset``
    or ``IterDataset``. Direct datasets remain Grain objects rather than being
    copied into a Representax dataset class, but must provide ``data_contract``
    so checkpoint resume can identify their semantics.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if prefetch_buffer_size < 0:
        raise ValueError("prefetch_buffer_size must be non-negative")
    if isinstance(distribution, DataDistributionConfig):
        dataset = build_dataset(
            distribution,
            resolvers=resolvers,
            mappers=mappers,
        ).batch(
            batch_size,
            drop_remainder=drop_remainder,
            batch_fn=batch_fn,
        )
        iterator = dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        source_contract: Mapping[str, Any] = {
            "kind": "configured-distribution",
            "distribution": distribution.model_dump(mode="json"),
            "distribution_fingerprint": distribution.fingerprint(),
            "implementations": _data_implementations(
                distribution,
                batch_fn=batch_fn,
                resolvers=resolvers,
                mappers=mappers,
            ),
        }
    elif isinstance(distribution, grain.MapDataset):
        if data_contract is None:
            raise ValueError("direct Grain datasets require data_contract")
        dataset = distribution.batch(
            batch_size,
            drop_remainder=drop_remainder,
            batch_fn=batch_fn,
        )
        iterator = dataset.to_iter_dataset(
            grain.ReadOptions(
                num_threads=num_threads,
                prefetch_buffer_size=prefetch_buffer_size,
            )
        )
        source_contract = {
            "kind": "grain-map-dataset",
            **dict(data_contract),
            "batch_mapper": _batch_implementation(batch_fn),
        }
    elif isinstance(distribution, grain.IterDataset):
        if data_contract is None:
            raise ValueError("direct Grain datasets require data_contract")
        iterator = distribution.batch(
            batch_size,
            drop_remainder=drop_remainder,
            batch_fn=batch_fn,
        )
        source_contract = {
            "kind": "grain-iter-dataset",
            **dict(data_contract),
            "batch_mapper": _batch_implementation(batch_fn),
        }
    else:
        raise TypeError(
            "distribution must be DataDistributionConfig or a Grain dataset"
        )
    return DataLoader(
        dataset=iterator,
        global_batch_size=batch_size if drop_remainder else None,
        data_contract={
            "schema_version": "representax-data-loader-v2",
            "loader": "grain",
            "grain_version": grain.__version__,
            "source": source_contract,
            "batch_size": batch_size,
            "drop_remainder": drop_remainder,
        },
    )
