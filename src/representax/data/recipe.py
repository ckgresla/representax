"""Git-trackable recipes over immutable upstream artifacts."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from .resolvers import BUILTIN_RESOLVERS, ArtifactResolver

Mapper = str | Callable[[Any], Any]


@dataclass(frozen=True)
class GrainBatchSource:
    """Iterable Grain batches with an optional exact global-size contract."""

    dataset: Any
    global_batch_size: int | None

    def __iter__(self):
        return iter(self.dataset)


def _mapper_id(mapper: Mapper) -> str:
    if isinstance(mapper, str):
        if not mapper:
            raise ValueError("mapper import path must be non-empty")
        return mapper
    module = getattr(mapper, "__module__", "")
    qualname = getattr(mapper, "__qualname__", "")
    if not module or not qualname or "<lambda>" in qualname or "<locals>" in qualname:
        raise ValueError("recipe mappers must be named importable callables")
    return f"{module}.{qualname}"


def load_mapper(path: str) -> Callable[[Any], Any]:
    """Import a recipe mapper from its stable dotted path."""

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
        raise TypeError(f"recipe mapper {path!r} is not callable")
    return value


@dataclass(frozen=True)
class ArtifactSource:
    """One upstream artifact and the code mapping its records into a task."""

    uri: str
    mapper: str
    revision: str | None = None
    split: str | None = None
    subset: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.uri:
            raise ValueError("source uri must be non-empty")
        _mapper_id(self.mapper)

    @property
    def scheme(self) -> str:
        return urlparse(self.uri).scheme or "file"

    @property
    def mapper_id(self) -> str:
        return self.mapper


@dataclass(frozen=True)
class MixtureRecipe:
    """A deterministic sampling policy over one or more artifact sources."""

    sources: tuple[ArtifactSource, ...]
    weights: tuple[float, ...]
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("a recipe must contain at least one source")
        if len(self.sources) != len(self.weights):
            raise ValueError("weights must match sources")
        if any(not math.isfinite(weight) or weight <= 0 for weight in self.weights):
            raise ValueError("sampling weights must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        names = [source.name for source in self.sources if source.name is not None]
        if len(names) != len(set(names)):
            raise ValueError("named sources must be unique")

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)

    def fingerprint(self) -> str:
        """Hash the reproducibility-relevant recipe, excluding data contents."""

        payload = {
            "sources": [
                {
                    **asdict(source),
                    "mapper": source.mapper_id,
                }
                for source in self.sources
            ],
            "weights": self.weights,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def source(
    uri: str,
    *,
    map: Mapper,
    revision: str | None = None,
    split: str | None = None,
    subset: str | None = None,
    name: str | None = None,
) -> ArtifactSource:
    """Declare an upstream artifact without reading or copying it."""

    return ArtifactSource(
        uri=uri,
        mapper=_mapper_id(map),
        revision=revision,
        split=split,
        subset=subset,
        name=name,
    )


def mix(
    *sources: ArtifactSource | Mapping[str, Any],
    weights: Sequence[float] | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> MixtureRecipe:
    """Declare a sampling policy; one source is the ordinary dataset case."""

    resolved_sources = tuple(
        item if isinstance(item, ArtifactSource) else ArtifactSource(**dict(item))
        for item in sources
    )
    resolved_weights = (
        tuple(1.0 for _ in resolved_sources) if weights is None else tuple(weights)
    )
    return MixtureRecipe(
        sources=resolved_sources,
        weights=resolved_weights,
        seed=seed,
        shuffle=shuffle,
    )


def build_grain_dataset(
    recipe: MixtureRecipe,
    *,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
):
    """Resolve artifacts and compose the recipe as a lazy Grain MapDataset.

    Representax resolves ``hf://`` and local artifacts by default. Additional
    schemes can be registered without changing recipe or task semantics.
    """

    try:
        import grain
    except ImportError as error:  # pragma: no cover - broken installation
        raise ImportError(
            "Grain is required for Representax training; reinstall representax"
        ) from error

    resolver_registry = dict(BUILTIN_RESOLVERS)
    if resolvers is not None:
        resolver_registry.update(resolvers)
    mapper_registry = {} if mappers is None else dict(mappers)
    datasets = []
    for index, artifact in enumerate(recipe.sources):
        try:
            resolver = resolver_registry[artifact.scheme]
        except KeyError as error:
            raise ValueError(
                f"no resolver registered for source scheme {artifact.scheme!r}"
            ) from error
        raw_source = resolver(artifact)
        mapper = mapper_registry.get(artifact.mapper)
        if mapper is None:
            mapper = load_mapper(artifact.mapper)
        dataset = grain.MapDataset.source(raw_source).seed(recipe.seed + index)
        dataset = dataset.map(mapper)
        if recipe.shuffle:
            dataset = dataset.shuffle()
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return grain.MapDataset.mix(datasets, weights=recipe.normalized_weights)


def build_grain_iterator(
    recipe: MixtureRecipe,
    *,
    batch_size: int,
    batch_fn: Callable[[Sequence[Any]], Any] | None = None,
    drop_remainder: bool = True,
    num_threads: int = 16,
    prefetch_buffer_size: int = 2,
    resolvers: Mapping[str, ArtifactResolver] | None = None,
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
):
    """Build a prefetched Grain iterator of static, model-ready batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_threads < 0:
        raise ValueError("num_threads must be non-negative")
    if prefetch_buffer_size < 0:
        raise ValueError("prefetch_buffer_size must be non-negative")
    try:
        import grain
    except ImportError as error:  # pragma: no cover - broken installation
        raise ImportError(
            "Grain is required for Representax training; reinstall representax"
        ) from error
    dataset = build_grain_dataset(
        recipe,
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
    return GrainBatchSource(
        dataset=iterator,
        global_batch_size=batch_size if drop_remainder else None,
    )
