"""Git-trackable recipes over immutable upstream artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

Mapper = str | Callable[[Any], Any]


def _mapper_id(mapper: Mapper) -> str:
    if isinstance(mapper, str):
        if not mapper:
            raise ValueError("mapper import path must be non-empty")
        return mapper
    module = getattr(mapper, "__module__", "")
    qualname = getattr(mapper, "__qualname__", "")
    if not module or not qualname or "<lambda>" in qualname:
        raise ValueError("recipe mappers must be named importable callables")
    return f"{module}.{qualname}"


@dataclass(frozen=True)
class ArtifactSource:
    """One upstream artifact and the code mapping its records into a task."""

    uri: str
    mapper: str
    revision: str | None = None
    split: str | None = None
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
    name: str | None = None,
) -> ArtifactSource:
    """Declare an upstream artifact without reading or copying it."""

    return ArtifactSource(
        uri=uri,
        mapper=_mapper_id(map),
        revision=revision,
        split=split,
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
    resolvers: Mapping[str, Callable[[ArtifactSource], Sequence[Any]]],
    mappers: Mapping[str, Callable[[Any], Any]] | None = None,
):
    """Resolve artifacts and compose the recipe as a lazy Grain MapDataset.

    Resolvers own transport-specific behavior for ``hf``, ``s3``, and local
    artifacts. Representax owns the task mapping and sampling policy. This
    avoids forcing user artifacts into a framework-specific intermediate form.
    """

    try:
        import grain
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Grain support requires the 'data' extra: pip install representax[data]"
        ) from error

    mapper_registry = {} if mappers is None else dict(mappers)
    datasets = []
    for index, artifact in enumerate(recipe.sources):
        try:
            resolver = resolvers[artifact.scheme]
        except KeyError as error:
            raise ValueError(
                f"no resolver registered for source scheme {artifact.scheme!r}"
            ) from error
        raw_source = resolver(artifact)
        try:
            mapper = mapper_registry[artifact.mapper]
        except KeyError as error:
            raise ValueError(f"no mapper registered for {artifact.mapper!r}") from error
        dataset = grain.MapDataset.source(raw_source).seed(recipe.seed + index)
        dataset = dataset.map(mapper)
        if recipe.shuffle:
            dataset = dataset.shuffle()
        datasets.append(dataset)
    if len(datasets) == 1:
        return datasets[0]
    return grain.MapDataset.mix(datasets, weights=recipe.normalized_weights)
