"""Generic host preprocessing for native model implementations."""

from __future__ import annotations

import hashlib
import inspect
import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax.numpy as jnp
import numpy as np

from representax.core import Modality, Route
from representax.data import Artifact


def _shape(value: Sequence[int], *, name: str) -> tuple[int, ...]:
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool):
            raise ValueError(f"{name} must contain positive integer dimensions")
        try:
            normalized = operator.index(dimension)
        except TypeError as error:
            raise ValueError(
                f"{name} must contain positive integer dimensions"
            ) from error
        if normalized <= 0:
            raise ValueError(f"{name} must contain positive integer dimensions")
        dimensions.append(normalized)
    if not dimensions:
        raise ValueError(f"{name} must contain positive integer dimensions")
    return tuple(dimensions)


def select_static_shape_bucket(
    required_shape: Sequence[int],
    admitted_shapes: Sequence[Sequence[int]],
) -> tuple[int, ...]:
    """Choose the unique smallest admitted shape containing ``required_shape``.

    Ordering is dimension-wise. Incomparable candidates are rejected because
    trading, for example, frame count against image resolution is a model-owned
    scientific policy rather than a generic memory heuristic.
    """

    required = _shape(required_shape, name="required_shape")
    raw_buckets = tuple(admitted_shapes)
    if not raw_buckets:
        raise ValueError("at least one static shape bucket is required")
    buckets = tuple(
        dict.fromkeys(
            _shape(shape, name="static shape buckets") for shape in raw_buckets
        )
    )
    if any(len(bucket) != len(required) for bucket in buckets):
        raise ValueError("static shape buckets must match required_shape rank")
    candidates = tuple(
        bucket
        for bucket in buckets
        if all(
            needed <= admitted
            for needed, admitted in zip(required, bucket, strict=True)
        )
    )
    if not candidates:
        raise ValueError(
            f"required shape {required!r} exceeds admitted buckets {buckets!r}"
        )
    minima = tuple(
        candidate
        for candidate in candidates
        if not any(
            other != candidate
            and all(left <= right for left, right in zip(other, candidate, strict=True))
            for other in candidates
        )
    )
    if len(minima) != 1:
        raise ValueError(
            f"required shape {required!r} has incomparable admitted buckets "
            f"{minima!r}; the model processor must choose explicitly"
        )
    return minima[0]


Process = Callable[..., Any]
MediaProbe = Callable[..., Sequence[int]]
MediaPrepare = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class Processor:
    """One generic host processor associated with a native model artifact.

    Model loaders construct this object with their tokenizer/media operations
    and finite shape policy. It contains no model parameters and remains outside
    every Equinox PyTree and compiled JAX program.
    """

    process: Process = field(repr=False)
    contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not callable(self.process):
            raise TypeError("processor process must be callable")
        contract = dict(self.contract)
        try:
            json.dumps(contract, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise TypeError("processor contract must be JSON serializable") from error
        object.__setattr__(self, "contract", MappingProxyType(contract))

    def data_contract(self) -> Mapping[str, Any]:
        """Return the serializable processing and finite-shape contract."""

        return dict(self.contract)

    def __call__(
        self,
        artifacts: Sequence[Any],
        *,
        route: Route = Route.GENERIC,
        seed: int | None = None,
        **options: Any,
    ) -> Any:
        """Convert raw artifacts into one model-native fixed-shape batch."""

        return self.process(
            artifacts,
            route=Route(route),
            seed=seed,
            **options,
        )


def _callable_name(value: Callable[..., Any]) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _callable_contract(value: Callable[..., Any]) -> Mapping[str, str]:
    identity = _callable_name(value)
    callable_value = (
        value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
    )
    source_path = inspect.getsourcefile(callable_value)
    if source_path is not None:
        source = Path(source_path).read_bytes()
        digest_name = "module_file_sha256"
    else:
        try:
            source = inspect.getsource(callable_value).encode()
        except (OSError, TypeError):
            return {"callable": identity}
        digest_name = "callable_source_sha256"
    return {
        "callable": identity,
        digest_name: hashlib.sha256(source).hexdigest(),
    }


def _media_artifact(value: Any, *, modality: Modality) -> Artifact:
    if isinstance(value, Artifact):
        if value.modality != modality:
            raise TypeError(
                f"{modality} processors cannot consume {value.modality} artifacts"
            )
        return value
    return Artifact.inline(modality, value)


def make_media_processor(
    *,
    modality: Modality | str,
    admitted_shapes: Sequence[Sequence[int]],
    probe: MediaProbe,
    prepare: MediaPrepare,
    batch_builder: Callable[..., Any],
    configuration: Mapping[str, Any],
) -> Processor:
    """Build one model-associated finite-shape media processor.

    ``probe`` inspects artifact metadata and returns the model-required shape
    without decoding media. One dimension-wise bucket is selected for the
    complete batch. ``prepare`` then resolves, decodes, selects, transforms,
    and pads one artifact into arrays for that exact bucket. ``batch_builder``
    receives the stacked JAX arrays and constructs the model-native batch.

    This is deliberately callable-based rather than an adapter hierarchy:
    image tiling, audio windows, video frame sampling, normalization, and
    special model fields remain owned by the model integration.
    """

    resolved_modality = Modality(modality)
    if not callable(probe) or not callable(prepare) or not callable(batch_builder):
        raise TypeError("media probe, prepare, and batch_builder must be callable")
    media_configuration = dict(configuration)
    try:
        json.dumps(media_configuration, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "media processor configuration must be JSON serializable"
        ) from error
    raw_shapes = tuple(admitted_shapes)
    if not raw_shapes:
        raise ValueError("at least one admitted media shape is required")
    shapes = tuple(
        dict.fromkeys(
            _shape(shape, name="admitted media shapes") for shape in raw_shapes
        )
    )
    rank = len(shapes[0])
    if any(len(shape) != rank for shape in shapes):
        raise ValueError("admitted media shapes must have one consistent rank")

    def process(
        artifacts: Sequence[Any],
        *,
        route: Route,
        seed: int | None,
    ) -> Any:
        if not artifacts:
            raise ValueError("media processor batches must be non-empty")
        values = tuple(
            _media_artifact(artifact, modality=resolved_modality)
            for artifact in artifacts
        )
        requirements = tuple(
            _shape(
                probe(artifact, route=route),
                name=f"{resolved_modality} shape requirement",
            )
            for artifact in values
        )
        if any(len(requirement) != rank for requirement in requirements):
            raise ValueError(
                f"{resolved_modality} shape requirements must have rank {rank}"
            )
        required = tuple(max(values) for values in zip(*requirements, strict=True))
        bucket = select_static_shape_bucket(required, shapes)
        seed_sequence = None if seed is None else np.random.SeedSequence(seed)
        child_seeds = (
            (None,) * len(values)
            if seed_sequence is None
            else seed_sequence.spawn(len(values))
        )
        rows = []
        for artifact, child_seed in zip(values, child_seeds, strict=True):
            rng = None if child_seed is None else np.random.default_rng(child_seed)
            prepared = prepare(
                artifact,
                bucket=bucket,
                route=route,
                rng=rng,
            )
            if not isinstance(prepared, Mapping) or not prepared:
                raise TypeError("media prepare must return a non-empty mapping")
            rows.append({name: np.asarray(value) for name, value in prepared.items()})
        fields = tuple(rows[0])
        if any(tuple(row) != fields for row in rows[1:]):
            raise ValueError("media prepare fields must be identical for every row")
        arrays = {}
        for name in fields:
            field_shapes = tuple(row[name].shape for row in rows)
            if len(set(field_shapes)) != 1:
                raise ValueError(
                    f"prepared media field {name!r} has unstable shapes "
                    f"{field_shapes!r} for bucket {bucket!r}"
                )
            arrays[name] = jnp.asarray(np.stack([row[name] for row in rows]))
        return batch_builder(**arrays)

    return Processor(
        process=process,
        contract={
            "schema_version": "representax-media-processor-v1",
            "modality": resolved_modality.value,
            "admitted_shapes": [list(shape) for shape in shapes],
            "probe": _callable_contract(probe),
            "prepare": _callable_contract(prepare),
            "batch_builder": _callable_contract(batch_builder),
            "configuration": media_configuration,
        },
    )


def make_image_processor(
    *,
    admitted_shapes: Sequence[Sequence[int]],
    probe: MediaProbe,
    prepare: MediaPrepare,
    batch_builder: Callable[..., Any],
    configuration: Mapping[str, Any],
) -> Processor:
    """Build an image processor whose shapes conventionally mean ``(H, W)``."""

    if any(len(tuple(shape)) != 2 for shape in admitted_shapes):
        raise ValueError("image admitted shapes must be (height, width)")
    return make_media_processor(
        modality=Modality.IMAGE,
        admitted_shapes=admitted_shapes,
        probe=probe,
        prepare=prepare,
        batch_builder=batch_builder,
        configuration=configuration,
    )


def make_audio_processor(
    *,
    admitted_shapes: Sequence[Sequence[int]],
    probe: MediaProbe,
    prepare: MediaPrepare,
    batch_builder: Callable[..., Any],
    configuration: Mapping[str, Any],
) -> Processor:
    """Build an audio processor whose shapes conventionally mean ``(samples,)``."""

    if any(len(tuple(shape)) != 1 for shape in admitted_shapes):
        raise ValueError("audio admitted shapes must be (samples,)")
    return make_media_processor(
        modality=Modality.AUDIO,
        admitted_shapes=admitted_shapes,
        probe=probe,
        prepare=prepare,
        batch_builder=batch_builder,
        configuration=configuration,
    )


def make_video_processor(
    *,
    admitted_shapes: Sequence[Sequence[int]],
    probe: MediaProbe,
    prepare: MediaPrepare,
    batch_builder: Callable[..., Any],
    configuration: Mapping[str, Any],
) -> Processor:
    """Build a video processor with ``(frames, H, W)`` shape convention."""

    if any(len(tuple(shape)) != 3 for shape in admitted_shapes):
        raise ValueError("video admitted shapes must be (frames, height, width)")
    return make_media_processor(
        modality=Modality.VIDEO,
        admitted_shapes=admitted_shapes,
        probe=probe,
        prepare=prepare,
        batch_builder=batch_builder,
        configuration=configuration,
    )


def make_text_processor(
    *,
    tokenizer: Any,
    batch_builder: Callable[..., Any],
    max_sequence_length: int | None = None,
    sequence_length_buckets: Sequence[int] | None = None,
    prompts: Mapping[str, str] | None = None,
    default_prompt_name: str | None = None,
    include_prompt: bool = True,
    pooling_batch_builder: Callable[[Any, Any], Any] | None = None,
) -> Processor:
    """Construct the shared finite-shape text processor used by native models."""

    if sequence_length_buckets is None:
        if max_sequence_length is None:
            raise ValueError(
                "max_sequence_length or sequence_length_buckets is required"
            )
        sequence_length_buckets = (max_sequence_length,)
    lengths = tuple(sequence_length_buckets)
    if not lengths or any(
        not isinstance(length, int) or isinstance(length, bool) or length <= 0
        for length in lengths
    ):
        raise ValueError("sequence_length_buckets must be positive integers")
    lengths = tuple(sorted(set(lengths)))
    if max_sequence_length is not None and lengths[-1] > max_sequence_length:
        raise ValueError("sequence length buckets cannot exceed max_sequence_length")
    prompt_values = {} if prompts is None else dict(prompts)
    if default_prompt_name is not None and default_prompt_name not in prompt_values:
        raise ValueError("default_prompt_name does not name a configured prompt")
    if not include_prompt and pooling_batch_builder is None:
        raise ValueError("prompt exclusion requires pooling_batch_builder")
    maximum = lengths[-1]

    def route_prompt(
        route: Route,
        prompt_name: str | None,
        prompt: str | None,
    ) -> str:
        if prompt_name is not None and prompt is not None:
            raise ValueError("specify prompt_name or prompt, not both")
        if prompt is not None:
            return prompt
        if prompt_name is not None:
            try:
                return prompt_values[prompt_name]
            except KeyError as error:
                raise KeyError(f"unknown prompt {prompt_name!r}") from error
        if route is Route.QUERY:
            candidates = ("query",)
        elif route is Route.DOCUMENT:
            candidates = ("document", "passage", "corpus")
        elif default_prompt_name is not None:
            candidates = (default_prompt_name,)
        else:
            candidates = ()
        return next(
            (prompt_values[name] for name in candidates if name in prompt_values),
            "",
        )

    def tokenize(texts: Sequence[str]) -> Mapping[str, np.ndarray]:
        if not texts:
            raise ValueError("processor batches must be non-empty")
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=maximum,
            return_tensors="np",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("text tokenizer must return a mapping")
        arrays = {name: np.asarray(value) for name, value in encoded.items()}
        try:
            input_ids = arrays["input_ids"]
        except KeyError as error:
            raise KeyError("tokenizer output is missing 'input_ids'") from error
        if input_ids.ndim != 2 or input_ids.shape[0] != len(texts):
            raise ValueError("tokenizer input_ids must have shape [batch, sequence]")
        bucket_length = select_static_shape_bucket(
            (input_ids.shape[1],),
            tuple((length,) for length in lengths),
        )[0]
        pad_token_id = getattr(
            tokenizer,
            "pad_token_id",
            getattr(tokenizer, "pad_id", 0),
        )
        if pad_token_id is None:
            raise ValueError("text tokenizer must define a pad token")
        padded = {}
        for name, value in arrays.items():
            if value.ndim != 2 or value.shape[0] != len(texts):
                padded[name] = value
                continue
            if value.shape[1] != input_ids.shape[1]:
                raise ValueError(
                    f"tokenizer field {name!r} has a different sequence length"
                )
            fill = int(pad_token_id) if name == "input_ids" else 0
            padded[name] = np.pad(
                value,
                ((0, 0), (0, bucket_length - value.shape[1])),
                mode="constant",
                constant_values=fill,
            )
        return padded

    def prompt_length(prefix: str) -> int:
        if not prefix:
            return 0
        encoded = tokenizer(
            [prefix],
            padding=True,
            truncation=True,
            max_length=maximum,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"])
        length = int(input_ids.shape[-1])
        special_ids = frozenset(getattr(tokenizer, "all_special_ids", ()))
        if length and int(input_ids[0, -1]) in special_ids:
            length -= 1
        return length

    def process(
        artifacts: Sequence[str | Artifact],
        *,
        route: Route,
        seed: int | None,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        del seed
        texts = []
        for artifact in artifacts:
            if isinstance(artifact, Artifact):
                if artifact.modality != Modality.TEXT or not isinstance(
                    artifact.data, str
                ):
                    raise TypeError("text processors require inline text artifacts")
                texts.append(artifact.data)
            elif isinstance(artifact, str):
                texts.append(artifact)
            else:
                raise TypeError("text processors require strings or artifacts")
        prefix = route_prompt(route, prompt_name, prompt)
        encoded = tokenize(tuple(prefix + text for text in texts))
        try:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        except KeyError as error:
            raise KeyError(
                f"tokenizer output is missing required field {error.args[0]!r}"
            ) from error
        inputs = batch_builder(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
            token_type_ids=(
                None
                if encoded.get("token_type_ids") is None
                else jnp.asarray(encoded["token_type_ids"])
            ),
        )
        if include_prompt or not prefix:
            return inputs
        pooling_mask = np.array(attention_mask, copy=True)
        first_token = np.argmax(pooling_mask, axis=1)
        positions = np.arange(pooling_mask.shape[1])[None, :]
        pooling_mask[positions < (first_token + prompt_length(prefix))[:, None]] = 0
        if pooling_batch_builder is None:  # pragma: no cover - guarded above
            raise AssertionError("pooling batch builder disappeared")
        return pooling_batch_builder(inputs, jnp.asarray(pooling_mask))

    tokenizer_type = type(tokenizer)
    return Processor(
        process=process,
        contract={
            "schema_version": "representax-text-processor-v1",
            "tokenizer_type": (
                f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"
            ),
            "batch_builder": _callable_contract(batch_builder),
            "max_sequence_length": maximum,
            "sequence_length_buckets": list(lengths),
            "prompts": prompt_values,
            "default_prompt_name": default_prompt_name,
            "include_prompt": include_prompt,
        },
    )


__all__ = [
    "Processor",
    "make_audio_processor",
    "make_image_processor",
    "make_media_processor",
    "make_text_processor",
    "make_video_processor",
    "select_static_shape_bucket",
]
