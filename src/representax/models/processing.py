"""Generic host preprocessing for native model implementations."""

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
            "batch_builder": _callable_name(batch_builder),
            "max_sequence_length": maximum,
            "sequence_length_buckets": list(lengths),
            "prompts": prompt_values,
            "default_prompt_name": default_prompt_name,
            "include_prompt": include_prompt,
        },
    )


__all__ = [
    "Processor",
    "make_text_processor",
    "select_static_shape_bucket",
]
