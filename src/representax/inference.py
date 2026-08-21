"""Host-side preprocessing and batching over compiled native encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from representax.core import Modality, Route, encode
from representax.data import Artifact
from representax.models.sentence import SentenceBatch, SentenceEncoder


@eqx.filter_jit
def _compiled_encode(
    model: SentenceEncoder,
    batch: Any,
    route: Route,
) -> Float[Array, "batch representation"]:
    return encode(model, batch, route=route)


class SentenceTransformerProcessor:
    """Host tokenizer and batch builder bundled with a sentence model."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        max_sequence_length: int,
        prompts: Mapping[str, str] | None = None,
        default_prompt_name: str | None = None,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        prompts = {} if prompts is None else dict(prompts)
        if default_prompt_name is not None and default_prompt_name not in prompts:
            raise ValueError("default_prompt_name does not name a configured prompt")
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.prompts = MappingProxyType(prompts)
        self.default_prompt_name = default_prompt_name

    def data_contract(self) -> Mapping[str, Any]:
        """Return the serializable preprocessing semantics used for resume."""

        tokenizer_type = type(self.tokenizer)
        return {
            "schema_version": "representax-sentence-processor-v1",
            "tokenizer_type": (
                f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"
            ),
            "max_sequence_length": self.max_sequence_length,
            "prompts": dict(self.prompts),
            "default_prompt_name": self.default_prompt_name,
        }

    def _route_prompt(
        self,
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
                return self.prompts[prompt_name]
            except KeyError as error:
                raise KeyError(f"unknown prompt {prompt_name!r}") from error
        candidates: tuple[str, ...]
        if route is Route.QUERY:
            candidates = ("query",)
        elif route is Route.DOCUMENT:
            candidates = ("document", "passage", "corpus")
        elif self.default_prompt_name is not None:
            candidates = (self.default_prompt_name,)
        else:
            candidates = ()
        return next(
            (self.prompts[name] for name in candidates if name in self.prompts), ""
        )

    def _tokenize(self, texts: Sequence[str]) -> Mapping[str, np.ndarray]:
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="np",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("text processor must return a mapping")
        return {name: np.asarray(value) for name, value in encoded.items()}

    def _prompt_length(self, prompt: str) -> int:
        if not prompt:
            return 0
        encoded = self.tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"])
        length = int(input_ids.shape[-1])
        special_ids = frozenset(getattr(self.tokenizer, "all_special_ids", ()))
        if length and int(input_ids[0, -1]) in special_ids:
            length -= 1
        return length

    def batch(
        self,
        model: SentenceEncoder,
        artifacts: Sequence[str | Artifact],
        *,
        route: Route = Route.GENERIC,
        seed: int | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Batch strings or inline text artifacts into native model inputs."""

        del seed
        texts = []
        for artifact in artifacts:
            if isinstance(artifact, Artifact):
                if artifact.modality != Modality.TEXT or not isinstance(
                    artifact.data, str
                ):
                    raise TypeError("sentence processors require inline text artifacts")
                texts.append(artifact.data)
            elif isinstance(artifact, str):
                texts.append(artifact)
            else:
                raise TypeError("sentence processors require strings or artifacts")
        route = Route(route)
        prefix = self._route_prompt(route, prompt_name, prompt)
        encoded = self._tokenize(tuple(prefix + text for text in texts))
        try:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        except KeyError as error:
            raise KeyError(
                f"text processor output is missing required field {error.args[0]!r}"
            ) from error
        token_type_ids = encoded.get("token_type_ids")
        backbone_batch = model.make_batch(
            input_ids=jnp.asarray(input_ids),
            attention_mask=jnp.asarray(attention_mask),
            token_type_ids=(
                None if token_type_ids is None else jnp.asarray(token_type_ids)
            ),
        )
        if model.pooling.include_prompt or not prefix:
            return backbone_batch

        prompt_length = self._prompt_length(prefix)
        pooling_mask = np.array(attention_mask, copy=True)
        first_token = np.argmax(pooling_mask, axis=1)
        positions = np.arange(pooling_mask.shape[1])[None, :]
        pooling_mask[positions < (first_token + prompt_length)[:, None]] = 0
        return SentenceBatch(
            backbone_inputs=backbone_batch,
            pooling_mask=jnp.asarray(pooling_mask),
        )


class TextEmbeddingModel:
    """A native model bundle with a host-side sentence processor."""

    def __init__(
        self,
        *,
        model: SentenceEncoder,
        processor: Any,
        max_sequence_length: int,
        prompts: Mapping[str, str] | None = None,
        default_prompt_name: str | None = None,
        similarity_function: str = "cosine",
    ) -> None:
        if similarity_function not in ("cosine", "dot", "euclidean", "manhattan"):
            raise ValueError(
                f"unsupported similarity function: {similarity_function!r}"
            )
        self.model = model
        self.processor = SentenceTransformerProcessor(
            tokenizer=processor,
            max_sequence_length=max_sequence_length,
            prompts=prompts,
            default_prompt_name=default_prompt_name,
        )
        self.similarity_function = similarity_function

    def preprocess(
        self,
        texts: Sequence[str],
        *,
        route: Route = Route.GENERIC,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Turn host strings into a fixed-shape native token batch."""

        return self.processor.batch(
            self.model,
            texts,
            route=route,
            prompt_name=prompt_name,
            prompt=prompt,
        )

    def embed(
        self,
        inputs: str | Sequence[str],
        *,
        route: Route = Route.GENERIC,
        batch_size: int = 32,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Preprocess and encode strings with one fixed compiled batch signature."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        texts = (inputs,) if isinstance(inputs, str) else tuple(inputs)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("text embedding inputs must be strings")
        if not texts:
            return np.empty((0, self.model.metadata.output_dimension), np.float32)

        route = Route(route)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            real_size = len(chunk)
            if real_size < batch_size:
                chunk = chunk + ("",) * (batch_size - real_size)
            batch = self.preprocess(
                chunk,
                route=route,
                prompt_name=prompt_name,
                prompt=prompt,
            )
            output = _compiled_encode(self.model, batch, route)
            outputs.append(np.asarray(output[:real_size], dtype=np.float32))
        return np.concatenate(outputs, axis=0)

    def similarity(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        """Compute the configured pairwise similarity matrix on host arrays."""

        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
            raise ValueError("similarity inputs must be aligned rank-two embeddings")
        if self.similarity_function == "cosine":
            left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
            right = right / np.maximum(
                np.linalg.norm(right, axis=1, keepdims=True), 1e-12
            )
            return left @ right.T
        if self.similarity_function == "dot":
            return left @ right.T
        difference = left[:, None, :] - right[None, :, :]
        if self.similarity_function == "euclidean":
            return -np.linalg.norm(difference, axis=-1)
        return -np.sum(np.abs(difference), axis=-1)


def embed(
    model: TextEmbeddingModel,
    inputs: str | Sequence[str],
    *,
    route: Route = Route.GENERIC,
    batch_size: int = 32,
) -> np.ndarray:
    """Host-side functional spelling of :meth:`TextEmbeddingModel.embed`."""

    return model.embed(inputs, route=route, batch_size=batch_size)


__all__ = ["SentenceTransformerProcessor", "TextEmbeddingModel", "embed"]
