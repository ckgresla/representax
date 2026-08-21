"""Host-side preprocessing and batching over compiled native encoders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import numpy as np
from jaxtyping import Array, Float

from representax.core import Route, encode
from representax.models.processing import Processor
from representax.models.sentence import SentenceEncoder


@eqx.filter_jit
def _compiled_encode(
    model: SentenceEncoder,
    batch: Any,
    route: Route,
) -> Float[Array, "batch representation"]:
    return encode(model, batch, route=route)


class TextEmbeddingModel:
    """A native sentence model paired with its generic host processor."""

    def __init__(
        self,
        *,
        model: SentenceEncoder,
        processor: Processor,
        similarity_function: str = "cosine",
    ) -> None:
        if similarity_function not in ("cosine", "dot", "euclidean", "manhattan"):
            raise ValueError(
                f"unsupported similarity function: {similarity_function!r}"
            )
        self.model = model
        self.processor = processor
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

        return self.processor(
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


__all__ = ["TextEmbeddingModel", "embed"]
