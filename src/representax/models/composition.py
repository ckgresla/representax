"""Trainable model compositions used by representation tasks."""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from representax.core import Encoder

from .components import Linear


class EncoderPair(eqx.Module):
    """Two independently trainable encoders initialized from one checkpoint."""

    first: Any
    second: Any
    logit_scale: Float[Array, ""] | None

    @classmethod
    def from_encoder(
        cls,
        encoder: Encoder,
        *,
        scale: float | None = None,
    ) -> EncoderPair:
        """Duplicate one initialization into two distinct PyTree branches."""

        if scale is not None and (not math.isfinite(scale) or scale <= 0):
            raise ValueError("encoder-pair scale must be finite and positive")
        return cls(
            first=encoder,
            second=encoder,
            logit_scale=(
                None
                if scale is None
                else jnp.asarray(math.log(scale), dtype=jnp.float32)
            ),
        )

    @property
    def inference_encoder(self) -> Encoder:
        """Return the second branch, matching the upstream inference policy."""

        return self.second


class PairClassifier(eqx.Module):
    """One representation encoder plus a trainable pair-classification head."""

    encoder: Any
    classifier: Linear

    @classmethod
    def init(
        cls,
        encoder: Encoder,
        *,
        feature_dimension: int,
        class_count: int,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> PairClassifier:
        if feature_dimension <= 0 or class_count <= 1:
            raise ValueError(
                "pair classifier requires positive features and >1 classes"
            )
        return cls(
            encoder=encoder,
            classifier=Linear.init(
                feature_dimension,
                class_count,
                key=key,
                scale=feature_dimension**-0.5,
                dtype=dtype,
                bias=True,
            ),
        )

    def classify(
        self,
        features: Float[Array, "batch feature"],
    ) -> Float[Array, "batch class"]:
        return self.classifier(features)


@runtime_checkable
class ReconstructionDecoder(Protocol):
    """Causal decoder conditioned on one encoder-memory sequence."""

    def decode(
        self,
        input_ids: Int[Array, "batch target"],
        *,
        encoder_memory: Float[Array, "batch source hidden"],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch target vocabulary"]: ...


class TokenReconstructionDecoder(eqx.Module):
    """Small native reference decoder satisfying the reconstruction protocol."""

    token_embeddings: Float[Array, "vocabulary hidden"]
    memory_projection: Linear
    output_projection: Linear

    @classmethod
    def init(
        cls,
        *,
        vocabulary_size: int,
        hidden_size: int,
        key: PRNGKeyArray,
        dtype: jnp.dtype = jnp.float32,
    ) -> TokenReconstructionDecoder:
        if vocabulary_size <= 1 or hidden_size <= 0:
            raise ValueError("reconstruction decoder dimensions must be positive")
        embedding_key, memory_key, output_key = jax.random.split(key, 3)
        scale = hidden_size**-0.5
        return cls(
            token_embeddings=scale
            * jax.random.normal(
                embedding_key,
                (vocabulary_size, hidden_size),
                dtype=dtype,
            ),
            memory_projection=Linear.init(
                hidden_size,
                hidden_size,
                key=memory_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
            output_projection=Linear.init(
                hidden_size,
                vocabulary_size,
                key=output_key,
                scale=scale,
                dtype=dtype,
                bias=True,
            ),
        )

    def decode(
        self,
        input_ids: Int[Array, "batch target"],
        *,
        encoder_memory: Float[Array, "batch source hidden"],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch target vocabulary"]:
        del key
        tokens = self.token_embeddings[input_ids]
        memory = jnp.mean(encoder_memory, axis=1)
        hidden = jnp.tanh(tokens + self.memory_projection(memory)[:, None, :])
        return self.output_projection(hidden)


class DenoisingAutoEncoder(eqx.Module):
    """Representation encoder plus a causal reconstruction decoder."""

    encoder: Any
    decoder: Any


__all__ = [
    "DenoisingAutoEncoder",
    "EncoderPair",
    "PairClassifier",
    "ReconstructionDecoder",
    "TokenReconstructionDecoder",
]
