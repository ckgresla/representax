"""Native dense sentence-embedding composition over token-level backbones."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core import EncoderMetadata, Route

from .components import Linear, l2_normalize

PoolingMode = Literal[
    "cls",
    "max",
    "mean",
    "mean_sqrt_len_tokens",
    "weightedmean",
    "lasttoken",
]
DenseActivation = Literal["identity", "tanh", "relu", "gelu", "silu"]

POOLING_MODES: tuple[PoolingMode, ...] = (
    "cls",
    "max",
    "mean",
    "mean_sqrt_len_tokens",
    "weightedmean",
    "lasttoken",
)


def _gather_tokens(
    hidden: Float[Array, "batch sequence hidden"],
    indices: Int[Array, " batch"],
) -> Float[Array, "batch hidden"]:
    gather = indices[:, None, None]
    gather = jnp.broadcast_to(gather, (hidden.shape[0], 1, hidden.shape[2]))
    return jnp.take_along_axis(hidden, gather, axis=1)[:, 0]


class SentencePooling(eqx.Module):
    """Sentence Transformers-compatible padded-token pooling."""

    input_dimension: int = eqx.field(static=True)
    modes: tuple[PoolingMode, ...] = eqx.field(static=True)
    include_prompt: bool = eqx.field(static=True, default=True)

    def __post_init__(self) -> None:
        if self.input_dimension <= 0:
            raise ValueError("pooling input_dimension must be positive")
        if not self.modes:
            raise ValueError("pooling requires at least one mode")
        invalid = tuple(mode for mode in self.modes if mode not in POOLING_MODES)
        if invalid:
            raise ValueError(f"unsupported pooling modes: {invalid}")

    @property
    def output_dimension(self) -> int:
        return self.input_dimension * len(self.modes)

    def __call__(
        self,
        hidden: Float[Array, "batch sequence hidden"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
    ) -> Float[Array, "batch pooled"]:
        if hidden.ndim != 3:
            raise ValueError(
                "token embeddings must have shape [batch, sequence, hidden]"
            )
        if attention_mask.shape != hidden.shape[:2]:
            raise ValueError("attention mask must align with token embeddings")
        if hidden.shape[2] != self.input_dimension:
            raise ValueError(
                f"pooling expected hidden dimension {self.input_dimension}; "
                f"received {hidden.shape[2]}"
            )

        valid = attention_mask.astype(bool)
        mask = valid[..., None].astype(hidden.dtype)
        outputs: list[Float[Array, "batch hidden"]] = []
        mean_sum: Float[Array, "batch hidden"] | None = None
        mean_count: Float[Array, "batch hidden"] | None = None

        for mode in self.modes:
            if mode == "cls":
                first = jnp.argmax(valid.astype(jnp.int32), axis=1)
                outputs.append(_gather_tokens(hidden, first))
            elif mode == "max":
                outputs.append(
                    jnp.max(jnp.where(valid[..., None], hidden, -jnp.inf), axis=1)
                )
            elif mode in ("mean", "mean_sqrt_len_tokens"):
                if mean_sum is None or mean_count is None:
                    mean_sum = jnp.sum(hidden * mask, axis=1)
                    count = jnp.sum(mask, axis=1)
                    mean_count = jnp.maximum(count, jnp.asarray(1e-9, count.dtype))
                divisor = (
                    mean_count
                    if mode == "mean"
                    else jnp.sqrt(mean_count.astype(jnp.float32)).astype(hidden.dtype)
                )
                outputs.append(mean_sum / divisor)
            elif mode == "weightedmean":
                weights = jnp.arange(
                    1,
                    hidden.shape[1] + 1,
                    dtype=hidden.dtype,
                )[None, :, None]
                weighted_mask = mask * weights
                total = jnp.sum(hidden * weighted_mask, axis=1)
                divisor = jnp.maximum(
                    jnp.sum(weighted_mask, axis=1),
                    jnp.asarray(1e-9, hidden.dtype),
                )
                outputs.append(total / divisor)
            elif mode == "lasttoken":
                reversed_valid = valid[:, ::-1]
                found = jnp.max(reversed_valid, axis=1)
                reversed_index = jnp.argmax(reversed_valid.astype(jnp.int32), axis=1)
                last = hidden.shape[1] - reversed_index - 1
                last = jnp.where(found, last, hidden.shape[1] - 1)
                outputs.append(_gather_tokens(hidden * mask, last))
            else:  # pragma: no cover - guarded by construction
                raise AssertionError(f"unreachable pooling mode: {mode}")

        return jnp.concatenate(outputs, axis=-1)


class SentenceDense(eqx.Module):
    """One serialized Sentence Transformers Dense postprocessor."""

    linear: Linear
    activation: DenseActivation = eqx.field(static=True)

    @property
    def output_dimension(self) -> int:
        return self.linear.weight.shape[0]

    def __call__(
        self,
        value: Float[Array, "batch input"],
    ) -> Float[Array, "batch output"]:
        value = self.linear(value)
        if self.activation == "identity":
            return value
        if self.activation == "tanh":
            return jnp.tanh(value)
        if self.activation == "relu":
            return jax.nn.relu(value)
        if self.activation == "gelu":
            return jax.nn.gelu(value, approximate=False)
        if self.activation == "silu":
            return jax.nn.silu(value)
        raise ValueError(f"unsupported dense activation: {self.activation!r}")


class SentenceNormalize(eqx.Module):
    """Unit-normalize one dense representation."""

    def __call__(
        self,
        value: Float[Array, "batch hidden"],
    ) -> Float[Array, "batch hidden"]:
        return l2_normalize(value)


SentencePostprocessor = SentenceDense | SentenceNormalize


class SentenceBatch(eqx.Module):
    """Backbone inputs plus an optional prompt-aware pooling mask."""

    backbone_inputs: Any
    pooling_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"]


class SentenceEncoder(eqx.Module):
    """A token backbone plus serialized dense sentence modules."""

    backbone: Any
    pooling: SentencePooling
    postprocessors: tuple[SentencePostprocessor, ...]
    metadata: EncoderMetadata
    truncate_dimension: int | None = eqx.field(static=True, default=None)

    def make_batch(
        self,
        *,
        input_ids: Int[Array, "batch sequence"],
        attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
        token_type_ids: Int[Array, "batch sequence"] | None = None,
    ) -> Any:
        """Delegate host token-batch construction to the native backbone."""

        builder = getattr(self.backbone, "make_batch", None)
        if not callable(builder):
            raise TypeError("sentence backbones must implement make_batch")
        return builder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    def encode(
        self,
        inputs: Any,
        *,
        route: Route,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch representation"]:
        del route
        hidden_states = getattr(self.backbone, "hidden_states", None)
        if not callable(hidden_states):
            raise TypeError("sentence backbones must implement hidden_states")
        if isinstance(inputs, SentenceBatch):
            backbone_inputs = inputs.backbone_inputs
            attention_mask = inputs.pooling_mask
        else:
            backbone_inputs = inputs
            attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is None:
            raise TypeError("sentence inputs must expose attention_mask")
        value = self.pooling(hidden_states(backbone_inputs, key=key), attention_mask)
        for module in self.postprocessors:
            value = module(value)
        if self.truncate_dimension is not None:
            value = value[:, : self.truncate_dimension]
        return value


__all__ = [
    "DenseActivation",
    "POOLING_MODES",
    "PoolingMode",
    "SentenceBatch",
    "SentenceDense",
    "SentenceEncoder",
    "SentenceNormalize",
    "SentencePooling",
    "SentencePostprocessor",
]
