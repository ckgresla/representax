"""Small reusable numerical components for native model families."""

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from representax.core.sharding import (
    activation_out_sharding,
    constrain_activation,
    replicate,
)
from representax.planning import RematerializationPolicy

AttentionImplementation = Literal["xla", "cudnn"]
Activation = Literal["gelu", "gelu_new", "relu", "silu"]


def rematerialize(function: Any, policy: RematerializationPolicy) -> Any:
    """Apply one stable activation-rematerialization policy to a layer."""

    if policy == "none":
        return function
    if policy == "selective":
        checkpoint_policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    elif policy == "full":
        checkpoint_policy = jax.checkpoint_policies.nothing_saveable
    else:
        raise ValueError("rematerialization must be 'none', 'selective', or 'full'")
    return jax.checkpoint(
        function,
        policy=checkpoint_policy,
        prevent_cse=False,
    )


def embedding_lookup(
    table: Float[Array, "vocabulary hidden"],
    indices: Int[Array, "*batch"],
) -> Float[Array, "*batch hidden"]:
    """Gather embeddings; repeated-token gradients accumulate naturally."""

    output = table.at[indices].get(
        out_sharding=activation_out_sharding(indices.ndim + 1)
    )
    return constrain_activation(output)


class Linear(eqx.Module):
    """Batched linear projection with Hugging Face weight orientation."""

    weight: Float[Array, "output input"]
    bias: Float[Array, " output"] | None = None

    @classmethod
    def init(
        cls,
        input_size: int,
        output_size: int,
        *,
        key: PRNGKeyArray,
        scale: float,
        dtype: jnp.dtype,
        bias: bool = False,
    ) -> Linear:
        weight = scale * jax.random.normal(key, (output_size, input_size), dtype=dtype)
        return cls(
            weight=weight,
            bias=jnp.zeros((output_size,), dtype=dtype) if bias else None,
        )

    def __call__(
        self,
        value: Float[Array, "*batch input"],
    ) -> Float[Array, "*batch output"]:
        weight = replicate(self.weight)
        output = jnp.matmul(
            value,
            weight.T,
            out_sharding=activation_out_sharding(value.ndim),
        )
        if self.bias is not None:
            output = output + replicate(self.bias)
        return constrain_activation(output)


class LayerNorm(eqx.Module):
    """Last-axis layer normalization with FP32 statistics."""

    weight: Float[Array, " hidden"]
    bias: Float[Array, " hidden"] | None
    epsilon: float = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        size: int,
        *,
        epsilon: float,
        dtype: jnp.dtype,
        bias: bool = False,
    ) -> LayerNorm:
        return cls(
            weight=jnp.ones((size,), dtype=dtype),
            bias=jnp.zeros((size,), dtype=dtype) if bias else None,
            epsilon=epsilon,
        )

    def __call__(
        self,
        value: Float[Array, "*batch hidden"],
    ) -> Float[Array, "*batch hidden"]:
        source_dtype = value.dtype
        value = value.astype(jnp.float32)
        mean = jnp.mean(value, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(value - mean), axis=-1, keepdims=True)
        output = (value - mean) * jax.lax.rsqrt(variance + self.epsilon)
        output = output * replicate(self.weight).astype(jnp.float32)
        if self.bias is not None:
            output = output + replicate(self.bias).astype(jnp.float32)
        return output.astype(source_dtype)


class RMSNorm(eqx.Module):
    """Last-axis root-mean-square normalization with FP32 statistics."""

    weight: Float[Array, " hidden"]
    epsilon: float = eqx.field(static=True)

    def __call__(
        self,
        value: Float[Array, "*batch hidden"],
    ) -> Float[Array, "*batch hidden"]:
        source_dtype = value.dtype
        value = value.astype(jnp.float32)
        inverse_rms = jax.lax.rsqrt(
            jnp.mean(jnp.square(value), axis=-1, keepdims=True) + self.epsilon
        )
        output = value * inverse_rms * replicate(self.weight).astype(jnp.float32)
        return output.astype(source_dtype)


def activate(
    value: Float[Array, "*batch hidden"],
    activation: Activation,
) -> Float[Array, "*batch hidden"]:
    """Apply an explicitly named Transformers-compatible activation."""

    if activation == "gelu":
        return jax.nn.gelu(value, approximate=False)
    if activation == "gelu_new":
        return jax.nn.gelu(value, approximate=True)
    if activation == "relu":
        return jax.nn.relu(value)
    if activation == "silu":
        return jax.nn.silu(value)
    raise ValueError(f"unsupported activation {activation!r}")


def dropout(
    value: Float[Array, "*batch"],
    probability: float,
    *,
    key: PRNGKeyArray | None,
) -> Float[Array, "*batch"]:
    """Apply inverted dropout when a training key is supplied."""

    if key is None or probability == 0.0:
        return value
    if not 0.0 <= probability < 1.0:
        raise ValueError("dropout probability must be in [0, 1)")
    keep_probability = 1.0 - probability
    keep = jax.random.bernoulli(key, keep_probability, value.shape)
    return jnp.where(keep, value / keep_probability, 0.0)


def dot_product_attention(
    query: Float[Array, "batch target_sequence heads head"],
    key: Float[Array, "batch source_sequence heads head"],
    value: Float[Array, "batch source_sequence heads head"],
    *,
    attention_bias: Float[Array, "#batch #heads target_sequence source_sequence"]
    | None = None,
    attention_mask: Bool[Array, "#batch #heads target_sequence source_sequence"]
    | None = None,
    dropout_probability: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    local_window_size: tuple[int, int] | None = None,
    implementation: AttentionImplementation = "xla",
) -> Float[Array, "batch target_sequence heads head"]:
    """Run full/local attention, adding explicit probability dropout if needed."""

    if dropout_key is None or dropout_probability == 0.0:
        return jax.nn.dot_product_attention(
            query,
            key,
            value,
            bias=attention_bias,
            mask=attention_mask,
            local_window_size=local_window_size,
            implementation=implementation,
        )

    scores = jnp.einsum("bthd,bshd->bhts", query, key)
    scores = scores * jax.lax.rsqrt(jnp.asarray(query.shape[-1], scores.dtype))
    if attention_bias is not None:
        scores = scores + attention_bias.astype(scores.dtype)
    mask = attention_mask
    if local_window_size is not None:
        left, right = local_window_size
        target = jnp.arange(query.shape[1])[:, None]
        source = jnp.arange(key.shape[1])[None, :]
        local_mask = (source >= target - left) & (source <= target + right)
        mask = local_mask if mask is None else mask & local_mask
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
        value.dtype
    )
    probabilities = dropout(
        probabilities,
        dropout_probability,
        key=dropout_key,
    )
    return jnp.einsum("bhts,bshd->bthd", probabilities, value)


def mean_pool(
    hidden: Float[Array, "batch sequence hidden"],
    attention_mask: Bool[Array, "batch sequence"] | Int[Array, "batch sequence"],
) -> Float[Array, "batch hidden"]:
    """Mean valid token states in FP32."""

    hidden = hidden.astype(jnp.float32)
    mask = attention_mask.astype(bool)[..., None]
    total = jnp.sum(jnp.where(mask, hidden, 0.0), axis=1)
    count = jnp.maximum(jnp.sum(mask, axis=1), 1)
    return total / count


def l2_normalize(
    value: Float[Array, "*batch hidden"],
) -> Float[Array, "*batch hidden"]:
    """Normalize the final axis in FP32 with a finite zero-vector result."""

    value = value.astype(jnp.float32)
    norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
    return value / jnp.maximum(norm, jnp.asarray(1e-12, value.dtype))


__all__ = [
    "Activation",
    "AttentionImplementation",
    "LayerNorm",
    "Linear",
    "RMSNorm",
    "activate",
    "dot_product_attention",
    "dropout",
    "embedding_lookup",
    "l2_normalize",
    "mean_pool",
    "rematerialize",
]
