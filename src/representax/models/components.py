"""Small reusable numerical components for native model families."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray


@jax.custom_vjp
def embedding_lookup(
    table: Float[Array, "vocabulary hidden"],
    indices: Int[Array, "*batch"],
) -> Float[Array, "*batch hidden"]:
    """Gather embeddings with an explicit dense table-gradient scatter."""

    return table[indices]


def _embedding_lookup_forward(
    table: Float[Array, "vocabulary hidden"],
    indices: Int[Array, "*batch"],
) -> tuple[
    Float[Array, "*batch hidden"],
    tuple[Float[Array, "vocabulary hidden"], Int[Array, "*batch"]],
]:
    return table[indices], (table, indices)


def _embedding_lookup_backward(
    residual: tuple[
        Float[Array, "vocabulary hidden"],
        Int[Array, "*batch"],
    ],
    cotangent: Float[Array, "*batch hidden"],
) -> tuple[Float[Array, "vocabulary hidden"], None]:
    table, indices = residual
    gradient = jnp.zeros_like(table).at[indices].add(cotangent)
    return jax.lax.optimization_barrier(gradient), None


embedding_lookup.defvjp(_embedding_lookup_forward, _embedding_lookup_backward)


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
        output = value @ self.weight.T
        return output if self.bias is None else output + self.bias


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
        output = output * self.weight.astype(jnp.float32)
        if self.bias is not None:
            output = output + self.bias.astype(jnp.float32)
        return output.astype(source_dtype)


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
    "LayerNorm",
    "Linear",
    "embedding_lookup",
    "l2_normalize",
    "mean_pool",
]
