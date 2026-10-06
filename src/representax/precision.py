"""Explicit numerical precision boundaries for compiled representation learning."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

TreeT = TypeVar("TreeT")


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Resolved dtypes for stored state, forward compute, and objectives."""

    parameter_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    activation_dtype: jnp.dtype
    matrix_dtype: jnp.dtype
    accumulation_dtype: jnp.dtype
    loss_dtype: jnp.dtype

    @property
    def communication_dtype(self) -> jnp.dtype:
        """Dtype communicated when transient compute parameters are sharded."""

        return self.compute_dtype


FP32_POLICY = PrecisionPolicy(
    parameter_dtype=jnp.dtype(jnp.float32),
    compute_dtype=jnp.dtype(jnp.float32),
    activation_dtype=jnp.dtype(jnp.float32),
    matrix_dtype=jnp.dtype(jnp.float32),
    accumulation_dtype=jnp.dtype(jnp.float32),
    loss_dtype=jnp.dtype(jnp.float32),
)

_DTYPES = {
    "float32": jnp.dtype(jnp.float32),
    "bfloat16": jnp.dtype(jnp.bfloat16),
    "float8_e4m3fn": jnp.dtype(jnp.float8_e4m3fn),
}
_ACTIVE_PRECISION: ContextVar[PrecisionPolicy | None] = ContextVar(
    "representax_precision",
    default=None,
)


def resolve_precision_policy(config: Any) -> PrecisionPolicy:
    """Resolve a validated serializable config into JAX dtype objects."""

    return PrecisionPolicy(
        parameter_dtype=_DTYPES[config.parameter_dtype],
        compute_dtype=_DTYPES[config.compute_dtype],
        activation_dtype=_DTYPES[config.activation_dtype],
        matrix_dtype=_DTYPES[config.resolved_matrix_dtype],
        accumulation_dtype=_DTYPES[config.accumulation_dtype],
        loss_dtype=_DTYPES[config.loss_dtype],
    )


@contextmanager
def precision_context(policy: PrecisionPolicy) -> Iterator[None]:
    """Make one static policy visible at model/objective call boundaries."""

    token = _ACTIVE_PRECISION.set(policy)
    try:
        yield
    finally:
        _ACTIVE_PRECISION.reset(token)


def active_precision_policy() -> PrecisionPolicy | None:
    """Return the policy captured by the current traced computation."""

    return _ACTIVE_PRECISION.get()


def cast_floating_tree(tree: TreeT, dtype: jnp.dtype) -> TreeT:
    """Cast only floating or complex array leaves in an arbitrary PyTree."""

    return jax.tree.map(
        lambda value: value.astype(dtype) if eqx.is_inexact_array(value) else value,
        tree,
    )


def prepare_master_model(
    model: TreeT,
    policy: PrecisionPolicy,
    *,
    trainable_filter: Any = eqx.is_inexact_array,
) -> TreeT:
    """Cast trainable leaves to master dtype without expanding frozen state."""

    selected = (
        jax.tree.map(trainable_filter, model)
        if callable(trainable_filter)
        else trainable_filter
    )
    if jax.tree.structure(model) != jax.tree.structure(selected):
        raise ValueError("trainable filter must match the model PyTree")
    return jax.tree.map(
        lambda value, trainable: (
            value.astype(policy.parameter_dtype)
            if trainable and eqx.is_inexact_array(value)
            else value
        ),
        model,
        selected,
    )


def model_for_compute(model: TreeT, policy: PrecisionPolicy) -> TreeT:
    """Create the transient model view consumed by the forward program."""

    return cast_floating_tree(model, policy.compute_dtype)


def active_model_for_compute(model: TreeT) -> TreeT:
    """Create a compute view at the model-use site inside the transformed graph."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return model
    return model_for_compute(model, policy)


def compute_parameter(value: Float[Array, ...]) -> Float[Array, ...]:
    """Cast a parameter used outside the shared encoder entry point."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.compute_dtype)


def active_compute_dtype(fallback: Any) -> jnp.dtype:
    """Resolve a model's inference default against the active training policy."""

    policy = _ACTIVE_PRECISION.get()
    return jnp.dtype(fallback) if policy is None else policy.compute_dtype


def _scaled_quantize(
    value: Float[Array, ...],
    dtype: jnp.dtype,
) -> tuple[Float[Array, ...], Float[Array, ""]]:
    """Dynamically scale one operand into a finite low-precision range."""

    numeric = value.astype(jnp.float32)
    maximum = jnp.asarray(jnp.finfo(dtype).max, dtype=jnp.float32)
    scale = jnp.maximum(
        jnp.max(jnp.abs(numeric)) / maximum,
        jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32),
    )
    scale = jax.lax.stop_gradient(scale)
    return (numeric / scale).astype(dtype), scale


def _scaled_fp8_dot(
    left: Float[Array, "left contract"],
    right: Float[Array, "contract right"],
    *,
    left_dtype: jnp.dtype,
    right_dtype: jnp.dtype,
) -> Float[Array, "left right"]:
    """Run one scaled FP8 matrix product with an exact CPU emulation lane."""

    quantized_left, left_scale = _scaled_quantize(left, left_dtype)
    quantized_right, right_scale = _scaled_quantize(right, right_dtype)
    if jax.default_backend() == "gpu":
        output = jax.lax.dot_general(
            quantized_left,
            quantized_right,
            (((1,), (0,)), ((), ())),
            precision=jax.lax.DotAlgorithmPreset.ANY_F8_ANY_F8_F32,
            preferred_element_type=jnp.float32,
        )
    else:
        output = jax.lax.dot_general(
            quantized_left.astype(jnp.bfloat16),
            quantized_right.astype(jnp.bfloat16),
            (((1,), (0,)), ((), ())),
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
            preferred_element_type=jnp.float32,
        )
    return output * left_scale * right_scale


@jax.custom_vjp
def _fp8_linear_dot(
    left: Float[Array, "left contract"],
    right: Float[Array, "contract right"],
) -> Float[Array, "left right"]:
    """Scaled E4M3 linear product with an E5M2 backward program."""

    return _scaled_fp8_dot(
        left,
        right,
        left_dtype=jnp.dtype(jnp.float8_e4m3fn),
        right_dtype=jnp.dtype(jnp.float8_e4m3fn),
    )


def _fp8_linear_dot_forward(
    left: Float[Array, "left contract"],
    right: Float[Array, "contract right"],
) -> tuple[Float[Array, "left right"], tuple[jax.Array, jax.Array]]:
    output = _scaled_fp8_dot(
        left,
        right,
        left_dtype=jnp.dtype(jnp.float8_e4m3fn),
        right_dtype=jnp.dtype(jnp.float8_e4m3fn),
    )
    return output, (left, right)


def _fp8_linear_dot_backward(
    residuals: tuple[jax.Array, jax.Array],
    cotangent: Float[Array, "left right"],
) -> tuple[jax.Array, jax.Array]:
    left, right = residuals
    gradient_left = _scaled_fp8_dot(
        cotangent,
        right.T,
        left_dtype=jnp.dtype(jnp.float8_e5m2),
        right_dtype=jnp.dtype(jnp.float8_e4m3fn),
    )
    gradient_right = _scaled_fp8_dot(
        left.T,
        cotangent,
        left_dtype=jnp.dtype(jnp.float8_e4m3fn),
        right_dtype=jnp.dtype(jnp.float8_e5m2),
    )
    return gradient_left.astype(left.dtype), gradient_right.astype(right.dtype)


_fp8_linear_dot.defvjp(_fp8_linear_dot_forward, _fp8_linear_dot_backward)


def linear_matmul(
    left: Float[Array, "*batch contract"],
    right: Float[Array, "contract output"],
    *,
    out_sharding: Any = None,
) -> Float[Array, "*batch output"]:
    """Apply the active matrix policy to a batched linear projection."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None or policy.matrix_dtype != jnp.dtype(jnp.float8_e4m3fn):
        return jnp.matmul(left, right, out_sharding=out_sharding)
    leading_shape = left.shape[:-1]
    flattened = left.reshape((-1, left.shape[-1]))
    output = _fp8_linear_dot(flattened, right)
    output = output.reshape((*leading_shape, right.shape[-1]))
    return output.astype(policy.activation_dtype)


def linear_projection(
    value: Float[Array, "*batch input"],
    weight: Float[Array, "output input"],
    *,
    out_sharding: Any = None,
) -> Float[Array, "*batch output"]:
    """Project through an ``[output, input]`` weight without transposing it."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None or policy.matrix_dtype != jnp.dtype(jnp.float8_e4m3fn):
        return jnp.einsum(
            "...i,oi->...o",
            value,
            weight,
            out_sharding=out_sharding,
        )
    leading_shape = value.shape[:-1]
    flattened = value.reshape((-1, value.shape[-1]))
    output = _fp8_linear_dot(flattened, weight.T)
    output = output.reshape((*leading_shape, weight.shape[-2]))
    return output.astype(policy.activation_dtype)


def activation_inputs(inputs: Any) -> Any:
    """Cast model inputs without touching labels or other task-owned values."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return inputs
    return cast_floating_tree(inputs, policy.activation_dtype)


def objective_output(
    value: Float[Array, ...],
) -> Float[Array, ...]:
    """Cross from model activations into FP32-sensitive task computation."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.accumulation_dtype)


def loss_value(value: Float[Array, ""]) -> Float[Array, ""]:
    """Apply the explicit scalar-loss dtype at the optimizer boundary."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return value
    return value.astype(policy.loss_dtype)


def accumulated_values(tree: Any) -> Any:
    """Cast floating metric leaves to the configured reduction dtype."""

    policy = _ACTIVE_PRECISION.get()
    if policy is None:
        return tree
    return cast_floating_tree(tree, policy.accumulation_dtype)


__all__ = [
    "FP32_POLICY",
    "PrecisionPolicy",
    "active_compute_dtype",
    "active_model_for_compute",
    "active_precision_policy",
    "accumulated_values",
    "activation_inputs",
    "cast_floating_tree",
    "compute_parameter",
    "loss_value",
    "linear_matmul",
    "linear_projection",
    "model_for_compute",
    "objective_output",
    "precision_context",
    "prepare_master_model",
    "resolve_precision_policy",
]
